import os
import uuid
import yaml
import orjson
from yaml import CSafeLoader
import warnings
from typing import Dict, List
from mem0 import Memory
from sentence_transformers import SentenceTransformer, CrossEncoder

from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient

from logs.logging_setup import logger

# suppress verbose logging and deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MemoryLayer:
    _embedder = None
    _reranker = None

    def __init__(self):
        """
        initializes the memory layer with qdrant, mem0, and embedding models.
        
        """
        # use grpc connection to 127.0.0.1:6334 which is verified to work
        self.qdrant = QdrantClient(
            host="127.0.0.1",
            grpc_port=6334,
            prefer_grpc=True,
            check_compatibility=False
        )

        config_path = os.path.join(os.path.dirname(__file__), "config/sys_config.yaml")
        with open(config_path, "r") as f:
            full_config = yaml.load(f, Loader=CSafeLoader) or {}
        
        self.m_config = full_config.get("settings", {})
        config = full_config.get("memory", {})
        
        if "vector_store" not in config:
            config["vector_store"] = {"provider": "qdrant", "config": {}}
        config["vector_store"]["config"]["client"] = self.qdrant
        
        # baai/bge-small-en-v1.5 has a dimension of 384
        self.dims = config.get("embedder", {}).get("config", {}).get("embedding_dims", 384)
        self._ensure_collection("mem0", self.dims)
        self._ensure_collection("mem0_entities", self.dims)
        
        self.mem0 = Memory.from_config(config)

        if MemoryLayer._embedder is None:
            MemoryLayer._embedder = SentenceTransformer(
                config.get("embedder", {}).get("config", {}).get("model", "BAAI/bge-small-en-v1.5"), 
                trust_remote_code=True
            )
        self.embedder = MemoryLayer._embedder

        if MemoryLayer._reranker is None:
            reranker_cfg = config.get("reranker", {})
            model_name = reranker_cfg.get("config", {}).get("model") or reranker_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
            MemoryLayer._reranker = CrossEncoder(model_name)
        self.reranker = MemoryLayer._reranker

        self.collection = "agent_memory"
        self._ensure_collection(self.collection, self.dims)

        retrieval = config.get("retrieval_settings", {})
        self.rerank_limit = retrieval.get("rerank_limit", 15)
        self.embed_ratio = retrieval.get("embed_ratio", 3)
        self.cosine_threshold = retrieval.get("cosine_threshold", 0.7)
        self.rerank_threshold = retrieval.get("rerank_threshold", 0.7)
        
        # use archive.json as the single source of truth for sessions
        self.archive_path = os.path.join(os.path.dirname(__file__), "config/archive.json")
        self.max_sessions = self.m_config.get("max_sessions", 10)

    def _ensure_collection(self, name: str, size: int):
        """
        ensures that a qdrant collection exists, creating it if necessary.

        args:
            name (str): name of the collection.
            size (int): vector size for the collection.

        returns:
            none
        """
        if not self.qdrant.collection_exists(name):
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )

    def create_session(self, session_id: str):
        """
        explicitly initializes a session in archive and manages fifo limits.

        args:
            session_id (str): identifier for the new session.

        returns:
            none
        """
        if session_id in self.get_archive():
            self.clear_session(session_id)
        self._manage_sessions(session_id)

    def _manage_sessions(self, session_id: str):
        """
        implements fifo session management limited to 10 sessions using archive.json.

        args:
            session_id (str): identifier for the current session.

        returns:
            none
        """
        archive = {}
        if os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, "rb") as f:
                    archive = orjson.loads(f.read())
            except Exception:
                archive = {}

        if session_id in archive:
            return

        # new session id detected
        if len(archive) >= self.max_sessions:
            evicted_id = next(iter(archive))
            logger.info(f"fifo eviction: removing session '{evicted_id}' from archive and memory")
            
            # 1. remove from qdrant
            self.qdrant.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="session_id", match=MatchValue(value=evicted_id))
                    ]
                )
            )
            
            # 2. remove from mem0
            try:
                self.mem0.delete_all(user_id=evicted_id)
            except Exception as e:
                logger.warning(f"failed to delete mem0 data for {evicted_id}: {e}")
                pass
            
            # 3. remove from archive
            del archive[evicted_id]

        # initialize the new session in archive
        archive[session_id] = []

        # save updated archive
        with open(self.archive_path, "wb") as f:
            f.write(orjson.dumps(archive, option=orjson.OPT_INDENT_2))

    def store(self, text: str, session_id: str):
        """
        stores text into the vector database with session scoping.

        args:
            text (str): technical content to store.
            session_id (str): identifier for the current session.

        returns:
            none
        """
        self._manage_sessions(session_id)
        logger.info(f"storing technical record in session: {session_id}")
        point_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, text + session_id))
        vector = self.embedder.encode(text).tolist()

        self.qdrant.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id = point_id,
                    vector = vector,
                    payload = {"text": text, "session_id": session_id}
                )
            ]
        )

    def query(self, query: str, session_id: str, k: int = 4) -> str:
        """
        queries the vector database for relevant logs using reranking.

        args:
            query (str): search query.
            session_id (str): identifier for the session.
            k (int): number of top results to return.

        returns:
            str: formatted string of relevant documents.
        """
        self._manage_sessions(session_id)
        logger.info(f"querying technical logs for session: {session_id}")
        query_vec = self.embedder.encode(query).tolist()
        embed_limit = self.rerank_limit * self.embed_ratio

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit = embed_limit,
            query_filter=Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ]
            )
        ).points

        if not results:
            return ""

        filtered_results = [
            r for r in results 
            if r.score >= self.cosine_threshold
        ][:self.rerank_limit]

        if not filtered_results:
            return ""

        if len(filtered_results) <= k:
            ranked_texts = [r.payload.get("text", "") for r in filtered_results]
        else:
            texts = [r.payload.get("text", "") for r in filtered_results]
            scores = self.reranker.predict([[query, t] for t in texts])

            ranked = sorted(
                [
                    (t, float(score))
                    for t, score in zip(texts, scores)
                    if float(score) >= self.rerank_threshold
                ],
                key=lambda x: x[1],
                reverse=True
            )
            ranked_texts = [item[0] for item in ranked[:k]]

        if not ranked_texts:
            return ""
            
        output = "### RELEVANT AGENT MEMORY:\n"
        for i, text in enumerate(ranked_texts, 1):
            output += f"\n[DOCUMENT {i}]\n{text}\n"
        return output.strip()

    def get_context(self, query: str, session_id: str, k: int = 4) -> str:
        """
        retrieves and merges context from both mem0 (facts) and qdrant (logs).

        args:
            query (str): user's technical question or topic.
            session_id (str): identifier for the current session.
            k (int): number of log entries to retrieve.

        returns:
            str: merged context from facts and logs.
        """
        facts = self.recall(query, session_id=session_id)
        logs = self.query(query, session_id=session_id, k=k)
        
        context_parts = []
        if facts:
            context_parts.append(facts)
        if logs:
            context_parts.append(logs)
            
        return "\n\n".join(context_parts)

    def commit(self, query: str, report: str, session_id: str) -> None:
        """
        final commitment of a task: stores in both mem0 and vdb.

        args:
            query (str): the user's initial query.
            report (str): the final technical report or result.
            session_id (str): identifier for the session.

        returns:
            none
        """
        # 1. store the full report in vdb for future document-level retrieval
        self.store(f"USER_QUERY: {query}\nFINAL_REPORT:\n{report}", session_id=session_id)
        
        # 2. memorize in mem0 for entity/fact extraction
        self.memorize([
            {"role": "user", "content": query},
            {"role": "assistant", "content": report}
        ], session_id=session_id)

    def memorize(self, messages: list, session_id: str) -> None:
        """
        stores a conversation fragment into long-term fact memory (mem0).

        args:
            messages (list): list of message dictionaries.
            session_id (str): identifier for the session.

        returns:
            none
        """
        self._manage_sessions(session_id)
        self.mem0.add(messages, user_id=session_id)

    def recall(self, query: str, session_id: str) -> str:
        """
        recalls extracted facts and entities from long-term memory.

        args:
            query (str): search query for facts.
            session_id (str): identifier for the session.

        returns:
            str: formatted string of recalled memories.
        """
        self._manage_sessions(session_id)
        logger.info(f"recalling facts for session: {session_id}")
        results = self.mem0.search(query, filters={"user_id": session_id})
        if not results:
            return ""
        
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
            
        recalled_texts = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    recalled_texts.append(r.get("memory") or r.get("text") or "")
                else:
                    recalled_texts.append(str(r))
        else:
            recalled_texts.append(str(results))

        recalled_texts = [t for t in recalled_texts if t.strip()]
        if not recalled_texts:
            return ""

        output = "### RECALLED LONG-TERM MEMORY:\n"
        for i, text in enumerate(recalled_texts, 1):
            output += f"\n[MEMORY {i}]\n{text}\n"
        return output.strip()

    def get_archive(self) -> Dict[str, List[Dict[str, str]]]:
        """
        loads and returns the entire session archive.

        returns:
            dict: complete session archive mapping.
        """
        if os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, "rb") as f:
                    return orjson.loads(f.read())
            except Exception:
                return {}
        return {}

    def save_archive(self, archive: Dict[str, List[Dict[str, str]]]):
        """
        saves the provided archive to archive.json.

        args:
            archive (dict): session archive to persist.

        returns:
            none
        """
        safe_archive = {str(k): v for k, v in archive.items()}
        with open(self.archive_path, "wb") as f:
            f.write(orjson.dumps(safe_archive, option=orjson.OPT_INDENT_2))

    def clear_session(self, session_id: str):
        """
        purges memory data and archive for a specific session.

        args:
            session_id (str): session identifier to purge.

        returns:
            none
        """
        # 1. remove from qdrant
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id))
                ]
            )
        )
        # 2. remove from mem0
        try:
            self.mem0.delete_all(user_id=session_id)
        except Exception:
            pass
        
        # 3. remove from archive.json
        archive = self.get_archive()
        if session_id in archive:
            del archive[session_id]
            self.save_archive(archive)

    def purge_all(self):
        """
        purges all memory data, archive, and collections.

        returns:
            none
        """
        try:
            self.mem0.reset()
        except Exception:
            pass

        # explicitly purge and re-ensure collections
        collections = ["mem0", "mem0_entities", "agent_memory"]
        for name in collections:
            if self.qdrant.collection_exists(name):
                self.qdrant.delete_collection(name)
            self._ensure_collection(name, 384)
            
        # clear archive.json
        self.save_archive({})
