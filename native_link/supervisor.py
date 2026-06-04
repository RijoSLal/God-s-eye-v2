from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from typing import Dict, Optional
import asyncio
from structure import Task, PlanState, CreatePlanOutput, EditTaskOutput, ReplanOutput, Nuance, AnalystReport, SupervisorRunSchema, MemoryQuerySchema
from client import MCPClient
from memory import MemoryLayer
from langchain.messages import AIMessageChunk
from logs.logging_setup import logger

import os
import yaml
import orjson
from yaml import CSafeLoader
from datetime import datetime

# ============================================================
# model configuration
# ============================================================

class Model:
    # shared resources and task coordination
    _shared_memory = None
    _shared_mcp = None
    _terminal_queue = None
    _task_subscribers = []

    @property
    def terminal_queue(self):
        """
        initializes or retrieves the shared terminal queue.

        returns:
            asyncio.Queue: the shared terminal queue.
        """
        if Model._terminal_queue is None:
            try:
                asyncio.get_running_loop()
                Model._terminal_queue = asyncio.Queue()
            except RuntimeError:
                return None
        return Model._terminal_queue

    @classmethod
    def subscribe_tasks(cls) -> asyncio.Queue:
        """
        creates and registers a new task subscriber queue.

        returns:
            asyncio.Queue: the new subscriber queue.
        """
        q = asyncio.Queue()
        cls._task_subscribers.append(q)
        return q

    @classmethod
    def unsubscribe_tasks(cls, q: asyncio.Queue):
        """
        removes a task subscriber queue.

        args:
            q (asyncio.Queue): the queue to unsubscribe.

        returns:
            none
        """
        if q in cls._task_subscribers:
            cls._task_subscribers.remove(q)

    @classmethod
    def broadcast_status(cls, data: dict):
        """
        broadcasts generic status data to all task subscribers.

        args:
            data (dict): the status data to broadcast.

        returns:
            none
        """
        for q in cls._task_subscribers:
            q.put_nowait(data)

    def __init__(self):
        """
        initializes the base model with system configuration and shared clients.

        returns:
            none
        """
        # load system config
        config_path = os.path.join(os.path.dirname(__file__), "config/sys_config.yaml")
        with open(config_path, "r") as f:
            full_config = yaml.load(f, Loader=CSafeLoader) or {}

        self.m_config = full_config.get("settings", {})
        self.prompts = full_config.get("prompts", {})

        # create a fresh model instance for this component
        self.core_model = ChatOpenAI(
            model=self.m_config.get("model"),
            base_url=self.m_config.get("base_url"),
            temperature=self.m_config.get("temperature", 0.0),
            api_key=self.m_config.get("api_key")
        )
        
        # initialize shared clients if not present
        if Model._shared_mcp is None:
            logger.info("initializing shared mcp client")
            Model._shared_mcp = MCPClient()
        self.mcp_client = Model._shared_mcp

        if Model._shared_memory is None:
            logger.info("initializing shared memory layer")
            Model._shared_memory = MemoryLayer()
        self.memory = Model._shared_memory

    async def ping(self):
        """
        verifies connectivity to the model provider.

        returns:
            str: response from the model or 'offline'.
        """
        try:
            res = await self.core_model.ainvoke("ping", config={"max_tokens": 5})
            return res.content
        except Exception as e:
            logger.error(f"model ping failure: {e}")
            return "offline"

    def fetch_prompt(self, query: str, fallback: str = ""):
        """
        retrieves a prompt template from the system configuration.

        args:
            query (str): dot-notated key for the prompt.
            fallback (str): default value if key not found.

        returns:
            str: the prompt template.
        """
        segments = query.split('.')
        cursor = self.prompts
        for s in segments:
            if isinstance(cursor, dict):
                cursor = cursor.get(s)
            else:
                return fallback
            if cursor is None:
                return fallback
        return cursor or fallback


class Supervisor(Model):
    # manages high-level planning and task execution
    def __init__(self):
        """
        initializes the supervisor with structured generation models and recon agent.

        returns:
            none
        """
        super().__init__()
        from scripting import Recon
        # use separate model instances for structured generation to prevent leakage
        self.plan_model = ChatOpenAI(
            model=self.m_config.get("model"),
            base_url=self.m_config.get("base_url"),
            temperature=self.m_config.get("temperature", 0.0),
            api_key=self.m_config.get("api_key")
        )
        self.plan_gen = self.plan_model.with_structured_output(CreatePlanOutput)
        self.edit_gen = self.plan_model.with_structured_output(EditTaskOutput)
        self.replan_gen = self.plan_model.with_structured_output(ReplanOutput)
        
        self.max_edit = self.m_config.get("max_edit", 1)
        self.max_replan = self.m_config.get("max_replan", 2)
        self.recon = Recon()
        self.current_os = "unknown"

    async def create_plan(self, goal: str) -> Dict[str, Task]:
        """
        creates an initial execution plan for a goal.

        args:
            goal (str): the user's high-level goal.

        returns:
            Dict[str, Task]: a mapping of task identifiers to task objects.
        """
        logger.info(f"creating execution plan for goal: {goal[:50]}")
        prompt = self.fetch_prompt("supervisor.create_plan").replace("{goal}", goal)
        try:
            res = await self.plan_gen.ainvoke(prompt)
            # handle cases where the model might return the raw text if structured output fails
            if hasattr(res, "tasks"):
                tasks = res.tasks
            elif isinstance(res, dict) and "tasks" in res:
                tasks = res["tasks"]
            else:
                tasks = {}
        except Exception as e:
            logger.error(f"planning exception for goal '{goal[:50]}': {e}")
            tasks = {}
        
        if isinstance(tasks, dict) and tasks:
            sanitized = {}
            all_tids = set(tasks.keys())
            for tid, t in tasks.items():
                if tid == "error" or "invalid" in str(t.description).lower():
                    continue
                # ensure t is a task object
                if isinstance(t, dict):
                    t = Task(**t)
                
                # sanitize depends_on: only keep strings that are valid task ids
                if hasattr(t, "depends_on") and isinstance(t.depends_on, list):
                    t.depends_on = [d for d in t.depends_on if isinstance(d, str) and d in all_tids]
                
                # explicitly ensure fresh tasks start with null findings and failure reasons
                t.findings_summary = None
                t.failure_reason = None
                
                sanitized[tid] = t
            return sanitized
        return {}

    async def edit_task(self, goal: str, state_json: str, task_id: str, task_json: str) -> Task:
        """
        refines or revises a specific task based on context and failures.

        args:
            goal (str): the user's high-level goal.
            state_json (str): current global state in json format.
            task_id (str): identifier of the task being edited.
            task_json (str): the current task data in json format.

        returns:
            Task: the updated task object.
        """
        prompt = self.fetch_prompt("supervisor.edit_task") \
            .replace("{goal}", goal) \
            .replace("{state_json}", state_json) \
            .replace("{task_id}", task_id) \
            .replace("{task_json}", task_json)
        res = await self.edit_gen.ainvoke(prompt)
        return res.replacement_task

    async def replan(self, goal: str, state_json: str) -> Dict[str, Task]:
        """
        restructures the remaining plan when the workflow is blocked.

        args:
            goal (str): the user's high-level goal.
            state_json (str): current global state in json format.

        returns:
            Dict[str, Task]: the new set of tasks for the replan.
        """
        prompt = self.fetch_prompt("supervisor.replan") \
            .replace("{goal}", goal) \
            .replace("{state_json}", state_json)
        res = await self.replan_gen.ainvoke(prompt)
        return res.tasks

    def dependencies_satisfied(self, task: Task, tasks: Dict[str, Task]) -> bool:
        """
        checks if all dependencies for a task are completed.

        args:
            task (Task): the task to check.
            tasks (Dict[str, Task]): the global task mapping.

        returns:
            bool: true if dependencies are done, false otherwise.
        """
        return all(tasks.get(d) and tasks[d].status == "done" for d in task.depends_on)

    def is_blocked(self, tasks: Dict[str, Task]) -> bool:
        """
        detects if the workflow is blocked by skipped tasks.

        args:
            tasks (Dict[str, Task]): the global task mapping.

        returns:
            bool: true if blocked, false otherwise.
        """
        for _ , t in tasks.items():
            if t.status != "pending": continue
            for dep in t.depends_on:
                if tasks.get(dep) and tasks[dep].status == "skipped":
                    return True
        return False

    def merge_tasks(self, old_tasks: Dict[str, Task], new_tasks: Dict[str, Task]) -> Dict[str, Task]:
        """
        merges completed tasks with a newly generated plan.

        args:
            old_tasks (Dict[str, Task]): existing tasks from the previous plan.
            new_tasks (Dict[str, Task]): tasks from the new replan.

        returns:
            Dict[str, Task]: the merged task mapping.
        """
        merged = {}
        # keep done tasks
        for tid, t in old_tasks.items():
            if t.status == "done": merged[tid] = t
        # add new tasks
        for tid, t in new_tasks.items():
            if tid not in merged: merged[tid] = t
        # keep skipped tasks if not replaced
        for tid, t in old_tasks.items():
            if t.status == "skipped" and tid not in merged:
                merged[tid] = t
        return merged

    async def dispatch(self, task: Task, session_id: str) -> AnalystReport | str:
        """
        dispatches a task to the reconnaissance agent.

        args:
            task (Task): the task object to execute.
            session_id (str): identifier for the current session.

        returns:
            AnalystReport | str: result from the execution agent.
        """
        logger.info(f"dispatching task to recon agent: {task.description[:50]}")
        return await self.recon(task.description, session_id=session_id, os_info=self.current_os)

    async def __call__(self, goal: str, session_id: str) -> str:
        """
        entry point for executing a goal through the supervisor.

        args:
            goal (str): the user's high-level goal.
            session_id (str): identifier for the current session.

        returns:
            str: final execution log.
        """
        return await self.run(goal, session_id)

    async def run(self, goal: str, session_id: str) -> str:
        """
        orchestrates the entire planning and execution lifecycle.

        args:
            goal (str): the user's high-level goal.
            session_id (str): identifier for the current session.

        returns:
            str: final execution log.
        """
        # wipe machine_state.json completely for every new run (initial plan)
        self.wipe_state()
        
        # fetch remote os info for environmental awareness
        os_info = "unknown"
        try:
            ping_data = await self.mcp_client.ping()
            if isinstance(ping_data, dict):
                os_info = ping_data.get("os", "unknown")
        except Exception:
            pass
        
        self.current_os = os_info

        # inject mem0 facts into planning to make the supervisor environment-aware
        facts = self.memory.recall(goal, session_id=session_id)
        planning_goal = f"GOAL: {goal}\nOS: {self.current_os}"
        if facts:
            planning_goal += f"\n\nRECALLED_ENVIRONMENT_FACTS:\n{facts}"

        state = PlanState(goal=goal, tasks={})
        self.save_state(state)
        
        tasks = await self.create_plan(planning_goal)
        if not tasks:
            state.status = "failure"
            state.abort_reason = "planning failed"
            self.save_state(state)
            return self.generate_log(state)

        state.tasks = tasks
        self.save_state(state)
        replan_count = 0

        # main execution loop
        while state.status == "running":
            progress_made = False
            current_task_ids = list(state.tasks.keys())
            
            for tid in current_task_ids:
                task = state.tasks[tid]
                if task.status != "pending": continue
                if not self.dependencies_satisfied(task, state.tasks): 
                    continue
                
                task.status = "running"
                self.save_state(state)
                success = False
                
                # retry/edit loop
                for attempt in range(self.max_edit + 1):
                    if attempt > 0:
                        # inject facts into editing to help bypass obstacles
                        facts = self.memory.recall(task.description, session_id=session_id)
                        edit_goal = f"{state.goal}\n\nENVIRONMENT_CONTEXT:\n{facts}" if facts else state.goal
                        task = await self.edit_task(edit_goal, state.model_dump_json(), tid, task.model_dump_json())
                        state.tasks[tid] = task
                        self.save_state(state)

                    try:
                        res = await self.dispatch(task, session_id=session_id)
                        if hasattr(res, "status") and res.status == "done":
                            task.findings_summary = res.findings_summary
                            task.failure_reason = None
                            if hasattr(res, "nuances") and res.nuances:
                                for key, value in res.nuances.items():
                                    state.nuances[key] = Nuance(value=str(value), source_task=tid)
                            success = True
                            break
                        else:
                            reason = res.failure_reason if hasattr(res, "failure_reason") else str(res)
                            logger.warning(f"task '{tid}' reported failure: {reason}")
                            task.failure_reason = reason
                    except Exception as e:
                        logger.error(f"exception during dispatch of task '{tid}': {e}")
                        task.failure_reason = str(e)
                
                if success:
                    task.status = "done"
                    progress_made = True
                else:
                    task.status = "skipped"
                
                logger.info(f"task {tid} completed with status: {task.status}")
                if task.failure_reason:
                    logger.info(f"task {tid} failure reason: {task.failure_reason}")
                
                self.save_state(state)

            # check for completion or blockages
            if all(t.status == "done" for t in state.tasks.values()):
                state.status = "success"
                self.save_state(state)
                break
            
            if all(t.status in ["done", "skipped"] for t in state.tasks.values()):
                if not self.is_blocked(state.tasks):
                    state.status = "success"
                    state.abort_reason = "some tasks were skipped"
                    self.save_state(state)
                    break

            if self.is_blocked(state.tasks):
                if replan_count < self.max_replan:
                    replan_count += 1
                    # replan maintains the existing state file, just updating tasks
                    new_tasks = await self.replan(state.goal, state.model_dump_json())
                    state.tasks = self.merge_tasks(state.tasks, new_tasks)
                    self.save_state(state)
                    continue
                else:
                    state.status = "failure"
                    state.abort_reason = "max replans exceeded due to blockages"
                    self.save_state(state)
                    break

            if not progress_made:
                state.status = "failure"
                state.abort_reason = "stalled - no progress possible"
                self.save_state(state)
                break
        log = self.generate_log(state)
        print(f"-------------suprevisor----------------\n{log}")
        return log

    def generate_log(self, state: PlanState) -> str:
        """
        generates a clean, llm-friendly log of the execution flow from a planstate.

        args:
            state (PlanState): the final global state.

        returns:
            str: formatted execution log.
        """
        log = ["### EXECUTION LOG ###"]
        log.append(f"GOAL: {state.goal}")
        log.append(f"FINAL STATUS: {state.status.upper()}")
        if state.abort_reason:
            log.append(f"ABORT REASON: {state.abort_reason}")
        
        log.append("\n#### WORKFLOW FLOWCHART ####")
        
        task_ids = list(state.tasks.keys())
        for tid in task_ids:
            t = state.tasks[tid]
            log.append(f"- {tid} ({t.status}): {t.description}")
            if t.findings_summary:
                log.append(f"  > FINDINGS: {t.findings_summary}")
            if t.failure_reason:
                log.append(f"  > FAILURE: {t.failure_reason}")
        
        if state.nuances:
            log.append("\n#### EXTRACTED NUANCES ####")
            for key, nuance in state.nuances.items():
                log.append(f"- {key}: {nuance.value} (from {nuance.source_task})")
      
        return "\n".join(log)

    def wipe_state(self):
        """
        resets the global state management file and task queues.

        returns:
            none
        """
        state_path = os.path.join(os.path.dirname(__file__), "config/machine_state.json")
        with open(state_path, "wb") as f:
            f.write(b"{}")
        
        # clear all subscriber queues and broadcast a clear signal
        for q in self._task_subscribers:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    break

    def save_state(self, state: PlanState):
        """
        persists the current planstate to disk and broadcasts to subscribers.

        args:
            state (PlanState): the current global state object.

        returns:
            none
        """
        state_path = os.path.join(os.path.dirname(__file__), "config/machine_state.json")
        try:
            dump = state.model_dump()
            with open(state_path, "wb") as f:
                f.write(orjson.dumps(dump, option=orjson.OPT_INDENT_2))
            
            # broadcast to all real-time subscribers
            for q in self._task_subscribers:
                q.put_nowait(dump.get("tasks", {}))
        except Exception:
            pass


class Reporter(Model):
    # final agent responsible for synthesizing results and interacting with the user
    def __init__(self):
        """
        initializes the reporter agent with its supervisor.

        returns:
            none
        """
        super().__init__()
        self.supervisor = Supervisor()
        self.agent = None
        self.tools = []
        self.active_session = None
        self.max_iter = self.m_config.get("agents", {}).get("reporter_max_iter", 10)

    async def _ensure_agent(self):
        """
        ensures the synthetic agent is initialized with filtered mcp and internal tools.

        returns:
            none
        """
        if self.agent: return

        mcp_tools = await self.mcp_client.get_tools(["web_search"])
        for tool in mcp_tools:
            if tool.name == "web_search":
                tool.description = (
                    "Search the web for information. Use 'surface=True' for general factual queries, "
                    "news, and non-technical info. Use 'surface=False' ONLY for deep "
                    "technical investigations, coding issues, or security research."
                )
        
        self.tools = [
            StructuredTool.from_function(
                name="orchestrator",
                coroutine=self.supervisor_run,
                description="Run multi-step local system tasks or code investigations. Returns a status tree. Findings are committed to memory.",
                args_schema=SupervisorRunSchema
            ),
            StructuredTool.from_function(
                name="memory_query",
                coroutine=self.memory_query,
                description="Search internal memory for facts and findings from previous tasks.",
                args_schema=MemoryQuerySchema
            )
        ] + mcp_tools
        
        self.agent = create_agent(
            model=self.core_model,
            system_prompt=self.fetch_prompt("reporter.system_prompt"),
            tools=self.tools
        )

    async def memory_query(self, query: str) -> str:
        """
        queries internal document-level memory for context.

        args:
            query (str): the search query.

        returns:
            str: verification results from technical logs.
        """
        # tool call now focuses on document-level retrieval from qdrant
        return self.memory.query(query, session_id=self.active_session)

    async def supervisor_run(self, task: str) -> str:
        """
        executes a complex, multi-step task through the supervisor.

        args:
            task (str): multi-step local request to execute.

        returns:
            str: llm-friendly log of the execution flow.
        """
        return await self.supervisor.run(task, session_id=self.active_session)

    async def __call__(self, query: str, session_id: str):
        """
        processes a user query by coordinating facts, history, and the agent.

        args:
            query (str): the user's initial question.
            session_id (str): identifier for the current session.

        returns:
            asyncgenerator: yields text chunks and tool status updates.
        """
        print(f"\n--- reporter received query ---\n{query}\n")
        self.active_session = session_id
        await self._ensure_agent()
        
        # inject high-level facts (mem0) into the prompt context
        facts = self.memory.recall(query, session_id=session_id)

        archive_all = self.memory.get_archive()
        session_history = archive_all.get(session_id, [])

        messages = session_history + [{"role": "user", "content": query}]
        if facts:
            messages.insert(0, {"role": "system", "content": f"RECALLED FACTS:\n{facts}"})

        response = ""

        try:
            
            async for chunk in self.agent.astream({"messages": messages}, config={"max_iterations": self.max_iter}, stream_mode="messages", version="v3"):
                # chunk is (msg, metadata)
         
                if not isinstance(chunk, tuple):
                    continue
                
                msg, metadata = chunk
                
                # yield tool calls for the ui status bar
                if metadata.get("langgraph_node") == "agent":
                    # tool_calls = getattr(msg, "tool_calls", [])
                    # if tool_calls:
                    #     for tc in tool_calls:
                    #         yield {"tool": tc["name"], "args": tc["args"]}
                    continue

                # 1. we only want messages from the agent/model node, not the tools node
                if metadata.get("langgraph_node") == "tools":
                    continue
                
                # 2. we only want ai messages (or chunks)
                if "AIMessage" not in msg.__class__.__name__:
                    continue
                
                # 3. skip messages that contain tool calls
                if getattr(msg, "tool_calls", []):
                    continue
                    
                # 4. extract and yield content
                content = getattr(msg, "content", "")
                text = ""
                if isinstance(content, str): 
                    text = content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text": 
                            text += part.get("text", "")
                        elif isinstance(part, str): 
                            text += part
                
                if text:
                    response += text
                    yield text
                            
        except Exception as e:
            logger.error(f"streaming error in session {session_id}: {e}")
            yield f"\n[streaming error] {str(e)}"

        archive_all = self.memory.get_archive()
        if session_id not in archive_all: archive_all[session_id] = []
        archive_all[session_id].append({"role": "user", "content": query})
        archive_all[session_id].append({"role": "assistant", "content": response})
        if len(archive_all[session_id]) > 10: archive_all[session_id] = archive_all[session_id][-10:]
        self.memory.save_archive(archive_all)

    def clear_session(self, session_id: Optional[str] = None):
        """
        purges memory and state data for a specific session.

        args:
            session_id (Optional[str]): session identifier to purge.

        returns:
            none
        """
        target_session = session_id or self.active_session
        # wipe the global plan state whenever a session is cleared
        self.supervisor.wipe_state()
        self.memory.clear_session(target_session)

    def purge_all(self):
        """
        purges all memory data and global state.

        returns:
            none
        """
        # wipe the global plan state during a total purge
        self.supervisor.wipe_state()
        self.memory.purge_all()


