from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.tools import StructuredTool
from typing import Dict, Optional
import asyncio
from structure import Task, PlanState, CreatePlanOutput, EditTaskOutput, ReplanOutput, Nuance, AnalystReport, OrchestratorRunSchema, MemoryQuerySchema
from mcp_client import MCPClient
from memory import MemoryLayer
from logs.logging_setup import logger

import os
import yaml
import orjson
from yaml import CSafeLoader


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


class Orchestrator(Model):
    # manages high-level planning and task execution
    def __init__(self):
        """
        initializes the orchestrator with structured generation models and recon agent.

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

    async def create_plan(self, goal: str, facts: str = "") -> Dict[str, Task]:
        """
        creates an initial execution plan for a goal.

        args:
            goal (str): the user's high-level goal.
            facts (str): recalled environment facts.

        returns:
            Dict[str, Task]: a mapping of task identifiers to task objects.
        """
        logger.info(f"creating execution plan for goal: {goal[:50]}")
        prompt = self.fetch_prompt("orchestrator.create_plan").format(
            goal=goal,
            os_info=self.current_os,
            facts=facts
        )
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

    async def edit_task(self, goal: str, history: str, task_id: str, task: Task, all_valid_tids: set, facts: str = "") -> Task:
        """
        refines or revises a specific task based on context and failures.

        args:
            goal (str): the user's high-level goal.
            history (str): execution log history in tree format.
            task_id (str): identifier of the task being edited.
            task (Task): the current task object.
            all_valid_tids (set): set of all valid task IDs in the current state.
            facts (str): recalled environment facts.

        returns:
            Task: the updated task object.
        """
        prompt = self.fetch_prompt("orchestrator.edit_task").format(
            goal=goal,
            os_info=self.current_os,
            history=history,
            task_id=task_id,
            task_description=task.description,
            success_criteria=task.expected_output,
            facts=facts
        )
        res = await self.edit_gen.ainvoke(prompt)
        rt = res.replacement_task
        if isinstance(rt, dict):
            rt = Task(**rt)
        
        # ensure it doesn't hallucinate dependencies that don't exist in the current global state
        if hasattr(rt, "depends_on") and isinstance(rt.depends_on, list):
            rt.depends_on = [d for d in rt.depends_on if isinstance(d, str) and d in all_valid_tids]
        
        # reset findings for the new attempt
        rt.findings_summary = None
        rt.failure_reason = None
        return rt

    async def replan(self, goal: str, history: str, existing_tids: set, facts: str = "") -> Dict[str, Task]:
        """
        restructures the remaining plan when the workflow is blocked.

        args:
            goal (str): the user's high-level goal.
            history (str): execution log history in tree format.
            existing_tids (set): set of IDs from tasks that are already 'done'.
            facts (str): recalled environment facts.

        returns:
            Dict[str, Task]: the new set of tasks for the replan.
        """
        prompt = self.fetch_prompt("orchestrator.replan").format(
            goal=goal,
            os_info=self.current_os,
            history=history,
            facts=facts
        )
        res = await self.replan_gen.ainvoke(prompt)
        tasks = res.tasks if hasattr(res, "tasks") else {}
        
        sanitized = {}
        if isinstance(tasks, dict):
            # allow dependencies on new tasks OR tasks that were already completed
            all_valid_tids = set(tasks.keys()) | existing_tids
            for tid, t in tasks.items():
                if isinstance(t, dict):
                    t = Task(**t)
                if hasattr(t, "depends_on") and isinstance(t.depends_on, list):
                    t.depends_on = [d for d in t.depends_on if isinstance(d, str) and d in all_valid_tids]
                t.findings_summary = None
                t.failure_reason = None
                sanitized[tid] = t
        return sanitized

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

    async def dispatch(self, task: Task, state: PlanState, session_id: str) -> AnalystReport | str:
        """
        dispatches a task to the reconnaissance agent with a structured ASCII tree of dependencies.

        args:
            task (Task): the task object to execute.
            state (PlanState): the current global plan state.
            session_id (str): identifier for the current session.

        returns:
            AnalystReport | str: result from the execution agent.
        """
        prompt_lines = [f"### TASK: {task.description}"]
        
        if task.depends_on:
            prompt_lines.append("\n### DEPENDENCY HISTORY")
            prompt_lines.append("The following technical context was gathered from parent tasks:\n")
            
            for i, dep_id in enumerate(task.depends_on):
                dep_task = state.tasks.get(dep_id)
                if not dep_task:
                    continue
                
                is_last = (i == len(task.depends_on) - 1)
                prefix = "└── " if is_last else "┌── "
                pipe = "    " if is_last else "│   "
                
                prompt_lines.append(f"{prefix}{dep_id} -> {dep_task.description}")
                
                # findings
                findings = dep_task.findings_summary or "No findings recorded."
                # nuances for this task
                dep_nuances = {k: n.value for k, n in state.nuances.items() if n.source_task == dep_id}
                
                if dep_nuances:
                    prompt_lines.append(f"{pipe}├── Findings: {findings}")
                    # format nuances as string key: value
                    nuance_str = ", ".join([f"{k}: {v}" for k, v in dep_nuances.items()])
                    prompt_lines.append(f"{pipe}└── Information:     {nuance_str}")
                else:
                    prompt_lines.append(f"{pipe}└── Findings: {findings}")
                
                if not is_last:
                    prompt_lines.append("│")

        prompt = "\n".join(prompt_lines)
        logger.info(f"DISPATCHING TASK TO RECON AGENT:\n{prompt}")
        return await self.recon(prompt, session_id=session_id, os_info=self.current_os)


    async def __call__(self, goal: str, session_id: str) -> str:
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

        # recall environment facts for better planning context
        facts = self.memory.recall(goal, session_id=session_id)

        state = PlanState(goal=goal, tasks={})
        self.save_state(state)
        
        tasks = await self.create_plan(goal, facts=facts)
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
                        task = await self.edit_task(state.goal, self.generate_log(state), tid, task, set(state.tasks.keys()), facts=facts)
                        state.tasks[tid] = task
                        self.save_state(state)

                    try:
                        res = await self.dispatch(task, state, session_id=session_id)
                        
                        # capture nuances regardless of status to build a technical environment map
                        if hasattr(res, "nuances") and res.nuances:
                            for key, value in res.nuances.items():
                                state.nuances[key] = Nuance(value=str(value), source_task=tid)

                        if hasattr(res, "status") and res.status == "done":
                            task.findings_summary = res.findings_summary
                            task.failure_reason = None
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
                    logger.info(f"workflow blocked. initiating replan (attempt {replan_count})")
                    
                    done_tids = {tid for tid, t in state.tasks.items() if t.status == "done"}
                    new_tasks = await self.replan(state.goal, self.generate_log(state), done_tids, facts=facts)
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
        # print(f"-------------suprevisor----------------\n{log}")
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
            dep_str = f" [depends: {', '.join(t.depends_on)}]" if t.depends_on else ""
            log.append(f"- {tid} ({t.status}){dep_str}: {t.description}")
            log.append(f"  > EXPECTED: {t.expected_output}")
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
        initializes the reporter agent with its orchestrator.

        returns:
            none
        """
        super().__init__()
        self.orchestrator = Orchestrator()
        self.agent = None
        self.tools = []
        self.active_session = None
        self.max_iter = self.m_config.get("agents", {}).get("reporter_max_iter", 10)

    async def _ensure_agent(self):
        """
        ensures the synthetic agent is initialized with filtered mcp and internal tools.
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
                coroutine=self.orchestrator_run,
                description="Run multi-step local system tasks, terminal commands (e.g. ls, cat, etc), or code investigations. Returns a status tree. Findings are committed to memory. USE THIS TOOL whenever you need to interact with the local file system or OS.",
                args_schema=OrchestratorRunSchema
            ),
            StructuredTool.from_function(
                name="memory_query",
                coroutine=self.memory_query,
                description="Search internal memory for facts and findings from previous tasks.",
                args_schema=MemoryQuerySchema
            )
        ] + mcp_tools
        
        @dynamic_prompt
        def reporter_prompt(request):
            facts = request.state.get("facts", "")
            os_info = request.state.get("os_info", "unknown")
            prompt = self.prompts.get("reporter", {}).get("system_prompt", "")
            return prompt.format(facts=facts, os_info=os_info)

        self.agent = create_agent(
            model=self.core_model,
            tools=self.tools,
            middleware=[reporter_prompt],
            name="reporter"
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

    async def orchestrator_run(self, task: str) -> str:
        """
        executes a complex, multi-step task through the orchestrator.

        args:
            task (str): multi-step local request to execute.

        returns:
            str: llm-friendly log of the execution flow.
        """
        return await self.orchestrator(task, session_id=self.active_session)

    async def __call__(self, query: str, session_id: str):
        """
        processes a user query by coordinating facts, history, and the agent.

        args:
            query (str): the user's initial question.
            session_id (str): identifier for the current session.

        returns:
            asyncgenerator: yields text chunks and tool status updates.
        """
        # print(f"\n--- reporter received query ---\n{query}\n")
        self.active_session = session_id
        response = ""

        try:
            await self._ensure_agent()
            
            # inject high-level facts (mem0) into the prompt context
            facts = self.memory.recall(query, session_id=session_id)

            # fetch os info for environmental awareness
            os_info = self.orchestrator.current_os
            if os_info == "unknown":
                try:
                    ping_data = await self.mcp_client.ping()
                    if isinstance(ping_data, dict):
                        os_info = ping_data.get("os", "unknown")
                        self.orchestrator.current_os = os_info
                except Exception:
                    pass

            archive_all = self.memory.get_archive()
            session_history = archive_all.get(session_id, [])

            messages = session_history + [{"role": "user", "content": query}]

            async for chunk in self.agent.astream(
                {"messages": messages, "facts": f"RECALLED FACTS:\n{facts}" if facts else "", "os_info": os_info}, 
                config={"max_iterations": self.max_iter}, 
                stream_mode="messages"
            ):
         
                if not isinstance(chunk, tuple):
                    continue
                
                msg, metadata = chunk
                node = metadata.get("langgraph_node")
                
                # filter out non-reporter agents, tool nodes, non-AI messages, and messages with tool calls
                if (metadata.get("lc_agent_name") != "reporter" or 
                    node == "tools" or 
                    "AIMessage" not in msg.__class__.__name__ or 
                    getattr(msg, "tool_calls", [])):
                    
                    # UI status bar exception: Yield tool calls specifically from the reporter node
                    if node == "reporter" and getattr(msg, "tool_calls", []):
                        for tc in msg.tool_calls:
                            yield {"tool": tc["name"], "args": tc["args"]}
                    continue

                # extract and yield content
                content = getattr(msg, "content", "")
                parts = content if isinstance(content, list) else [content]
                text = "".join(p.get("text", "") if isinstance(p, dict) else p for p in parts)

                if text:
                    response += text
                    yield text
        except Exception as e:
            import traceback
            error_msg = f"streaming error in session {session_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # if it's an ExceptionGroup (anyio TaskGroup error), log sub-exceptions
            if hasattr(e, "exceptions"):
                for i, sub_e in enumerate(e.exceptions):
                    logger.error(f"sub-exception {i}: {sub_e}")
                    
            yield f"\n[REPORTER ERROR] {str(e)}"

        archive_all = self.memory.get_archive()
        if session_id not in archive_all: archive_all[session_id] = []
        archive_all[session_id].append({"role": "user", "content": query})
        archive_all[session_id].append({"role": "assistant", "content": response})
        if len(archive_all[session_id]) > 10: archive_all[session_id] = archive_all[session_id][-10:]
        self.memory.save_archive(archive_all)
        
        # extract facts from the conversation to update long-term memory (mem0)
        self.memory.memorize([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ], session_id=session_id)

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
        self.orchestrator.wipe_state()
        self.memory.clear_session(target_session)

    def purge_all(self):
        """
        purges all memory data and global state.

        returns:
            none
        """
        # wipe the global plan state during a total purge
        self.orchestrator.wipe_state()
        self.memory.purge_all()

