from supervisor import Model
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.tools import StructuredTool
from auditing import Analyst
from logs.logging_setup import logger
import orjson

class Recon(Model):
    def __init__(self):
        """
        initializes the reconnaissance tool.

        returns:
            none
        """
        super().__init__()

        self.max_iter = self.m_config.get("agents", {}).get("recon_max_iter", 5)
        self.analyst_agent = Analyst()
        self.script_agent = None


    async def terminal(self, command: str) -> str:
        """
        executes a shell command on the remote terminal via ssh.

        args:
            command (str): the command to execute.

        returns:
            str: the output of the command.
        """
        # send command immediately so ui shows what's happening
        await self.terminal_queue.put({"command": command})
        
        if not hasattr(self, "_mcp_terminal"):
            tools = await self.mcp_client.get_tools(["terminal"])
            if not tools:
                logger.error("terminal tool not found on mcp server")
                return "error: terminal tool not found"
            self._mcp_terminal = tools[0]
        
        logger.info(f"executing terminal command: {command[:50]}...")
        try:
            res = await self._mcp_terminal.ainvoke({"command": command})
            
            # extract text from mcp list format robustly
            raw_res = ""
            if isinstance(res, list):
                for i in res:
                    if isinstance(i, dict):
                        raw_res += i.get('text', str(i))
                    elif hasattr(i, 'text') and isinstance(i.text, str):
                        raw_res += i.text
                    elif hasattr(i, 'content') and isinstance(i.content, str):
                        raw_res += i.content
                    else:
                        raw_res += str(i)
            elif isinstance(res, dict):
                raw_res = res.get('text', str(res))
            else:
                raw_res = str(res)
            
            out = raw_res
        except Exception as e:
            out = f"error executing command: {e}"
        
        prompt = "[bold #aaaaaa]remote[/]:[#666666]~[/][bold #aaaaaa]$ [/]" # replace this to console_interface only later
        
        # send output to the queue
        await self.terminal_queue.put({"output": out if out.strip() else "\n"})
        # print(f"\n[remote] command output:\n{out}\n")
        
        # send the new prompt
        await self.terminal_queue.put({"new_prompt": prompt})
        
        return out


    def _operation_history(self, result: dict, user_query: str) -> str:
        """
        builds a visual tree representation of the operation history.

        args:
            result (dict): result from the langgraph/langchain execution.
            user_query (str): the original user query.

        returns:
            str: formatted tree string of tool executions.
        """
        messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
    
        tree_segments = ["."]
        tree_segments.append("├── USER_QUERY")
        tree_segments.append(f"│   └── {user_query}")
        
        tree_segments.append("└── EXECUTION_FLOW")

        tool_inputs = {}
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
            if tool_calls:
                for tc in tool_calls:
                    is_dict = isinstance(tc, dict)
                    tc_id = tc.get("id") if is_dict else getattr(tc, "id", None)
                    tc_args = tc.get("args") if is_dict else getattr(tc, "args", None)
                    if tc_id:
                        tool_inputs[tc_id] = orjson.dumps(tc_args).decode('utf-8') if tc_args else "None"

        tool_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_type = msg.get("type", "")
                role = msg.get("role", "")
                if msg_type == "tool" or role == "tool" or "ToolMessage" in str(msg.__class__):
                    tool_messages.append(msg)
            else:
                msg_type = msg.__class__.__name__
                if msg_type == "ToolMessage":
                    tool_messages.append(msg)
        
        total_tools = len(tool_messages)

        if total_tools == 0:
            # fallback to ai conclusion if no tools were called
            for msg in reversed(messages):
                content = ""
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    if role == "assistant": content = msg.get("content", "")
                elif hasattr(msg, "content"):
                    if msg.__class__.__name__ == "AIMessage": content = msg.content
                
                if content:
                    tree_segments.append(f"    └── Conclusion: {content}")
                    return "\n".join(tree_segments)
            
            tree_segments.append("    └── No execution history found.")
            return "\n".join(tree_segments)

        for i, msg in enumerate(tool_messages):
            is_last_tool = (i == total_tools - 1)
            tool_branch = "    └── " if is_last_tool else "    ├── "
            info_indent  = "        " if is_last_tool else "    │   "
            
            if isinstance(msg, dict):
                tool_id = msg.get("tool_call_id")
                tool_name = msg.get("name")
                tool_content = msg.get("content")
            else:
                tool_id = getattr(msg, "tool_call_id", None)
                tool_name = getattr(msg, "name", None)
                tool_content = getattr(msg, "content", None)
            
            args_str = tool_inputs.get(tool_id, "None")

            tree_segments.append(f"{tool_branch}Step {i + 1}: {tool_name}")
            tree_segments.append(f"{info_indent}├── Input:  {args_str}")
            
            # unpack tool output if it's in the common list-of-dicts format
            clean_content = tool_content
            if isinstance(tool_content, list) and len(tool_content) > 0:
                # check for [{'type': 'text', 'text': '...'}] format
                if isinstance(tool_content[0], dict) and 'text' in tool_content[0]:
                    clean_content = "\n".join([item['text'] for item in tool_content if isinstance(item, dict) and 'text' in item])
            
            if clean_content:
                lines = str(clean_content).splitlines()
                if lines:
                    # truncate huge outputs to keep analyst prompts lean
                    if len(lines) > 20:
                        lines = lines[:10] + ["... [TRUNCATED] ..."] + lines[-10:]
                    
                    tree_segments.append(f"{info_indent}└── Output: {lines[0]}")
                    for line in lines[1:]:
                        tree_segments.append(f"{info_indent}            {line}")
            else:
                tree_segments.append(f"{info_indent}└── Output: None")

        return "\n".join(tree_segments)
    
    async def memory_query(self, query: str) -> str:
        """
        searches internal technical logs for cross-verification of current results.

        args:
            query (str): the search query to run against memory.

        returns:
            str: verification results from technical logs.
        """
        # recon tool focuses on technical logs in qdrant
        return self.memory.query(query, session_id=getattr(self, "active_session", "isolated_session"))

    async def __call__(self, query: str, session_id: str, os_info: str = None):
        """
        executes the reconnaissance process on a query.

        args:
            query (str): the user's technical question or task.
            session_id (str): identifier for the current session.
            os_info (str): optional operating system information.

        returns:
            AnalystReport: technical audit and summary from the analyst.
        """
        self.active_session = session_id
        if not self.script_agent:
            # ... (rest of tool setup)
            mcp_tools = await self.mcp_client.get_tools(["web_search"])
            
            for tool in mcp_tools:
                if tool.name == "web_search":
                    tool.description = (
                        "Search the web for information. Use 'surface=True' for general factual queries, "
                        "news, and non-technical info. Use 'surface=False' (default) ONLY for deep "
                        "technical investigations, coding issues, or security research."
                    )
            
            terminal_tool = StructuredTool.from_function(
                name="terminal",
                coroutine=self.terminal,
                description="Execute a shell command on the remote terminal via SSH. Use this for remote file operations or system checks."
            )

            memory_tool = StructuredTool.from_function(
                name="memory_query",
                coroutine=self.memory_query,
                description="Search internal memory for facts and findings from previous tasks. USE THIS if you need historical context."
            )
            
            combined_tools = mcp_tools + [terminal_tool, memory_tool]
            
            @dynamic_prompt
            def recon_prompt(request):
                os_info = request.state.get("os_info", "unknown")
                prompt = self.prompts.get("recon", {}).get("system_prompt", "")
                return prompt.format(os_info=os_info)

            self.script_agent = create_agent(
                model = self.core_model,
                tools = combined_tools, 
                middleware=[recon_prompt],
                name="recon"
            )

        # 1. use provided os info or fetch if missing
        if not os_info:
            try:
                ping_data = await self.mcp_client.ping()
                if isinstance(ping_data, dict):
                    os_info = ping_data.get("os", "unknown")
            except Exception:
                os_info = "unknown"

        # 2. build initial message list
        messages = [{"role": "user", "content": query}]

        result = await self.script_agent.ainvoke(
            {"messages": messages, "os_info": os_info},
            config = {"max_iterations": self.max_iter, "callbacks": []}
        )
        
        operations = self._operation_history(result, query)
        logger.info(f"\n--- recon operation history ---\n{operations}\n")
        audit_report = await self.analyst_agent(operations, session_id=session_id, original_query=query)
        
        logger.info(f"\n--- recon audit report ---\nstatus: {audit_report.status}\nfindings: {audit_report.findings_summary}\n")
          
        return audit_report

