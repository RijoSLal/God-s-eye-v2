from supervisor import Model
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_core.tools import StructuredTool
from structure import AnalystReport, AnalystSchema
from langchain.agents.structured_output import ProviderStrategy
from logs.logging_setup import logger

class Analyst(Model):
    def __init__(self):
        """
        initializes the auditing analyst with technical verification tools.
        """
        super().__init__()
        self.max_iter = self.m_config.get("agents", {}).get("analyst_max_iter", 5)

        self.tools = [
            StructuredTool.from_function(
                func = None,
                coroutine = self.insert_doc,
                name = self.fetch_prompt("analyst.tools.record.name", "record"),
                description =  self.fetch_prompt("analyst.tools.record.description"),
                args_schema = AnalystSchema,
            ),
            StructuredTool.from_function(
                name="memory_query",
                coroutine=self.memory_query,
                description="Search internal technical logs for cross-verification of current results."
            )
        ]
        
        @dynamic_prompt
        def analyst_prompt(request):
            facts = request.state.get("facts", "")
            os_info = request.state.get("os_info", "unknown")
            prompt = self.prompts.get("analyst", {}).get("system_prompt", "")
            return prompt.format(facts=facts, os_info=os_info)

        self.analyst_agent = create_agent(
            model = self.core_model,
            tools = self.tools,
            response_format=ProviderStrategy(schema=AnalystReport),
            middleware=[analyst_prompt],
            name="analyst"
        )

    async def memory_query(self, query: str) -> str:
        """
        searches internal technical logs for cross-verification of current results.

        args:
            query (str): the search query to run against memory.

        returns:
            str: verification results from technical logs.
        """
        # technical log lookup for verification
        return self.memory.query(query, session_id=getattr(self, "active_session", "audit_session"))

    async def insert_doc(self, info: str):
        """
        records a specific data point or finding into the session memory.

        args:
            info (str): technical information to be recorded.

        returns:
            str: confirmation message upon successful recording.
        """
        # print(f"    [agent: analyst] tool: record (info={info[:100]}...)")
        self.memory.store(info, session_id=getattr(self, "active_session", "audit_session"))
        return "Successfully recorded data point for future analysis."

    async def __call__(self, operations: str, session_id: str, original_query: str = None):
        """
        executes the auditing process on a set of operations.

        args:
            operations (str): formatted string of technical steps taken.
            session_id (str): identifier for the current session.
            original_query (str): the user's initial question.

        returns:
            AnalystReport: technical audit and summary.
        """
        self.active_session = session_id
        logger.info(f"running technical audit for session: {session_id}")
        
        # fetch os info for environmental awareness during audit
        os_info = "unknown"
        try:
            ping_data = await self.mcp_client.ping()
            if isinstance(ping_data, dict):
                os_info = ping_data.get("os", "unknown")
        except Exception:
            pass

        # inject facts (mem0) into the audit to detect contradictions with reality
        facts = self.memory.recall(original_query or operations, session_id=session_id)

        # construct user message using the prompt from yaml
        user_content = self.fetch_prompt("analyst.user_prompt", "{operations}").format(
            operations=operations,
            goal=original_query
        )
            
        messages = [{"role": "user", "content": user_content}]

        # execute the analyst agent with stdout redirected to the log queue to keep the main console clean

        result = await self.analyst_agent.ainvoke(
            {"messages": messages, "facts": f"VERIFIED_FACTS:\n{facts}" if facts else "", "os_info": os_info},
            config = {"max_iterations": self.max_iter, "callbacks": []}
        ) 
    
       
        report = result.get("structured_response") if isinstance(result, dict) else None
        
        # verify structured report has all required content
        expected_fields = ["status", "findings_summary", "failure_reason", "nuances"]
        if not report or not all(hasattr(report, field) for field in expected_fields):
            # fallback: structured output missing or incomplete is always a failure
            last_msg_content = ""
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    last_msg_content = last_msg.content
                elif isinstance(last_msg, dict):
                    last_msg_content = last_msg.get("content", "")

            report = AnalystReport(
                status = "failure",
                findings_summary = last_msg_content or "NO FINDINGS EXTRACTED.",
                failure_reason = "AGENT FAILED TO PROVIDE A VALID STRUCTURED REPORT",
                nuances = {}
            )

        # final commitment: store the verified audit report
        commit_key = original_query or operations
        self.memory.commit(commit_key, report.findings_summary, session_id=session_id)
        # print(f"\n--- analyst agent output ---\nstatus: {report.status}\nfindings: {report.findings_summary}\nfailure reason: {report.failure_reason}\n")
        return report
