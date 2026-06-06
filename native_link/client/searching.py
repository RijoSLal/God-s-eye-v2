from langchain.agents import create_agent
from supervisor import Model
from logs.logging_setup import logger

class Searcher(Model):
    def __init__(self):
        """
        initializes the searcher agent.

        returns:
            none
        """
        super().__init__(callbacks=[])
        self.search_agent = None
        self.max_iter = self.m_config.get("agents", {}).get("searcher_max_iter", 5)

    async def __call__(self, query: str):
        """
        executes a web search and synthesizes the result.

        args:
            query (str): the search query.

        returns:
            str: synthesized search results.
        """
        if not self.search_agent:
            logger.info("initializing search agent tools")
            mcp_tools = await self.mcp_client.get_tools(["web_search"])
            
            self.search_agent = create_agent(
                model = self.core_model,
                system_prompt = self.fetch_prompt("searcher.system_prompt"), 
                tools = mcp_tools, 
                name="searcher"
            )

        logger.info(f"executing web search for: {query[:50]}...")
        result = await self.search_agent.ainvoke({
            "messages": [
                    {
                        "role": "user",
                        "content":  query
                    }
                ]
            },
            config = {
                "max_iterations": self.max_iter,
                "callbacks": []
            }
        )
        
        # print the synthesized result for visibility
        # print(f"\n--- search result synthesis ---\n{str(result)[:500]}...\n")
        
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_msg = messages[-1]
                content = ""
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                elif isinstance(last_msg, dict) and "content" in last_msg:
                    content = last_msg["content"]
                else:
                    content = str(last_msg)
                
                return content
        
        return result
 
