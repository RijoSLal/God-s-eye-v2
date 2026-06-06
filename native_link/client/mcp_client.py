import httpx
import os
import yaml
from yaml import CSafeLoader
from urllib.parse import urlparse
from typing import List, Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from logs.logging_setup import logger

class MCPClient:
    """
    a client for the gods eye mcp server that provides langchain-compatible tools.
    """
    def __init__(self, url: Optional[str] = None):
        """
        initializes the mcp client.
        
        args:
            url (Optional[str]): the base url of the mcp server. if none, loads from config.

        returns:
            none
        """
        if url is None:
            config_path = os.path.join(os.path.dirname(__file__), "config/sys_config.yaml")
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.load(f, Loader=CSafeLoader) or {}
                url = cfg.get("settings", {}).get("mcp_url", "http://127.0.0.1:8000/mcp")
            except Exception as e:
                logger.warning(f"failed to load mcp_url from config: {e}. using default.")
                url = "http://127.0.0.1:8000/mcp"

        self.url = url
        # 'http' transport in langchain-mcp-adapters maps to streamable-http/sse.
        self.client = MultiServerMCPClient({
            "gods_eye": {
                "transport": "http",
                "url": self.url
            }
        })

    async def get_tools(self, tool_names: Optional[List[str]] = None):
        """
        retrieves tools from the mcp server and filters them by name if specified.

        args:
            tool_names (Optional[List[str]]): list of tool names to filter by.

        returns:
            list: list of langchain-compatible tools.
        """
        logger.info(f"fetching tools from mcp server: {tool_names}")
        # fetch all tools from the server
        all_tools = await self.client.get_tools()
        
        if not tool_names:
            return all_tools
        
        # return only the requested tools
        return [tool for tool in all_tools if tool.name in tool_names]

    async def ping(self):
        """
        calls the custom /ping route on the mcp server to get local system info.

        returns:
            dict: json response from the server or error info.
        """
        parsed = urlparse(self.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        logger.info(f"pinging mcp server at: {base_url}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{base_url}/ping")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"ping failed for {base_url}: {e}")
                return {"error": f"ping failed: {e}"}

