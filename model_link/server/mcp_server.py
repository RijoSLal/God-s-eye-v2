from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from tools import Recon, Searcher
from starlette.requests import Request
from starlette.responses import JSONResponse
from config_manager import tools_config
from logs.logging_setup import logger
# initialize fastmcp
mcp = FastMCP(
    "gods_eye_v2",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
)
streamable_http_app = mcp.streamable_http_app

# initialize helpers
logger.info("initializing mcp helpers: recon and searcher")
recon = Recon()
searcher = Searcher()

@mcp.tool(
    name=tools_config.fetch("tools.web_search.name"),
    description=tools_config.fetch("tools.web_search.description")
)
async def web_search(query: str, surface: bool = False) -> str:
    """
    search the web for a query and return a summary of findings.
    args:
        query: search query.
        surface: whether to use surface search (faster) or deep search.
    """
    res = await searcher(query, surface=surface)
    return res

@mcp.tool(
    name=tools_config.fetch("tools.terminal.name"),
    description=tools_config.fetch("tools.terminal.description")
)
async def terminal(command: str) -> str:
    """
    execute a shell command on the remote vm via ssh.
    args:
        command: the shell command to run.
    """
    res = await recon.terminal(command)
    return res

@mcp.custom_route("/ping", methods=["GET"])
async def ping(request: Request) -> JSONResponse:
    """
    custom route to verify mcp server connectivity and status.
    """
    logger.info("ping request received")
    
    res = await recon.ping_ssh()
    search_status = await searcher.ping()
    
    res["crawler"] = "online" if search_status else "offline"
    
    return JSONResponse(res)

# if __name__ == "__main__":
#     import uvicorn
#     # use uvicorn to run the streamable_http_app with specific host and port
#     uvicorn.run(mcp.streamable_http_app, host="127.0.0.1", port=8000)
