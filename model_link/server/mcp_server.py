from mcp.server.fastmcp import FastMCP
from tools import Recon, Searcher
from starlette.requests import Request
from starlette.responses import JSONResponse
from config_manager import tools_config
from logs.logging_setup import logger
# initialize fastmcp
mcp = FastMCP("gods_eye_v2")

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
    search the web for information and return a condensed summary.
    args:
        query (str): the search query.
        surface (bool): whether to perform a surface-level search (faster) or deep search.
    returns:
        str: condensed summary of extracted content from the web results.
    """
    logger.info(f"web_search call: query='{query}', surface={surface}")
    result =  await searcher(query, surface)
    return result

@mcp.tool(
    name=tools_config.fetch("tools.terminal.name"),
    description=tools_config.fetch("tools.terminal.description")
)
async def terminal(command: str) -> str:
    """
    execute a shell command on the remote vm via ssh.
    args:
        command (str): the terminal command to execute.
    returns:
        str: output from the remote terminal.
    """
    logger.info(f"terminal call: command='{command}'")
    return await recon.terminal(command)

@mcp.custom_route("/ping", methods=["GET"])
async def ping(request: Request) -> JSONResponse:
    """endpoint that returns system info and connectivity status."""
    logger.info("ping request received")
    ssh_status = await recon.ping_ssh()
    search_status = await searcher.ping()
    
    # merge the ssh status with the crawler status
    res = ssh_status if isinstance(ssh_status, dict) else {"status": "failed", "os": None}
    res["crawler"] = "online" if search_status else "offline"
    
    return JSONResponse(res)

if __name__ == "__main__":
    import uvicorn
    logger.info("starting gods_eye_v2 mcp server on 127.0.0.1:8000")
    # use uvicorn to run the streamable_http_app with specific host and port
    uvicorn.run(mcp.streamable_http_app, host="127.0.0.1", port=8000)
