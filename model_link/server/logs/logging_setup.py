import logging
import os

def setup_logging(name: str = "gods_eye"):
    """
    sets up aesthetic console logging for the mcp server.

    args:
        name (str): name of the logger instance.

    returns:
        logging.logger: the configured logger.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ensure no duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # formatter including timestamp and logger name for docker visibility
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # file handler for persistent logs with rotation
    from logging.handlers import RotatingFileHandler
    log_path = "/var/log/mcp_server/server.log"
    
    # ensure log directory exists
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass # docker mount might be read-only or handled externally

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024, # 10MB
        backupCount=1
    )
    file_handler.setFormatter(console_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # console handler - standard for docker environments
    if os.environ.get("GODS_EYE_TUI") == "true":
        handler = logging.NullHandler()
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(console_formatter)
        handler.setLevel(logging.INFO)

    logger.addHandler(handler)

    # silence noisy libraries
    logging.getLogger("crawl4ai").setLevel(logging.WARNING)
    logging.getLogger("pyserxng").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)

    return logger

# primary logger instance
logger = setup_logging()
