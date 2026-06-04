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
