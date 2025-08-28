import logging
import os
import sys
import config

def setup_logger():
    level = getattr(logging, os.getenv("LOG_LEVEL", config.LOG_LEVEL).upper(), logging.INFO)
    logger = logging.getLogger()
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger
