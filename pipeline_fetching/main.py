import os
import sys
import faulthandler

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(__file__))

from logging_utils import setup_logger
import config
from pipeline import run as run_pipeline

# Watchdog: dump stacks périodiquement si ça se fige
faulthandler.enable()
if config.WATCHDOG_SECONDS > 0:
    faulthandler.dump_traceback_later(config.WATCHDOG_SECONDS, repeat=True)

setup_logger()

if __name__ == "__main__":
    run_pipeline(
        what=os.getenv("WHAT", "all"),
        resources=os.getenv("RESOURCES", "candidates,job-applications,jobs"),
        per_page=int(os.getenv("PER_PAGE", "100")),
        max_workers=int(os.getenv("MAX_WORKERS", str(config.MAX_WORKERS))),
        force=os.getenv("FORCE", "false").lower() == "true",
        with_files=os.getenv("WITH_FILES", "true").lower() == "true",
        force_files=os.getenv("FORCE_FILES", "false").lower() == "true",
    )
