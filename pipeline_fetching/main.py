# main.py
import os, sys
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(__file__))

from logging_utils import setup_logger
setup_logger()  # <— ajouter cette ligne

from pipeline import run as run_pipeline

if __name__ == "__main__":
    run_pipeline(
        what="all",
        resources=os.getenv("RESOURCES", "candidates,jobs,job-applications"),
        per_page=int(os.getenv("PER_PAGE", "100")),
        max_workers=int(os.getenv("MAX_WORKERS", "5")),
        force=os.getenv("FORCE", "false").lower() == "true",
        with_files=os.getenv("WITH_FILES", "true").lower() == "true",
    )
