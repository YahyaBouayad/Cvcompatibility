from __future__ import annotations

import typer
from typing import List, Tuple

from logging_utils import setup_logger
from teamtailor import TTClient
from storage import BlobStore
from state import StateStore
from enrich import enrich_resource, backfill_jobs_from_jobapps
import config as cfg

app = typer.Typer(add_completion=False)
logger = setup_logger()

def _split_resources(resources: str) -> List[str]:
    return [r.strip() for r in resources.split(",") if r.strip()]

def _client_store_state() -> Tuple[TTClient, BlobStore, StateStore]:
    client = TTClient()
    store = BlobStore()
    state = StateStore(store)
    return client, store, state

def run(
    what: str = "all",
    resources: str = "candidates,job-applications,jobs",
    per_page: int = 100,
    max_workers: int = cfg.MAX_WORKERS,
    force: bool = False,
    with_files: bool = True,
    force_files: bool = False,
):
    """
    Orchestration de la pipeline de fetching/enrichissement.

    Ordre:
      1) candidates      -> raw + enriched (+ files si with_files=True)
      2) job-applications-> raw + enriched
      3) jobs            -> backfill par IDs depuis job-applications
    """
    if what != "all":
        raise ValueError("Only 'all' is supported for now.")

    client, store, state = _client_store_state()
    res_list = _split_resources(resources)

    # Traiter tout sauf 'jobs' d'abord
    early = [r for r in res_list if r != "jobs"]
    for res in early:
        include = cfg.INCLUDES.get(res)
        logger.info(f"Start resource: {res}")
        enrich_resource(
            client=client,
            store=store,
            state=state,
            resource=res,
            per_page=per_page,
            max_workers=max_workers,
            include=include,
            force=force,
            with_files=(with_files and res == "candidates"),
            force_files=force_files,
        )
        logger.info(f"Resource done: {res}")

    # Puis backfill jobs
    if "jobs" in res_list:
        logger.info("Start resource: jobs (backfill depuis job-applications)")
        backfill_jobs_from_jobapps(
            client=client,
            store=store,
            state=state,
            per_page=per_page,
            max_workers=max_workers,
            include=cfg.INCLUDES.get("jobs"),
            force=force,
        )
        logger.info("Resource done: jobs (backfill)")

@app.command("run")
def cli_run(
    what: str = typer.Option("all"),
    resources: str = typer.Option("candidates,job-applications,jobs"),
    per_page: int = typer.Option(100),
    max_workers: int = typer.Option(cfg.MAX_WORKERS),
    force: bool = typer.Option(False),
    with_files: bool = typer.Option(True, help="Télécharger aussi les fichiers des candidats"),
    force_files: bool = typer.Option(False, help="Re-télécharger les fichiers même s'ils sont à jour"),
):
    run(
        what=what,
        resources=resources,
        per_page=per_page,
        max_workers=max_workers,
        force=force,
        with_files=with_files,
        force_files=force_files,
    )

if __name__ == "__main__":
    app()
