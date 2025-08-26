# pipeline.py
from __future__ import annotations
import typer
from typing import Optional, List
from logging_utils import setup_logger
from teamtailor import TTClient
from storage import BlobStore
from state import StateStore
from enrich import enrich_resource
import config as cfg

app = typer.Typer(add_completion=False)
logger = setup_logger()

def _split_resources(resources: str) -> List[str]:
    return [r.strip() for r in resources.split(",") if r.strip()]

def _client_store_state():
    # Respecte la conf de rate limit
    client = TTClient(rate_per_sec=cfg.RATE_MAX_CALLS_PER_SEC)
    store = BlobStore()
    state = StateStore(store)
    return client, store, state

def run(
    what: str = "all",
    resources: str = "candidates,jobs,job-applications",
    per_page: int = 100,
    max_workers: int = cfg.MAX_WORKERS,
    force: bool = False,
    with_files: bool = True,   # ne télécharge vraiment que pour "candidates"
):
    """
    Lance la pipeline:
      - list → enrich JSON détaillés
      - (optionnel) téléchargement des uploads pour les candidats
      - mise à jour du state
    """
    if what != "all":
        raise ValueError("Only 'all' is supported for now.")

    client, store, state = _client_store_state()
    for res in _split_resources(resources):
        include = cfg.INCLUDES.get(res)
        logger.info(f"Start resource: {res}")
        enrich_resource(
            client=client,
            store=store,
            state=state,
            resource=res,
            per_page=per_page,
            max_workers=max_workers,
            force=force,
            include=include,
            with_files=(with_files and res == "candidates"),
        )
        logger.info(f"Resource done: {res}")

# === CLI Typer (optionnel) ===
@app.command("run")
def cli_run(
    what: str = typer.Option("all"),
    resources: str = typer.Option("candidates,jobs,job-applications"),
    per_page: int = typer.Option(100),
    max_workers: int = typer.Option(cfg.MAX_WORKERS),
    force: bool = typer.Option(False),
    with_files: bool = typer.Option(True, help="Télécharger aussi les fichiers des candidats"),
):
    run(what=what, resources=resources, per_page=per_page, max_workers=max_workers, force=force, with_files=with_files)

if __name__ == "__main__":
    app()
