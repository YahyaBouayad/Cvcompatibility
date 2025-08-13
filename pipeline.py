from __future__ import annotations
import typer
from typing import Optional
from .logging_utils import setup_logger
from .teamtailor import TTClient
from .storage import BlobStore
from .state import StateStore
from .enrich import enrich_resource
from . import config as cfg

app = typer.Typer(add_completion=False)
logger = setup_logger()

def _client_and_store():
    client = TTClient()
    store = BlobStore()
    state = StateStore(store)
    return client, store, state

@app.command()
def fetch(
    resource: str = typer.Argument(..., help="candidates | jobs | job-applications"),
    per_page: int = typer.Option(100, help="Items per page"),
    include: Optional[str] = typer.Option(None, help="Comma-separated includes"),
    force: bool = typer.Option(False, help="Force re-upload even if raw blob exists"),
):
    client, store, state = _client_and_store()
    inc = include.split(",") if include else None
    from .enrich import fetch_and_store_raw
    fetch_and_store_raw(client, store, resource, per_page, include=inc, force=force)
    typer.echo("Done.")

@app.command()
def enrich(
    resource: str = typer.Argument(...),
    per_page: int = typer.Option(100),
    include: Optional[str] = typer.Option(None),
    max_workers: int = typer.Option(cfg.MAX_WORKERS),
    force: bool = typer.Option(False),
    with_files: bool = typer.Option(False, help="Only for candidates: download uploads/resumes"),
):
    client, store, state = _client_and_store()
    inc = include.split(",") if include else cfg.INCLUDES.get(resource)
    enrich_resource(client, store, state, resource, per_page, max_workers,
                    force=force, include=inc, with_files=with_files)
    typer.echo("Done.")

@app.command()
def run(
    what: str = typer.Argument(..., help="'all' to run everything"),
    resources: str = typer.Option("candidates,jobs,job-applications"),
    per_page: int = typer.Option(100),
    max_workers: int = typer.Option(cfg.MAX_WORKERS),
    force: bool = typer.Option(False),
    with_files: bool = typer.Option(True, help="Enable uploads download for candidates"),
):
    if what != "all":
        raise typer.BadParameter("Only 'all' is supported for now.")
    client, store, state = _client_and_store()
    for res in resources.split(","):
        inc = cfg.INCLUDES.get(res)
        enrich_resource(client, store, state, res, per_page, max_workers,
                        force=force, include=inc, with_files=(with_files and res == "candidates"))
        logger.info(f"Resource done: {res}")

if __name__ == "__main__":
    app()
