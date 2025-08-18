import os, sys
# Si 'cvcompat' est à côté de ce fichier :
sys.path.append(os.path.dirname(__file__))

from .pipeline import run as run_pipeline

if __name__ == "__main__":
    # Lis tes variables d'env depuis le Job (recommandé) ou définir ici par défaut
    run_pipeline(
        what="all",
        resources=os.getenv("RESOURCES", "candidates,jobs,job-applications"),
        per_page=int(os.getenv("PER_PAGE", "100")),
        max_workers=int(os.getenv("MAX_WORKERS", "5")),
        force=os.getenv("FORCE", "false").lower() == "true",
        with_files=os.getenv("WITH_FILES", "true").lower() == "true",
    )
