import os
from dotenv import load_dotenv
load_dotenv()

# === Teamtailor ===
TEAMTAILOR_BASE_URL = os.getenv("TEAMTAILOR_BASE_URL", "https://api.teamtailor.com/v1/")
TEAMTAILOR_API_KEY = os.getenv("TEAMTAILOR_API_KEY", "")
TEAMTAILOR_API_VERSION = os.getenv("TEAMTAILOR_API_VERSION", "20240904")

# === Azure Blob ===
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "cvcompat")

# === Runtime / perf ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API Teamtailor ~50 req / 10 s ⇒ ~5 rps. Démarre conservateur.
RATE_MAX_CALLS_PER_SEC = float(os.getenv("RATE_MAX_CALLS_PER_SEC", "4"))

# Workers pour les appels API
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

# Workers dédiés aux téléchargements/uploads de fichiers (non soumis au rate limit API)
FILES_MAX_WORKERS = int(os.getenv("FILES_MAX_WORKERS", "16"))

# Garde-fou pagination
MAX_PAGES = int(os.getenv("MAX_PAGES", "1000"))

# Pool HTTP (connexions) = MAX_WORKERS × multiplicateur
HTTP_POOL_MULTIPLIER = int(os.getenv("HTTP_POOL_MULTIPLIER", "2"))

# Watchdog (dump des stacks si attente longue) — mets 0 pour désactiver
WATCHDOG_SECONDS = int(os.getenv("WATCHDOG_SECONDS", "600"))

# Parallélisme dédié aux JSON (raw/enriched) pour éviter d'ouvrir trop de connexions Azure en même temps
BLOB_JSON_PARALLELISM = int(os.getenv("BLOB_JSON_PARALLELISM", "2"))


# === Includes JSON:API ===
INCLUDES = {
    "job-applications": [
        "job", "candidate", "stage", "reject-reason", "nps-responses"
    ],
    "candidates": [
        "job-applications", "activities", "answers", "questions",
        "form-answers", "custom-field-values", "uploads"
    ],
    "jobs": [
        "activities", "candidates", "stages", "team-memberships", "custom-fields"
    ],
}
