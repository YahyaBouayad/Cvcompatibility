import os

TEAMTAILOR_BASE_URL = os.getenv("TEAMTAILOR_BASE_URL", "https://api.teamtailor.com/v1/")
TEAMTAILOR_API_KEY = os.getenv("TEAMTAILOR_API_KEY", "")
TEAMTAILOR_API_VERSION = os.getenv("TEAMTAILOR_API_VERSION", "20240904")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "cvcompat")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_MAX_CALLS_PER_SEC = float(os.getenv("RATE_MAX_CALLS_PER_SEC", "4"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

# Includes par défaut pour l’enrichissement
INCLUDES = {
    "job-applications": ["candidate", "job", "stage", "reject-reason"],
    "candidates": ["applications", "tags", "uploads", "custom-field-values"],
    "jobs": ["company", "locations", "department"],
}
