# Minimal mapping to avoid .bin when possible
MIME_EXT = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/rtf": ".rtf",
    "application/zip": ".zip",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

def extension_from_mime(content_type: str, default: str = ".bin") -> str:
    if not content_type:
        return default
    ct = content_type.split(";")[0].strip().lower()
    return MIME_EXT.get(ct, default)
