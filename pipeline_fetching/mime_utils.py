def extension_from_mime(content_type: str, default: str = ".bin") -> str:
    if not content_type:
        return default
    ct = content_type.lower()
    if "pdf" in ct:
        return ".pdf"
    if "wordprocessingml" in ct:
        return ".docx"
    if "msword" in ct:
        return ".doc"
    if "rtf" in ct:
        return ".rtf"
    if "zip" in ct:
        return ".zip"
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "plain" in ct or "text" in ct:
        return ".txt"
    return default
