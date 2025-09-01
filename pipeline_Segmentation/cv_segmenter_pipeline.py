#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, json, argparse, tempfile, sys, time, random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# --- IO / Fichiers ---
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import ContentSettings
from pypdf import PdfReader
import docx2txt
import pandas as pd
from io import StringIO

# --- LLM: Azure OpenAI  ---
from openai import AzureOpenAI
import zipfile




# ============== CONFIG ==============
# IMPORTANT: côté Azure OpenAI, "model" (param API) = NOM DU DEPLOYMENT
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

AOAI_RPM=1          # 1 requête/minute
AOAI_TPM=60000      # 60k tokens/min (à adapter à ton quota réel)
AOAI_MIN_WAIT=1.0
CV_TEXT_MAX_CHARS=16000

#INCLUDE_UPLOADS_ONLY = os.getenv("CV_INCLUDE_UPLOADS_ONLY", "1") == "1"
INCLUDE_UPLOADS_ONLY=0
OUTPUT_BLOB_PREFIX = os.getenv("OUTPUT_BLOB_PREFIX", "processed/segmentation")
# ====================================

SECTION_SCHEMA = [
    "profile_summary",
    "experiences",
    "education",
    "skills",
    "languages",
    "certifications",
    "projects",
    "tools_and_technologies",
    "achievements",
    "interests"
]

SUMMARY_FIELDS = [
    "candidate_id",
    "full_name",
    "title_or_role",
    "total_years_experience",
    "last_company",
    "last_title",
    "highest_degree",
    "languages_summary",
    "main_skills_top5",
]

OUT_DIR = Path("out_segmentation")
OUT_JSON_DIR = OUT_DIR / "cv_json"
OUT_JSONL = OUT_DIR / "cv_parsed_full.jsonl"
OUT_CSV = OUT_DIR / "cv_parsed_summary.csv"


# --- CLI ---
def require_env(varname: str) -> str:
    v = os.getenv(varname, "")
    if not v:
        print(f"[CONFIG] ENV manquant: {varname}")
        sys.exit(2)
    return v


# --------- Azure Blob helpers ---------
def get_container_client(connection_string: str, container: str):
    svc = BlobServiceClient.from_connection_string(connection_string)
    return svc.get_container_client(container)

def blob_exists(container_client, name: str) -> bool:
    try:
        return container_client.get_blob_client(name).exists()
    except Exception:
        return False

def upload_bytes(container_client, data: bytes, dest_name: str, content_type: str):
    container_client.upload_blob(
        name=dest_name,
        data=data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type, cache_control="no-cache"),
    )

def upload_json(container_client, obj: Dict[str, Any], dest_name: str):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    upload_bytes(container_client, data, dest_name, content_type="application/json")

def upload_text(container_client, text: str, dest_name: str, content_type: str):
    upload_bytes(container_client, text.encode("utf-8"), dest_name, content_type=content_type)

def list_cv_blobs(container_client, prefix: str, limit: Optional[int] = None) -> List[str]:
    """
    Liste TOUS les blobs sous le préfixe (on ne filtre plus par extension).
    Optionnellement, on peut encore filtrer sur '/uploads/' via CV_INCLUDE_UPLOADS_ONLY.
    """
    names = []
    for b in container_client.list_blobs(name_starts_with=prefix or ""):
        name = b.name
        if INCLUDE_UPLOADS_ONLY and "/uploads/" not in name.lower():
            continue
        names.append(name)
        if limit and len(names) >= limit:
            break

    # petit log de debug pour vérifier
    print(f"   → {len(names)} blobs trouvés sous '{prefix}' (uploads_only={INCLUDE_UPLOADS_ONLY})")
    for n in names[:5]:
        print(f"     - {n}")
    if not names and INCLUDE_UPLOADS_ONLY:
        print("   (hint) Aucun blob trouvé avec le filtre '/uploads/'. Mets CV_INCLUDE_UPLOADS_ONLY=0")
    return names


def download_blob_to_bytes(container_client, blob_name: str) -> bytes:
    return container_client.download_blob(blob_name).readall()
# -------- Extraction texte --------
def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"

def extract_text_from_pdf_bytes(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        reader = PdfReader(tmp.name)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)

def extract_text_from_docx_bytes(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        text = docx2txt.process(tmp.name)
        return text or ""

def extract_text_from_txt_bytes(data: bytes, encoding="utf-8", fallback="latin-1") -> str:
    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        return data.decode(fallback, errors="ignore")

def looks_like_docx_bytes(data: bytes) -> bool:
    # .docx = zip avec 'word/document.xml'
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            if not zipfile.is_zipfile(tmp.name):
                return False
            with zipfile.ZipFile(tmp.name) as z:
                names = z.namelist()
                return ("word/document.xml" in names) or ("[Content_Types].xml" in names)
    except Exception:
        return False

def extract_cv_text(blob_name: str, data: bytes) -> str:
    lname = blob_name.lower()

    # 1) PDF par signature
    if is_pdf_bytes(data) or lname.endswith(".pdf") or (lname.endswith(".bin") and is_pdf_bytes(data)):
        return extract_text_from_pdf_bytes(data)

    # 2) DOCX par signature zip
    if lname.endswith(".docx") or looks_like_docx_bytes(data):
        return extract_text_from_docx_bytes(data)

    # 3) TXT sinon (tentative de décodage)
    if lname.endswith(".txt"):
        return extract_text_from_txt_bytes(data)

    # 4) Fallback auto: PDF ? DOCX ? sinon texte
    if is_pdf_bytes(data):
        return extract_text_from_pdf_bytes(data)
    if looks_like_docx_bytes(data):
        return extract_text_from_docx_bytes(data)
    return extract_text_from_txt_bytes(data)



# -------- Nettoyage --------
def simple_clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


# -------- LLM (Azure OpenAI) --------
def get_azure_client() -> AzureOpenAI:
    api_key = require_env("AZURE_OPENAI_API_KEY")
    endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    if not endpoint.startswith("https://") or "openai.azure.com" not in endpoint:
        print(f"[CONFIG] AZURE_OPENAI_ENDPOINT invalide: {endpoint}")
        sys.exit(2)
    if not AZURE_OPENAI_DEPLOYMENT:
        print("[CONFIG] AZURE_OPENAI_DEPLOYMENT vide (mettre le NOM du déploiement Azure).")
        sys.exit(2)
    print(f"[AZURE] endpoint={endpoint} deployment={AZURE_OPENAI_DEPLOYMENT} api={AZURE_OPENAI_API_VERSION}")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=AZURE_OPENAI_API_VERSION)

def _estimate_tokens(text: str) -> int:
    # approx grossière : 1 token ≈ 4 caractères en moyenne.
    return max(1, int(len(text) / 4))

def _throttle_before_call(prompt_tokens: int, completion_tokens_budget: int = 800):
    """
    Endort assez longtemps pour rester sous RPM/TPM.
    - RPM -> intervalle mini entre appels = 60 / RPM secondes
    - TPM -> on dort proportionnellement aux tokens qu'on va consommer
    """
    rpm_sleep = 60.0 / max(1, AOAI_RPM)
    tpm_sleep = 60.0 * (prompt_tokens + completion_tokens_budget) / max(1, AOAI_TPM)
    sleep_s = max(AOAI_MIN_WAIT, rpm_sleep, tpm_sleep)
    time.sleep(sleep_s)

def _sleep_from_429_message(msg: str):
    """
    Azure renvoie souvent 'Please retry after 60 seconds.' On parse et on dort.
    """
    # Cherche un entier juste avant 'second'
    m = re.search(r"after\s+(\d+)\s*second", msg.lower())
    if m:
        secs = int(m.group(1)) + 2  # petite marge
        print(f"   ⏳ 429 reçu: pause {secs}s")
        time.sleep(secs)
    else:
        # fallback si pas d’indication
        print("   ⏳ 429 reçu: pause 65s (fallback)")
        time.sleep(65)

def _call_with_retry(func, max_retries=8, base_delay=2.0, jitter=0.25):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            low = msg.lower()

            # 429 -> on respecte le "retry after", sinon backoff long
            if " 429 " in f" {msg} " or "rate limit" in low:
                _sleep_from_429_message(msg)
                continue

            retriable = any(k in low for k in ["timeout", "temporar", "service unavailable", "connection reset"])
            if not retriable or attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2 ** attempt) * (1 + random.uniform(-jitter, jitter)), 30)
            print(f"   ↻ Retry {attempt+2}/{max_retries} dans ~{delay:.1f}s - cause: {e}")
            time.sleep(delay)

def llm_segment(cv_text: str, hint_name=None) -> dict:
    client = get_azure_client()

    # Tronque le texte pour réduire la conso tokens (et donc limiter le 429)
    cv_text = (cv_text or "")[:CV_TEXT_MAX_CHARS]

    prompt = f"""Tu es un assistant d'extraction d'informations de CV. Retourne STRICTEMENT du JSON valide.
Schéma attendu:
{{
  "full_name": str | null,
  "title_or_role": str | null,
  "contacts": {{
    "email": str | null, "phone": str | null, "location": str | null, "linkedin": str | null, "github": str | null, "other": [str]
  }},
  "profile_summary": str | null,
  "experiences": [{{"company": str | null, "title": str | null, "start_date": str | null, "end_date": str | null, "duration_months": int | null, "location": str | null, "description": str | null, "tech": [str]}}],
  "education": [{{"school": str | null, "degree": str | null, "field": str | null, "start_date": str | null, "end_date": str | null }}],
  "skills": [str],
  "languages": [{{"language": str, "level": str | null}}],
  "certifications": [{{"name": str, "issuer": str | null, "date": str | null}}],
  "projects": [{{"name": str, "description": str | null, "tech": [str]}}],
  "tools_and_technologies": [str],
  "achievements": [str],
  "interests": [str],
  "computed": {{
    "total_years_experience": float | null,
    "main_skills_top5": [str],
    "last_company": str | null,
    "last_title": str | null,
    "highest_degree": str | null,
    "languages_summary": str | null
  }}
}}
Consignes:
- Dates au format YYYY-MM quand possible.
- Remplis 'computed' (approximation ok).
- Si une info est introuvable, mets null (ne supprime aucun champ).
- Réponds UNIQUEMENT avec le JSON.
{('Nom supposé: ' + hint_name) if hint_name else ''}

=== CV TEXT START ===
{cv_text}
=== CV TEXT END ===
"""

    # Throttle proactif (RPM/TPM)
    est_tokens = _estimate_tokens(prompt) + 600  # budget output ~600 tokens max
    _throttle_before_call(est_tokens)

    def _do():
        return client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,         # NOM DU DEPLOYMENT Azure
            temperature=0.1,
            response_format={"type": "json_object"},
            # Limite la taille de sortie pour mieux respecter TPM
            max_tokens=600,
            messages=[
                {"role": "system", "content": "You are a careful information extraction system. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
        )

    resp = _call_with_retry(_do)
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"raw_parse_error": content}

# -------- Helpers candidat / fichiers --------
CANDIDATE_ID_REGEX = re.compile(r"candidate[_/-](\d+)", re.IGNORECASE)

def candidate_id_from_blob(name: str) -> Optional[str]:
    # 1) Cas Azure: .../candidates/<id>/...
    parts = name.split("/")
    for i, p in enumerate(parts):
        if p.lower() == "candidates" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.isdigit():
                return nxt

    # 2) Cas historique: candidate_12345 / candidate-12345
    m = CANDIDATE_ID_REGEX.search(name)
    if m:
        return m.group(1)

    # 3) Fallback: extraire un gros nombre dans le nom de fichier (ex: upload_61287010.pdf)
    stem = Path(name).stem
    m2 = re.search(r"(\d{6,})", stem)  # >= 6 chiffres
    if m2:
        return m2.group(1)

    # 4) Dernier recours: nom du fichier sans extension
    return stem or None

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

def already_done(candidate_id: str) -> bool:
    return (OUT_JSON_DIR / f"{candidate_id}.json").exists()

def write_candidate_json(candidate_id: str, parsed: Dict[str, Any], meta: Dict[str, Any]):
    payload = {"candidate_id": candidate_id, "parsed": parsed, "meta": meta}
    with open(OUT_JSON_DIR / f"{candidate_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def append_jsonl(records: List[Dict[str, Any]]):
    mode = "a" if OUT_JSONL.exists() else "w"
    with open(OUT_JSONL, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_summary_row(candidate_id: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    comp = (parsed.get("computed") or {})
    row = {
        "candidate_id": candidate_id,
        "full_name": parsed.get("full_name"),
        "title_or_role": parsed.get("title_or_role"),
        "total_years_experience": comp.get("total_years_experience"),
        "last_company": comp.get("last_company"),
        "last_title": comp.get("last_title"),
        "highest_degree": comp.get("highest_degree"),
        "languages_summary": comp.get("languages_summary"),
        "main_skills_top5": ", ".join(comp.get("main_skills_top5") or []),
    }
    return row


# --------- Pipeline ---------
def run_pipeline(
    conn_str: str,
    container: str,
    prefix: str,
    limit: Optional[int] = None,
    force: bool = False
):
    cont = get_container_client(conn_str, container)

    blobs = list_cv_blobs(cont, prefix=prefix, limit=limit)
    print(f"🔎 {len(blobs)} fichiers CV trouvés sous '{container}/{prefix}'")

    summary_rows: List[Dict[str, Any]] = []
    jsonl_lines: List[str] = []

    for i, blob_name in enumerate(blobs, 1):
        print(f"\n[{i}/{len(blobs)}] Traitement: {blob_name}")
        cand_id = candidate_id_from_blob(blob_name) or f"cand_{i}"

        # Skip si déjà présent dans le Blob (et pas de --force)
        dest_json = f"{OUTPUT_BLOB_PREFIX}/json/{cand_id}.json"
        if (not force) and blob_exists(cont, dest_json):
            print(f"   ↳ SKIP (déjà présent dans le blob): {dest_json}")
            continue

        # 1) Download
        try:
            raw = download_blob_to_bytes(cont, blob_name)
        except Exception as e:
            print(f"   ⚠️  Téléchargement impossible: {e}")
            continue

        # 2) Extraction texte
        try:
            text = extract_cv_text(blob_name, raw)
            text = simple_clean(text)
            if not text:
                print("   ⚠️  Texte vide après extraction.")
                continue
        except Exception as e:
            print(f"   ⚠️  Extraction texte KO: {e}")
            continue

        # 3) LLM
        try:
            parsed = llm_segment(text, hint_name=None)
            meta = {
                "blob_name": blob_name,
                "deployment": AZURE_OPENAI_DEPLOYMENT,
                "api_version": AZURE_OPENAI_API_VERSION,
                "ts": pd.Timestamp.utcnow().isoformat()
            }
            payload = {"candidate_id": cand_id, "parsed": parsed, "meta": meta}

            # 4) Upload JSON candidat
            upload_json(cont, payload, dest_json)
            print(f"   ✅ Upload JSON → {dest_json}")

            # 5) Accumulate pour JSONL/CSV
            jsonl_lines.append(json.dumps({"candidate_id": cand_id, "blob": blob_name, "parsed": parsed}, ensure_ascii=False))
            comp = parsed.get("computed") or {}
            summary_rows.append({
                "candidate_id": cand_id,
                "full_name": parsed.get("full_name"),
                "title_or_role": parsed.get("title_or_role"),
                "total_years_experience": comp.get("total_years_experience"),
                "last_company": comp.get("last_company"),
                "last_title": comp.get("last_title"),
                "highest_degree": comp.get("highest_degree"),
                "languages_summary": comp.get("languages_summary"),
                "main_skills_top5": ", ".join(comp.get("main_skills_top5") or []),
            })

        except Exception as e:
            print(f"   ⚠️  LLM/Upload KO: {e}")
            continue

    # 6) Upload agrégats (overwrite à chaque run)
    if jsonl_lines:
        dest_jsonl = f"{OUTPUT_BLOB_PREFIX}/cv_parsed_full.jsonl"
        upload_text(cont, "\n".join(jsonl_lines) + "\n", dest_jsonl, content_type="application/x-ndjson")
        print(f"🧾 Upload JSONL → {dest_jsonl}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8")
        dest_csv = f"{OUTPUT_BLOB_PREFIX}/cv_parsed_summary.csv"
        upload_text(cont, csv_buf.getvalue(), dest_csv, content_type="text/csv")
        print(f"📄 Upload CSV → {dest_csv}")


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline 02 - Segmentation CV via Azure OpenAI (Blob-only)")
    p.add_argument("--prefix", default=os.getenv("CV_BLOB_PREFIX", ""), help="Préfixe des chemins de CV dans le container")
    p.add_argument("--limit", type=int, default=50, help="Limiter le nombre de CV traités")
    p.add_argument("--force", action="store_true", help="Reparser même si déjà présent dans le blob")
    return p.parse_args()


def main():
    load_dotenv()

    conn_str = require_env("AZURE_STORAGE_CONNECTION_STRING")
    container = require_env("AZURE_BLOB_CONTAINER")
    require_env("AZURE_OPENAI_API_KEY")
    require_env("AZURE_OPENAI_ENDPOINT")
    require_env("AZURE_OPENAI_DEPLOYMENT")

    args = parse_args()
    print("➡️  Démarrage pipeline segmentation CV (Azure OpenAI, Blob-only)")
    print(f"   Container : {container}")
    print(f"   Prefix    : {args.prefix or '(racine)'}")
    print(f"   Limit     : {args.limit or '(illimité)'}")
    print(f"   Force     : {args.force}")
    print(f"   Output    : {OUTPUT_BLOB_PREFIX}/...")

    run_pipeline(
        conn_str=conn_str,
        container=container,
        prefix=args.prefix,
        limit=args.limit,
        force=args.force
    )


if __name__ == "__main__":
    main()