#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, json, argparse, tempfile, sys, time, hashlib, threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Chargement .env ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- IO / Fichiers / Azure ---
from azure.storage.blob import BlobServiceClient, ContentSettings

# --- PDF / DOCX ---
try:
    import fitz  # PyMuPDF (rapide)
    _has_pymupdf = True
except Exception:
    _has_pymupdf = False

try:
    from pypdf import PdfReader
    _has_pypdf = True
except Exception:
    _has_pypdf = False

try:
    import docx2txt
    _has_docx = True
except Exception:
    _has_docx = False

# --- Data utils ---
try:
    import pandas as pd
    _has_pandas = True
except Exception:
    _has_pandas = False

# --- Azure OpenAI ---
try:
    from openai import AzureOpenAI
except Exception as e:
    print("❌ openai (AzureOpenAI) n’est pas installé. pip install openai>=1.40.0")
    raise

# ============== CONFIG ==============
# Azure OpenAI
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
# Limites quota (env → override)
AOAI_RPM = int(os.getenv("AOAI_RPM", "140"))            # ex. 140
AOAI_TPM = int(os.getenv("AOAI_TPM", "140000"))         # ex. 140_000
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2400"))

# Entrée / sortie Blob
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "cvcompat")
INPUT_PREFIX = os.getenv("INPUT_PREFIX", "tt/files/candidates/")  # où sont les CV
OUTPUT_BLOB_PREFIX = os.getenv("OUTPUT_BLOB_PREFIX", "processed/segmentation")

# Filtrage fichiers
INCLUDE_UPLOADS_ONLY = os.getenv("CV_INCLUDE_UPLOADS_ONLY", "0") == "1"  # filtre '/uploads/'
EXTS = os.getenv("CV_EXTS", "pdf,docx,txt").lower().split(",")

# Taille texte
CV_TEXT_MAX_CHARS = int(os.getenv("CV_TEXT_MAX_CHARS", "16000"))

# Concurrence
from concurrent.futures import ThreadPoolExecutor, as_completed
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "4"))
MAX_DOWNLOAD_CONCURRENCY = int(os.getenv("BLOB_MAX_CONCURRENCY", "4"))

# Résumés (optionnels)
SAVE_SUMMARIES = os.getenv("SAVE_SUMMARIES", "1") == "1"

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# ============== TOKEN ESTIMATION ==============
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(txt: str) -> int:
        return len(_enc.encode(txt))
    def truncate_by_tokens(txt: str, max_tokens: int) -> str:
        ids = _enc.encode(txt)
        if len(ids) <= max_tokens:
            return txt
        return _enc.decode(ids[:max_tokens])
except Exception:
    def estimate_tokens(txt: str) -> int:
        return max(1, int(len(txt) / 3.6))  # fallback grossier
    def truncate_by_tokens(txt: str, max_tokens: int) -> str:
        # fallback grossier (~4 chars/token)
        return txt[:max_tokens * 4]

# ============== RATE LIMITER GLOBAL ==============
class RateLimiter:
    def __init__(self, rpm: int, tpm: int):
        self.rpm = max(1, rpm)
        self.tpm = max(1, tpm)
        self._lock = threading.Lock()
        self._call_times = []   # timestamps des appels (RPM, fenêtre 60s)
        self._tpm_used = 0      # tokens utilisés dans la fenêtre courante
        self._window_start = time.time()

    def acquire(self, prompt_tokens: int, completion_budget: int):
        need = prompt_tokens + completion_budget
        while True:
            with self._lock:
                now = time.time()
                # reset fenêtre TPM chaque 60s
                if now - self._window_start >= 60.0:
                    self._window_start = now
                    self._tpm_used = 0
                # purge RPM
                self._call_times = [t for t in self._call_times if now - t < 60.0]

                rpm_ok = (len(self._call_times) < self.rpm)
                tpm_ok = (self._tpm_used + need <= self.tpm)

                if rpm_ok and tpm_ok:
                    self._tpm_used += need
                    self._call_times.append(now)
                    return
                # attendre le plus bloquant
                sleep_rpm = 0.0
                if not rpm_ok and self._call_times:
                    sleep_rpm = 60.0 - (now - self._call_times[0]) + 0.01
                sleep_tpm = 0.0
                if not tpm_ok:
                    sleep_tpm = 60.0 - (now - self._window_start) + 0.01
            time.sleep(max(0.05, sleep_rpm, sleep_tpm))

rate_limiter = RateLimiter(AOAI_RPM, AOAI_TPM)

# ============== BLOB HELPERS ==============
def list_blobs_flat(container_client, prefix: str = "", limit: Optional[int] = None) -> List[str]:
    names = []
    for b in container_client.list_blobs(name_starts_with=prefix or ""):
        name = b.name
        if INCLUDE_UPLOADS_ONLY and "/uploads/" not in name.lower():
            continue
        # filtre extensions si on a un fichier final (…/xxx.pdf) ou nom sans extension
        base = name.rsplit("/", 1)[-1].lower()
        ext = base.rsplit(".", 1)[-1] if "." in base else ""
        if EXTS and ext and ext not in EXTS:
            continue
        names.append(name)
        if limit and len(names) >= limit:
            break
    print(f"   → {len(names)} blobs trouvés sous '{prefix}' (uploads_only={INCLUDE_UPLOADS_ONLY})")
    for n in names[:5]:
        print(f"     - {n}")
    if not names and INCLUDE_UPLOADS_ONLY:
        print("   (hint) Aucun blob trouvé avec le filtre '/uploads/'. Mets CV_INCLUDE_UPLOADS_ONLY=0")
    return names

def download_blob_to_bytes(container_client, blob_name: str) -> bytes:
    blob = container_client.get_blob_client(blob_name)
    return blob.download_blob(max_concurrency=MAX_DOWNLOAD_CONCURRENCY).readall()

def upload_json_to_blob(container_client, name: str, data: dict):
    content = json.dumps(data, ensure_ascii=False).encode("utf-8")
    ct = ContentSettings(content_type="application/json; charset=utf-8")
    container_client.upload_blob(name, content, overwrite=True, content_settings=ct)

def upload_text_to_blob(container_client, name: str, text: str, content_type="text/plain; charset=utf-8"):
    content = text.encode("utf-8")
    ct = ContentSettings(content_type=content_type)
    container_client.upload_blob(name, content, overwrite=True, content_settings=ct)

# ============== EXTRACTION TEXTE ==============
def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"

def extract_text_from_pdf_bytes(data: bytes) -> str:
    if _has_pymupdf:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            return "\n".join(page.get_text() or "" for page in doc)
        except Exception:
            pass
    # fallback pypdf via fichier temporaire
    if _has_pypdf:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            try:
                reader = PdfReader(tmp.name)
                return "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception:
                pass
    return ""

def extract_text_from_docx_bytes(data: bytes) -> str:
    if not _has_docx:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(data); tmp.flush()
        try:
            return docx2txt.process(tmp.name) or ""
        except Exception:
            return ""

def extract_cv_text(blob_name: str, raw: bytes) -> str:
    name = blob_name.lower()
    if is_pdf_bytes(raw) or name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(raw)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(raw)
    # txt/unknown → essayer utf-8
    try:
        return raw.decode("utf-8", "ignore")
    except Exception:
        return ""

def simple_clean(text: str) -> str:
    if not text:
        return ""
    # supprime contrôles
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    # espaces multiples
    text = re.sub(r"[ \t]{2,}", " ", text)
    # normalise retours ligne
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()

def parse_json_lenient(content: str) -> Optional[dict]:
    """
    Tente de parser une réponse JSON même si elle est 'bruitée' :
    - supprime d'éventuelles fences ```json ... ```
    - tente json.loads direct
    - extrait le plus grand bloc {...} si du texte entoure le JSON
    - essaie une petite réparation (virgules traînantes)
    Retourne dict si OK, sinon None.
    """
    if not content:
        return None

    s = content.strip()
    # 1) retirer d'éventuelles fences markdown
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)

    # 2) tentative directe
    try:
        return json.loads(s)
    except Exception:
        pass

    # 3) extraire le plus grand bloc JSON {...}
    m = _JSON_BLOCK_RE.search(s)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # 4) réparation mineure : virgules traînantes avant ] ou }
            candidate2 = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(candidate2)
            except Exception:
                pass

    # 4 bis) réparation directe sur la chaîne complète
    s2 = re.sub(r",\s*([\]}])", r"\1", s)
    try:
        return json.loads(s2)
    except Exception:
        return None
# ============== AZURE OPENAI CLIENT ==============
def make_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not endpoint or not api_key:
        print("[CONFIG] Manque AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY.")
        sys.exit(2)
    if not AZURE_OPENAI_DEPLOYMENT:
        print("[CONFIG] AZURE_OPENAI_DEPLOYMENT vide.")
        sys.exit(2)
    print(f"[AZURE] endpoint={endpoint} | deployment={AZURE_OPENAI_DEPLOYMENT} "
          f"| api_version={AZURE_OPENAI_API_VERSION} | rpm={AOAI_RPM} | tpm={AOAI_TPM}")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=AZURE_OPENAI_API_VERSION)

def _sleep_from_429_message(msg: str) -> float:
    m = re.search(r"after\s+(\d+)\s*second", (msg or "").lower())
    if m:
        return float(m.group(1)) + 2.0
    return 8.0  # valeur par défaut

def _call_with_retry(fn, max_attempts: int = 5):
    last = None
    for k in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            s = str(e)
            if "429" in s or "rate limit" in s.lower():
                sleep_s = _sleep_from_429_message(s)
                print(f"   ⏳ 429 rate-limit, retry dans {sleep_s:.1f}s (tentative {k}/{max_attempts})")
                time.sleep(sleep_s)
                continue
            if k < max_attempts:
                back = 2.0 * k
                print(f"   ⏳ Erreur {type(e).__name__}: retry dans {back:.1f}s (tentative {k}/{max_attempts})")
                time.sleep(back)
            else:
                raise
    raise last

# JSON Schema stricte pour la sortie
CV_JSON_SCHEMA = {
  "name": "cv_schema",
  "schema": {
    "type": "object",
    "properties": {
      "has_segmented_cv": {"type": "boolean"},
      "profile": {
        "type": "object",
        "properties": {
          "full_name": {"type": ["string","null"], "maxLength": 200},
          "title_or_role": {"type": ["string","null"], "maxLength": 200},
          "summary": {"type": ["string","null"], "maxLength": 600},
          "location": {"type": ["string","null"], "maxLength": 200}
        }
      },
      "experiences": {
        "type": "array", "maxItems": 20,
        "items": {
          "type": "object",
          "properties": {
            "company": {"type": "string", "maxLength": 200},
            "role": {"type": "string", "maxLength": 200},
            "start_date": {"type": "string", "maxLength": 100},
            "end_date": {"type": ["string","null"], "maxLength": 100},
            "description": {"type": ["string","null"], "maxLength": 700}
          },
          "required": ["company","role","start_date"]
        }
      },
      "education": {
        "type": "array", "maxItems": 20,
        "items": {
          "type": "object",
          "properties": {
            "school": {"type": "string", "maxLength": 200},
            "degree": {"type": ["string","null"], "maxLength": 200},
            "year": {"type": ["string","null"], "maxLength": 100}
          },
          "required": ["school"]
        }
      },
      "skills": {"type": "array", "maxItems": 128, "items": {"type": "string","maxLength": 80}},
      "languages": {"type": "array", "maxItems": 20, "items": {"type": "string","maxLength": 80}}
    },
    "required": ["has_segmented_cv"]
  }
}

# LLM segmentation
_client_singleton: Optional[AzureOpenAI] = None
def get_client() -> AzureOpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = make_client()
    return _client_singleton

def llm_segment(text: str) -> dict:
    client = get_client()
    # Tronquer par tokens pour rester dans TPM
    # On réserve de la marge pour la sortie
    max_prompt_tokens = max(256, AOAI_TPM // max(LLM_WORKERS, 1) - MAX_OUTPUT_TOKENS - 200)
    text_cut = truncate_by_tokens(text[:CV_TEXT_MAX_CHARS], max_prompt_tokens)

    prompt = (
        "Tu extrais un CV en JSON STRICT selon le schéma fourni. "
        "Si une section est absente dans le texte, renvoie un champ vide ([], {}). "
        "N'invente pas d'informations.\n\n"
        "=== TEXTE CV ===\n"
        f"{text_cut}"
    )

    # Le rate-limiter global coordonne les threads
    ptoks = estimate_tokens(prompt)
    rate_limiter.acquire(ptoks, MAX_OUTPUT_TOKENS)

    def _do():
        return client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": CV_JSON_SCHEMA},
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": "You are a careful information extraction system. Output STRICT JSON only."},
                {"role": "user", "content": prompt}
            ],
        )

    resp = _call_with_retry(_do)
    content = resp.choices[0].message.content

    parsed = parse_json_lenient(content)
    if parsed is not None:
        return parsed

# --- Retry optionnel : on élargit le budget si possible ---
    try:
        resp2 = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": CV_JSON_SCHEMA},
            # on double le budget de sortie, borné
            max_tokens=min(MAX_OUTPUT_TOKENS * 2, 4096),
            messages=[
                {"role": "system", "content": "You are a careful information extraction system. Output STRICT JSON only."},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Fournis un JSON COMPLET et VALIDE, sans texte hors JSON."}
            ],
        )
        content2 = resp2.choices[0].message.content
        parsed2 = parse_json_lenient(content2)
        if parsed2 is not None:
            return parsed2
    except Exception:
        pass

    # Si on n'a toujours rien de valide
    return {"raw_parse_error": content}

# ============== HELPERS ID CANDIDAT ==============
CANDIDATE_ID_PATTERNS = [
    re.compile(r"/candidates/(\d+)/", re.IGNORECASE),
    re.compile(r"candidate[_/-](\d+)", re.IGNORECASE),
    re.compile(r"/(\d+)\.(pdf|docx|txt)$", re.IGNORECASE),
]

def candidate_id_from_blob(name: str) -> Optional[str]:
    for rgx in CANDIDATE_ID_PATTERNS:
        m = rgx.search(name)
        if m:
            return m.group(1)
    # fallback: si le dernier segment est numérique
    last = name.rsplit("/", 1)[-1]
    head = last.split(".", 1)[0]
    if head.isdigit():
        return head
    return None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def text_hash_in_blob(container_client, cand_id: str) -> str:
    """Lire l'ancien JSON pour voir si le hash est identique. '' si absent."""
    try:
        dest_json = f"{OUTPUT_BLOB_PREFIX}/json/{cand_id}.json"
        b = container_client.get_blob_client(dest_json)
        if not b.exists():
            return ""
        raw = b.download_blob(max_concurrency=2).readall().decode("utf-8", "ignore")
        obj = json.loads(raw)
        return (obj.get("meta") or {}).get("text_hash", "")
    except Exception:
        return ""

# ============== TRAITEMENT D’UN BLOB ==============
def process_one_blob(container_client, blob_name: str, force: bool = False) -> Dict[str, Any]:
    cand_id = candidate_id_from_blob(blob_name) or "unknown"
    # 1) download
    raw = download_blob_to_bytes(container_client, blob_name)
    # 2) extract & clean
    text = extract_cv_text(blob_name, raw)
    text = simple_clean(text)
    if not text:
        raise RuntimeError("Texte vide après extraction.")

    # 2b) cache par hash
    th = sha1(text)
    if not force:
        prev = text_hash_in_blob(container_client, cand_id)
        if prev and prev == th:
            return {"candidate_id": cand_id, "blob_name": blob_name, "skipped": True, "reason": "unchanged"}

    # 3) LLM
    parsed = llm_segment(text)

    # 4) upload JSON
    meta = {
        "blob_name": blob_name,
        "deployment": AZURE_OPENAI_DEPLOYMENT,
        "api_version": AZURE_OPENAI_API_VERSION,
        "ts": datetime.utcnow().isoformat(),
        "text_hash": th,
    }
    payload = {"candidate_id": cand_id, "parsed": parsed, "meta": meta}
    dest_json = f"{OUTPUT_BLOB_PREFIX}/json/{cand_id}.json"
    upload_json_to_blob(container_client, dest_json, payload)

    return {"candidate_id": cand_id, "blob_name": blob_name, "skipped": False, "json_blob": dest_json,
            "has_segmented_cv": bool(parsed.get("has_segmented_cv"))}

# ============== PIPELINE PRINCIPALE ==============
def run_pipeline(conn_str: str, container: str, prefix: str, limit: Optional[int], force: bool):
    svc = BlobServiceClient.from_connection_string(conn_str)
    cont = svc.get_container_client(container)

    blobs = list_blobs_flat(cont, prefix=prefix, limit=limit)
    print(f"🔎 {len(blobs)} fichiers CV trouvés sous '{container}/{prefix}'")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as exe:
        futmap = {exe.submit(process_one_blob, cont, b, force): b for b in blobs}
        for fut in as_completed(futmap):
            b = futmap[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.get("skipped"):
                    print(f"   ⏭️  {res['candidate_id']} inchangé (skip).")
                else:
                    print(f"   ✅  {res['candidate_id']} → {res['json_blob']}")
            except Exception as e:
                print(f"   ⚠️  Erreur sur {b}: {e}")
                err = {"blob_name": b, "error": str(e), "ts": datetime.utcnow().isoformat()}
                errors.append(err)
                try:
                    err_name = f"{OUTPUT_BLOB_PREFIX}/errors/{int(time.time()*1000)}.json"
                    upload_json_to_blob(cont, err_name, err)
                except Exception:
                    pass

    done = sum(1 for r in results if not r.get("skipped", False))
    skipped = sum(1 for r in results if r.get("skipped", False))
    print(f"\n🏁 Terminé: {done} traités, {skipped} ignorés (cache), {len(errors)} erreurs.")

    # Résumés optionnels
    if SAVE_SUMMARIES:
        # JSON résumé
        summary = {
            "ts": datetime.utcnow().isoformat(),
            "prefix": prefix,
            "limit": limit,
            "force": force,
            "rpm": AOAI_RPM,
            "tpm": AOAI_TPM,
            "workers": LLM_WORKERS,
            "processed": done,
            "skipped": skipped,
            "errors": len(errors),
            "container": container,
        }
        try:
            upload_json_to_blob(cont, f"{OUTPUT_BLOB_PREFIX}/summary/summary_{int(time.time())}.json", summary)
        except Exception:
            pass

        # CSV + JSONL (si pandas dispo)
        if _has_pandas:
            try:
                rows = []
                for r in results:
                    rows.append({
                        "candidate_id": r.get("candidate_id"),
                        "skipped": r.get("skipped", False),
                        "json_blob": r.get("json_blob", ""),
                        "has_segmented_cv": r.get("has_segmented_cv", None),
                    })
                df = pd.DataFrame(rows)
                csv_text = df.to_csv(index=False)
                upload_text_to_blob(cont, f"{OUTPUT_BLOB_PREFIX}/cv_parsed_summary.csv", csv_text, content_type="text/csv; charset=utf-8")
            except Exception:
                pass
        # JSONL complet minimal (payloads ne sont pas rechargés ici pour éviter de gros RAM)
        # Option : on pourrait re-lire chaque JSON pour concaténer ; ici on reste léger.

# ============== PING LLM AVANT LANCEMENT ==============
def ping_llm_or_die():
    try:
        client = get_client()
        msg = "Réponds avec {\"ok\": true}."
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=5,
            messages=[
                {"role": "system", "content": "Réponds uniquement en JSON valide."},
                {"role": "user", "content": msg}
            ],
        )
        content = resp.choices[0].message.content
        json.loads(content)  # doit être JSON
        print("✅ Connexion LLM OK.")
    except Exception as e:
        print(f"❌ Connexion LLM échouée: {e}")
        sys.exit(1)

# ============== CLI ==============
def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline de segmentation CV (Azure Blob + Azure OpenAI)")
    ap.add_argument("--prefix", type=str, default=INPUT_PREFIX, help="Préfixe Blob des CV (par défaut INPUT_PREFIX env)")
    ap.add_argument("--limit", type=int, default=None, help="Limiter le nombre de blobs traités")
    ap.add_argument("--force", action="store_true", help="Forcer re-parse (ignore le cache par hash)")
    return ap.parse_args()

def main():
    if not AZURE_STORAGE_CONNECTION_STRING or not BLOB_CONTAINER:
        print("❌ Manque AZURE_STORAGE_CONNECTION_STRING ou BLOB_CONTAINER.")
        sys.exit(2)

    args = parse_args()
    print(f"[RUN] container={BLOB_CONTAINER} prefix='{args.prefix}' limit={args.limit} force={args.force}")

    # Vérifier la connexion LLM avant
    ping_llm_or_die()

    run_pipeline(
        conn_str=AZURE_STORAGE_CONNECTION_STRING,
        container=BLOB_CONTAINER,
        prefix=args.prefix,
        limit=args.limit,
        force=args.force
    )

if __name__ == "__main__":
    main()
