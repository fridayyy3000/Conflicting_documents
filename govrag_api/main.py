"""GOV-RAG FastAPI service with dynamic multi-corpus support."""

from __future__ import annotations

import glob
import os
import threading
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from corpus_store import CorpusNotFoundError, CorpusStore, CorpusStoreError, DocumentNotFoundError
from document_parser import (
    DocumentParseError,
    make_normalized_filename,
    parse_document_bytes,
    sanitize_filename,
    validate_extension,
)
from gov_rag_gemini import GovRAGGemini
from rate_limit import RateLimiter


# ============================================================
# CONFIG
# ============================================================

API_SECRET = os.getenv("GOVRAG_API_SECRET")
MAX_FILE_SIZE_MB = int(os.getenv("GOVRAG_MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DOCUMENTS_DIR = os.path.join(BASE_DIR, "easy")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GOVRAG_BUCKET = os.getenv("GOVRAG_BUCKET", "").strip() or None

# Server-controlled Vertex AI target for the public browser API; never client-supplied.
DEMO_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "project-79920195-9e86-44ea-8c9")
DEMO_REGION = os.getenv("VERTEX_AI_REGION", "us-central1")

DEMO_CORPUS_TTL_HOURS = float(os.getenv("DEMO_CORPUS_TTL_HOURS", "24"))
MAX_DEMO_DOCUMENTS = int(os.getenv("GOVRAG_MAX_DEMO_DOCUMENTS", "500"))
MAX_DEMO_CORPUS_MB = int(os.getenv("GOVRAG_MAX_DEMO_CORPUS_MB", "500"))
MAX_DEMO_CORPUS_BYTES = MAX_DEMO_CORPUS_MB * 1024 * 1024
MAX_QUESTION_LENGTH = int(os.getenv("GOVRAG_MAX_QUESTION_LENGTH", "2000"))

DEMO_RATE_LIMIT_PER_MINUTE = int(os.getenv("GOVRAG_DEMO_RATE_LIMIT_PER_MINUTE", "30"))
demo_rate_limiter = RateLimiter(max_requests=DEMO_RATE_LIMIT_PER_MINUTE, window_seconds=60.0)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("GOVRAG_ALLOWED_ORIGINS", "https://app.corvic.ai").split(",")
    if origin.strip()
]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="GOV-RAG API",
    description="Authority-aware RAG with conflict resolution",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    question: str
    project_id: str = "project-79920195-9e86-44ea-8c9"
    region: str = "us-central1"
    use_llm: bool = True
    top_k: int = 8


class CorpusCreateRequest(BaseModel):
    name: str


class CorpusIndexRequest(BaseModel):
    project_id: str = "project-79920195-9e86-44ea-8c9"
    region: str = "us-central1"
    use_llm: bool = True


class DemoCorpusCreateRequest(BaseModel):
    name: str = "Demo Workspace"


class DemoQueryRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH)


class SourceInfo(BaseModel):
    filename: str
    claim: Optional[str] = None
    status: str
    document_type: str
    semantic_score: float
    authority_score: float
    scope_score: float
    final_score: float
    supporting_sentence: Optional[str] = None
    selected: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    selected_source: Optional[str]
    reason: str
    conflict_detected: bool
    confidence: str
    num_conflicts: int
    top_sources: List[SourceInfo]
    retrieval_mode: Optional[str] = None


class CorpusQueryResponse(QueryResponse):
    corpus_id: str


# ============================================================
# GLOBAL STATE
# ============================================================

demo_rag_instance = None
demo_rag_config = None

corpus_store = CorpusStore(base_dir=BASE_DIR, bucket_name=GOVRAG_BUCKET)

# Cache entries keyed by "corpus_id|project|region|use_llm".
rag_instances: Dict[str, GovRAGGemini] = {}
rag_instance_metadata: Dict[str, Dict] = {}

cache_lock = threading.Lock()
corpus_locks: Dict[str, threading.Lock] = {}


# ============================================================
# HELPERS
# ============================================================

def require_api_key(x_api_key: Optional[str]):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def validate_corpus_id(corpus_id: str) -> None:
    try:
        parsed = uuid.UUID(corpus_id)
        if str(parsed) != corpus_id:
            raise ValueError("mismatch")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid corpus_id") from exc


def get_corpus_lock(corpus_id: str) -> threading.Lock:
    with cache_lock:
        if corpus_id not in corpus_locks:
            corpus_locks[corpus_id] = threading.Lock()
        return corpus_locks[corpus_id]


def build_cache_key(corpus_id: str, request: QueryRequest) -> str:
    return f"{corpus_id}|{request.project_id}|{request.region}|{int(request.use_llm)}"


def invalidate_corpus_cache(corpus_id: str) -> None:
    with cache_lock:
        keys = [k for k in rag_instances if k.startswith(f"{corpus_id}|")]
        for key in keys:
            rag_instances.pop(key, None)
            rag_instance_metadata.pop(key, None)


def get_client_key(http_request: Request) -> str:
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def enforce_demo_rate_limit(http_request: Request) -> None:
    if not demo_rate_limiter.allow(get_client_key(http_request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")


def get_demo_corpus_or_404(corpus_id: str) -> Dict:
    """Resolve a demo-kind corpus, treating expired/foreign-kind corpora as not found."""
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
    except CorpusNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Corpus not found") from exc

    if metadata.get("kind") != "demo":
        raise HTTPException(status_code=404, detail="Corpus not found")

    if corpus_store.is_expired(metadata):
        try:
            corpus_store.delete_corpus(corpus_id)
        except CorpusStoreError:
            pass
        invalidate_corpus_cache(corpus_id)
        raise HTTPException(status_code=404, detail="Corpus not found or expired")

    return metadata


def get_demo_rag_instance(request: QueryRequest) -> GovRAGGemini:
    global demo_rag_instance, demo_rag_config

    if not os.path.exists(DEMO_DOCUMENTS_DIR):
        raise HTTPException(
            status_code=500,
            detail=f"Documents directory not found: {DEMO_DOCUMENTS_DIR}",
        )

    config = (request.project_id, request.region, request.use_llm)
    if demo_rag_instance is None or demo_rag_config != config:
        print("Initializing demo GOV-RAG...")
        print(f"Documents: {DEMO_DOCUMENTS_DIR}")

        demo_rag_instance = GovRAGGemini(
            doc_dir=DEMO_DOCUMENTS_DIR,
            project_id=request.project_id,
            region=request.region,
            use_llm=request.use_llm,
        )
        demo_rag_config = config
        print("Demo GOV-RAG initialized.")

    return demo_rag_instance


def ensure_corpus_rag(corpus_id: str, request: QueryRequest, force_reindex: bool = False):
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
    except CorpusNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Corpus not found") from exc

    if metadata.get("document_count", 0) == 0:
        raise HTTPException(status_code=409, detail="Corpus has no documents")

    key = build_cache_key(corpus_id, request)
    lock = get_corpus_lock(corpus_id)

    with lock:
        metadata = corpus_store.get_corpus(corpus_id)
        cached_meta = rag_instance_metadata.get(key)
        can_reuse = (
            not force_reindex
            and key in rag_instances
            and cached_meta is not None
            and cached_meta.get("updated_at") == metadata.get("updated_at")
        )

        if can_reuse:
            return rag_instances[key], cached_meta

        materialized = corpus_store.materialize_corpus(corpus_id)
        rag = GovRAGGemini(
            doc_dir=materialized["local_dir"],
            project_id=request.project_id,
            region=request.region,
            use_llm=request.use_llm,
        )

        ready_meta = corpus_store.mark_ready(corpus_id)
        cache_meta = {
            "filename_map": materialized["filename_map"],
            "local_dir": materialized["local_dir"],
            "updated_at": ready_meta.get("updated_at"),
            "document_count": ready_meta.get("document_count", 0),
        }

        rag_instances[key] = rag
        rag_instance_metadata[key] = cache_meta
        return rag, cache_meta


def format_result(question: str, result: dict, top_k: int = 8, filename_map: Optional[Dict[str, str]] = None):
    """Map GovRAGGemini's internal chunk-level candidates to the public API shape.

    Retrieval operates over chunks internally, so multiple candidates can share
    the same source file; dedupe to one row per source (keeping its strongest
    chunk) so the UI doesn't show repeated identical-looking rows for one PDF.
    """
    filename_map = filename_map or {}
    selected_source_raw = result.get("source")
    selected_source = filename_map.get(selected_source_raw, selected_source_raw)
    conflict_detected = result.get("conflict_detected", False)

    candidates = result.get("candidates", [])

    # Keep only the highest-scoring candidate per source filename.
    best_by_file: Dict[str, dict] = {}
    for src in candidates:
        raw_filename = src.get("filename", "")
        current_best = best_by_file.get(raw_filename)
        if current_best is None or src.get("final_score", 0.0) > current_best.get("final_score", 0.0):
            best_by_file[raw_filename] = src

    deduped = sorted(best_by_file.values(), key=lambda c: c.get("final_score", 0.0), reverse=True)

    top_sources = []
    for src in deduped[:top_k]:
        raw_filename = src.get("filename", "")
        display_filename = filename_map.get(raw_filename, raw_filename)

        top_sources.append(
            SourceInfo(
                filename=display_filename,
                claim=src.get("claim"),
                status=src.get("status", "unknown"),
                document_type=src.get("doc_type", "unknown"),
                semantic_score=float(src.get("semantic_score", 0.0)),
                authority_score=float(src.get("authority_score", 0.0)),
                scope_score=float(src.get("scope_score", 0.0)),
                final_score=float(src.get("final_score", 0.0)),
                supporting_sentence=src.get("claim_sentence"),
                selected=(raw_filename == selected_source_raw),
            )
        )

    if conflict_detected:
        # Claim extraction is auxiliary metadata; only count distinct claims
        # among deduped sources when the pipeline actually flagged a conflict,
        # so incidental non-conflicting extracted numbers never show as "3
        # conflicts" on ordinary QA.
        claims = {
            str(src.get("claim")).strip()
            for src in deduped
            if src.get("claim") is not None
        }
        num_conflicts = max(len(claims) - 1, 0)
    else:
        num_conflicts = 0

    return QueryResponse(
        question=question,
        answer=result.get("answer"),
        selected_source=selected_source,
        reason=result.get("reason", ""),
        conflict_detected=conflict_detected,
        confidence=result.get("confidence", "unknown"),
        num_conflicts=num_conflicts,
        top_sources=top_sources,
        retrieval_mode=result.get("retrieval_mode"),
    )


def run_demo_query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    rag = get_demo_rag_instance(request)
    result = rag.query(question)
    return format_result(question, result, request.top_k)


def run_corpus_query(corpus_id: str, request: QueryRequest) -> CorpusQueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    rag, cache_meta = ensure_corpus_rag(corpus_id, request, force_reindex=False)
    result = rag.query(question)
    formatted = format_result(question, result, request.top_k, filename_map=cache_meta.get("filename_map", {}))

    return CorpusQueryResponse(corpus_id=corpus_id, **formatted.model_dump())


def map_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CorpusNotFoundError):
        return HTTPException(status_code=404, detail="Corpus not found")
    if isinstance(exc, DocumentNotFoundError):
        return HTTPException(status_code=404, detail="Document not found")
    if isinstance(exc, DocumentParseError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, CorpusStoreError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PUBLIC ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "GOV-RAG API",
        "status": "running",
        "version": "2.0.0",
    }


# Static assets (app.css/app.js) for the DocuResolve frontend, same-origin as the API.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/app")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "app.html"))


@app.get("/health")
async def health():
    demo_docs_exist = os.path.exists(DEMO_DOCUMENTS_DIR)
    return {
        "status": "healthy" if demo_docs_exist else "degraded",
        "demo_documents_directory": DEMO_DOCUMENTS_DIR,
        "demo_documents_available": demo_docs_exist,
        "demo_rag_initialized": demo_rag_instance is not None,
        "corpus_backend": "gcs" if GOVRAG_BUCKET else "local",
        "govrag_bucket": GOVRAG_BUCKET,
    }


@app.post("/demo_query", response_model=QueryResponse)
async def demo_query(request: QueryRequest):
    try:
        return run_demo_query(request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"DEMO QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PROTECTED DEMO ENDPOINTS
# ============================================================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        return run_demo_query(request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch_query")
async def batch_query(
    questions: List[str],
    project_id: str = "project-79920195-9e86-44ea-8c9",
    region: str = "us-central1",
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)

    try:
        request = QueryRequest(question="placeholder", project_id=project_id, region=region, use_llm=True, top_k=8)
        rag = get_demo_rag_instance(request)

        results = []
        for question in questions:
            clean_question = question.strip()
            if not clean_question:
                continue
            result = rag.query(clean_question)
            results.append(format_result(clean_question, result, 8))

        return results
    except HTTPException:
        raise
    except Exception as exc:
        print(f"BATCH ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# CORPUS ENDPOINTS (PROTECTED)
# ============================================================

@app.post("/corpora")
async def create_corpus(payload: CorpusCreateRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        corpus_id = str(uuid.uuid4())
        metadata = corpus_store.create_corpus(corpus_id=corpus_id, name=payload.name)
        return {
            "corpus_id": metadata["corpus_id"],
            "name": metadata["name"],
            "status": metadata["status"],
            "document_count": metadata["document_count"],
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora")
async def list_corpora(x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        corpora = corpus_store.list_corpora()
        return {
            "count": len(corpora),
            "corpora": [
                {
                    "corpus_id": c.get("corpus_id"),
                    "name": c.get("name"),
                    "status": c.get("status"),
                    "document_count": c.get("document_count", 0),
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                }
                for c in corpora
            ],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora/{corpus_id}")
async def get_corpus(corpus_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
        return {
            "corpus_id": metadata.get("corpus_id"),
            "name": metadata.get("name"),
            "status": metadata.get("status"),
            "document_count": metadata.get("document_count", 0),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "documents": metadata.get("documents", []),
        }
    except Exception as exc:
        raise map_store_error(exc)


async def handle_document_uploads(
    corpus_id: str,
    files: List[UploadFile],
    max_files: Optional[int] = None,
    max_total_size_bytes: Optional[int] = None,
) -> Dict:
    """Shared upload path for both the protected /corpora and public /demo/corpora APIs."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    existing_docs = corpus_store.list_documents(corpus_id)
    if max_files is not None and len(existing_docs) + len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Corpus is limited to {max_files} files")

    running_total = sum(d.get("size_bytes", 0) for d in existing_docs)
    pending_docs = []
    uploaded = []

    for upload in files:
        safe_filename = sanitize_filename(upload.filename or "")
        extension = validate_extension(safe_filename)
        content = await upload.read()
        size_bytes = len(content)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail=f"File is empty: {safe_filename}")
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds max size ({MAX_FILE_SIZE_MB} MB): {safe_filename}",
            )

        running_total += size_bytes
        if max_total_size_bytes is not None and running_total > max_total_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Corpus exceeds max total size ({max_total_size_bytes // (1024 * 1024)} MB)",
            )

        normalized_text = parse_document_bytes(content=content, extension=extension)
        if not normalized_text.strip():
            raise HTTPException(status_code=400, detail=f"No extractable text in file: {safe_filename}")

        document_id = str(uuid.uuid4())
        _, normalized_filename = make_normalized_filename(safe_filename, document_id)

        pending_docs.append({
            "document_id": document_id,
            "original_filename": safe_filename,
            "normalized_filename": normalized_filename,
            "file_type": extension,
            "size_bytes": size_bytes,
            "normalized_text": normalized_text,
        })
        uploaded.append({
            "document_id": document_id,
            "filename": safe_filename,
            "status": "uploaded",
        })

    # Single metadata.json write for the whole batch (GCS rate-limits per-object
    # mutations to ~1/sec, so writing once-per-file 429s on large batches).
    metadata = corpus_store.add_documents_bulk(corpus_id, pending_docs)

    # Uploading invalidates any cached GovRAGGemini instance for this corpus.
    invalidate_corpus_cache(corpus_id)

    return {
        "corpus_id": corpus_id,
        "uploaded": uploaded,
        "document_count": metadata.get("document_count", 0),
        "requires_reindex": True,
        "status": metadata.get("status"),
    }


@app.post("/corpora/{corpus_id}/documents")
async def upload_documents(
    corpus_id: str,
    files: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        return await handle_document_uploads(corpus_id, files)
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora/{corpus_id}/documents")
async def list_documents(corpus_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        documents = corpus_store.list_documents(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_count": len(documents),
            "documents": documents,
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.delete("/corpora/{corpus_id}/documents/{document_id}")
async def delete_document(corpus_id: str, document_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.delete_document(corpus_id, document_id)
        invalidate_corpus_cache(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_id": document_id,
            "status": "deleted",
            "document_count": metadata.get("document_count", 0),
            "requires_reindex": True,
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/corpora/{corpus_id}/index")
async def index_corpus(
    corpus_id: str,
    payload: CorpusIndexRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        request = QueryRequest(
            question="index",
            project_id=payload.project_id,
            region=payload.region,
            use_llm=payload.use_llm,
            top_k=8,
        )
        _, cache_meta = ensure_corpus_rag(corpus_id, request, force_reindex=True)
        return {
            "corpus_id": corpus_id,
            "status": "ready",
            "document_count": cache_meta.get("document_count", 0),
            "message": "Corpus indexed successfully",
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/corpora/{corpus_id}/query", response_model=CorpusQueryResponse)
async def query_corpus(
    corpus_id: str,
    request: QueryRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        return run_corpus_query(corpus_id, request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"CORPUS QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PUBLIC DEMO CORPUS API (for the Corvic browser Web App)
#
# No X-API-Key is required or accepted here. Safety instead relies on:
#   - unguessable corpus_id (uuid4, acts as a bearer capability)
#   - no endpoint that lists all demo corpora
#   - short TTL + opportunistic expiry cleanup
#   - file count/type/size caps and per-IP rate limiting
#   - server-controlled project_id/region (never taken from the client)
# These routes reuse the exact same corpus_store and GovRAGGemini pipeline
# as the protected /corpora endpoints; no RAG logic is duplicated here.
# ============================================================

@app.post("/demo/corpora")
async def create_demo_corpus(payload: DemoCorpusCreateRequest, http_request: Request):
    enforce_demo_rate_limit(http_request)

    # Best-effort cleanup of expired demo corpora; not a scheduled job.
    try:
        corpus_store.delete_expired_corpora(kind="demo")
    except Exception as exc:
        print(f"DEMO CLEANUP WARNING: {exc}")

    try:
        corpus_id = str(uuid.uuid4())
        metadata = corpus_store.create_corpus(
            corpus_id=corpus_id,
            name=payload.name,
            kind="demo",
            ttl_hours=DEMO_CORPUS_TTL_HOURS,
        )
        return {
            "corpus_id": metadata["corpus_id"],
            "name": metadata["name"],
            "status": metadata["status"],
            "document_count": metadata["document_count"],
            "created_at": metadata["created_at"],
            "expires_at": metadata["expires_at"],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/demo/corpora/{corpus_id}")
async def get_demo_corpus(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    metadata = get_demo_corpus_or_404(corpus_id)

    return {
        "corpus_id": metadata.get("corpus_id"),
        "name": metadata.get("name"),
        "status": metadata.get("status"),
        "document_count": metadata.get("document_count", 0),
        "created_at": metadata.get("created_at"),
        "expires_at": metadata.get("expires_at"),
    }


@app.post("/demo/corpora/{corpus_id}/documents")
async def upload_demo_documents(
    corpus_id: str,
    http_request: Request,
    files: List[UploadFile] = File(...),
):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        return await handle_document_uploads(
            corpus_id,
            files,
            max_files=MAX_DEMO_DOCUMENTS,
            max_total_size_bytes=MAX_DEMO_CORPUS_BYTES,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


def load_example_documents(corpus_id: str) -> Dict:
    """One-click load of the bundled 180-document Easy pack into a demo corpus."""
    paths = sorted(glob.glob(os.path.join(DEMO_DOCUMENTS_DIR, "*.md")))
    if not paths:
        raise HTTPException(status_code=500, detail="Example dataset not available")

    existing_docs = corpus_store.list_documents(corpus_id)
    if len(existing_docs) + len(paths) > MAX_DEMO_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Corpus is limited to {MAX_DEMO_DOCUMENTS} files")

    running_total = sum(d.get("size_bytes", 0) for d in existing_docs)
    pending_docs = []
    uploaded = []

    for path in paths:
        text = open(path, "r", encoding="utf-8").read()
        size_bytes = len(text.encode("utf-8"))

        running_total += size_bytes
        if running_total > MAX_DEMO_CORPUS_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Corpus exceeds max total size ({MAX_DEMO_CORPUS_MB} MB)",
            )

        original_filename = os.path.basename(path)
        document_id = str(uuid.uuid4())
        _, normalized_filename = make_normalized_filename(original_filename, document_id)

        pending_docs.append({
            "document_id": document_id,
            "original_filename": original_filename,
            "normalized_filename": normalized_filename,
            "file_type": ".md",
            "size_bytes": size_bytes,
            "normalized_text": text,
        })
        uploaded.append({
            "document_id": document_id,
            "filename": original_filename,
            "status": "uploaded",
        })

    # Single metadata.json write for all 180 files (see handle_document_uploads).
    metadata = corpus_store.add_documents_bulk(corpus_id, pending_docs)
    invalidate_corpus_cache(corpus_id)

    return {
        "corpus_id": corpus_id,
        "uploaded": uploaded,
        "document_count": metadata.get("document_count", 0),
        "requires_reindex": True,
        "status": metadata.get("status"),
    }


@app.post("/demo/corpora/{corpus_id}/load_example")
async def load_example_dataset(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        return load_example_documents(corpus_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/demo/corpora/{corpus_id}/documents")
async def list_demo_documents(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        documents = corpus_store.list_documents(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_count": len(documents),
            "documents": [
                {
                    "document_id": d.get("document_id"),
                    "filename": d.get("original_filename"),
                    "file_type": d.get("file_type"),
                    "uploaded_at": d.get("uploaded_at"),
                    "size_bytes": d.get("size_bytes"),
                }
                for d in documents
            ],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.delete("/demo/corpora/{corpus_id}")
async def delete_demo_corpus(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        corpus_store.delete_corpus(corpus_id)
        invalidate_corpus_cache(corpus_id)
        return {"corpus_id": corpus_id, "status": "deleted"}
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/demo/corpora/{corpus_id}/query", response_model=CorpusQueryResponse)
async def query_demo_corpus(
    corpus_id: str,
    payload: DemoQueryRequest,
    http_request: Request,
):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    # project_id/region/use_llm are fixed server-side; never taken from the browser.
    request = QueryRequest(
        question=payload.question,
        project_id=DEMO_PROJECT_ID,
        region=DEMO_REGION,
        use_llm=True,
        top_k=8,
    )

    try:
        return run_corpus_query(corpus_id, request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"DEMO CORPUS QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )