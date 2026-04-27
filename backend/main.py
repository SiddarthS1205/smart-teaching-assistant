"""
Smart Teaching Assistant — FastAPI Application Entry Point

Architecture:
  MCP Tools → Agents (QA / Summarizer / Router) → Guardrails → API
"""
import os
import time
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from config import settings
from observability import get_logger, get_query_log

# ── MCP Tools ─────────────────────────────────────────────────────────────────
from tools.document_loader import DocumentLoader
from tools.text_chunker import TextChunker
from tools.embedding_generator import EmbeddingGenerator
from tools.vector_database import VectorDatabase
from tools.retriever import Retriever
from tools.llm_generator import LLMGenerator
from tools.summarizer import Summarizer

# ── Agents ────────────────────────────────────────────────────────────────────
from agents.qa_agent import QAAgent
from agents.summarizer_agent import SummarizerAgent
from agents.router_agent import RouterAgent

# ── Guardrails ────────────────────────────────────────────────────────────────
from guardrails.validator import QueryValidator, UploadValidator, ValidationError

logger = get_logger(__name__)

# ── Ensure required directories exist ────────────────────────────────────────
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)

# ── Instantiate MCP tools (singletons shared across requests) ─────────────────
doc_loader = DocumentLoader()
chunker = TextChunker()
embedder = EmbeddingGenerator()
vector_db = VectorDatabase()
retriever = Retriever(embedder, vector_db)
generator = LLMGenerator()
summarizer_tool = Summarizer()

# ── Instantiate Agents ────────────────────────────────────────────────────────
qa_agent = QAAgent(retriever, generator)
summarizer_agent = SummarizerAgent(summarizer_tool, vector_db)
router = RouterAgent(qa_agent, summarizer_agent)

# ── Guardrail validators ──────────────────────────────────────────────────────
query_validator = QueryValidator()
upload_validator = UploadValidator()

# ── App state ─────────────────────────────────────────────────────────────────
app_state: dict = {
    "document_loaded": False,
    "document_name": None,
    "document_pages": 0,
    "chunk_count": 0,
}


# ── Lifespan: try to restore a previously persisted index ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Smart Teaching Assistant starting up…")
    if vector_db.load():
        app_state["document_loaded"] = True
        app_state["chunk_count"] = len(vector_db._chunks)
        logger.info("📂 Restored previous vector index (%d chunks)", app_state["chunk_count"])
    yield
    logger.info("🛑 Smart Teaching Assistant shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Teaching Assistant",
    description="RAG + MCP + Agentic System for academic Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# ── Simple HTTP Basic Auth ────────────────────────────────────────────────────
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, settings.DEMO_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, settings.DEMO_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ── Helper: validation error → HTTP 422 ──────────────────────────────────────
def _handle_validation_error(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": exc.message, "code": exc.code},
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the login page."""
    login_page = frontend_path / "index.html"
    if login_page.exists():
        return FileResponse(str(login_page))
    return {"message": "Smart Teaching Assistant API", "docs": "/docs"}


@app.get("/app", include_in_schema=False)
async def serve_app():
    """Serve the main application page."""
    app_page = frontend_path / "app.html"
    if app_page.exists():
        return FileResponse(str(app_page))
    return {"message": "App page not found"}


@app.post("/upload", summary="Upload and process a PDF document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract text, chunk it, embed it, and store in FAISS.

    - Validates file type and size (guardrails)
    - Runs the full MCP pipeline: load → chunk → embed → index
    - Returns processing statistics
    """
    start = time.perf_counter()
    logger.info("📤 Upload request received: %s", file.filename)

    # Read file bytes for size check
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Guardrail: validate upload
    try:
        upload_validator.validate(file.filename, file_size)
    except ValidationError as exc:
        logger.warning("Upload rejected: %s", exc.message)
        return _handle_validation_error(exc)

    # Save to disk
    save_path = Path(settings.UPLOAD_DIR) / file.filename
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    logger.info("💾 File saved: %s (%d bytes)", save_path, file_size)

    try:
        # MCP Pipeline
        # 1. Load document
        text = doc_loader.load(str(save_path))
        metadata = doc_loader.load_metadata(str(save_path))

        # 2. Chunk text
        chunks = chunker.chunk(text)

        # 3. Generate embeddings
        embeddings = embedder.embed(chunks)

        # 4. Build vector index
        vector_db.clear()
        vector_db.build(embeddings, chunks)

        # Update app state
        app_state.update({
            "document_loaded": True,
            "document_name": file.filename,
            "document_pages": metadata["pages"],
            "chunk_count": len(chunks),
        })

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("✅ Document processed in %.2f ms", elapsed)

        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully.",
            "stats": {
                "filename": file.filename,
                "pages": metadata["pages"],
                "chunks": len(chunks),
                "processing_time_ms": round(elapsed, 2),
            },
        }

    except Exception as exc:
        logger.error("❌ Processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(exc)}")


@app.get("/ask", summary="Ask a question about the uploaded document")
async def ask_question(q: str):
    """
    Ask a question. The Router Agent decides whether to use QA or Summarizer.

    - Validates query (guardrails)
    - Routes to appropriate agent
    - Returns answer with metadata
    """
    start = time.perf_counter()
    logger.info("❓ Question received: %s…", q[:80])

    # Guardrail: validate query
    try:
        clean_query = query_validator.validate(q)
    except ValidationError as exc:
        return _handle_validation_error(exc)

    if not app_state["document_loaded"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No document uploaded yet. Please upload a PDF first.",
                "code": "NO_DOCUMENT",
            },
        )

    try:
        # Router Agent decides which agent to use
        agent_name = router.route(clean_query)
        answer = router.run(clean_query)

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "status": "success",
            "query": clean_query,
            "answer": answer,
            "agent_used": agent_name,
            "document": app_state["document_name"],
            "response_time_ms": round(elapsed, 2),
        }

    except Exception as exc:
        logger.error("❌ Query processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(exc)}")


@app.get("/summary", summary="Get a structured summary of the uploaded document")
async def get_summary():
    """
    Generate a comprehensive academic summary of the uploaded document.
    Uses the Summarizer Agent directly.
    """
    start = time.perf_counter()
    logger.info("📋 Summary request received")

    if not app_state["document_loaded"]:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No document uploaded yet. Please upload a PDF first.",
                "code": "NO_DOCUMENT",
            },
        )

    try:
        summary = summarizer_agent.run("Generate a summary of the document")
        elapsed = (time.perf_counter() - start) * 1000

        return {
            "status": "success",
            "summary": summary,
            "document": app_state["document_name"],
            "response_time_ms": round(elapsed, 2),
        }

    except Exception as exc:
        logger.error("❌ Summary generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(exc)}")


@app.get("/status", summary="Get system status")
async def get_status():
    """Return current system state and document info."""
    return {
        "status": "online",
        "document_loaded": app_state["document_loaded"],
        "document_name": app_state["document_name"],
        "document_pages": app_state["document_pages"],
        "chunk_count": app_state["chunk_count"],
        "vector_db_ready": vector_db.is_ready(),
    }


@app.get("/logs", summary="Get recent query logs (observability)")
async def get_logs():
    """Return the in-memory query log for observability."""
    return {
        "status": "success",
        "total_queries": len(get_query_log()),
        "queries": get_query_log()[-50:],  # Last 50 entries
    }


@app.post("/login", summary="Authenticate user")
async def login(credentials: dict):
    """
    Validate username/password and return a session token.
    (Demo implementation — use JWT + proper auth in production.)
    """
    username = credentials.get("username", "")
    password = credentials.get("password", "")

    if (
        secrets.compare_digest(username, settings.DEMO_USERNAME)
        and secrets.compare_digest(password, settings.DEMO_PASSWORD)
    ):
        logger.info("✅ Login successful for user: %s", username)
        return {"status": "success", "message": "Login successful", "username": username}

    logger.warning("🛡️  Failed login attempt for user: %s", username)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password",
    )


@app.delete("/document", summary="Clear the current document")
async def clear_document():
    """Remove the current document and reset the vector store."""
    vector_db.clear()
    app_state.update({
        "document_loaded": False,
        "document_name": None,
        "document_pages": 0,
        "chunk_count": 0,
    })
    logger.info("🗑️  Document cleared")
    return {"status": "success", "message": "Document cleared successfully."}


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
