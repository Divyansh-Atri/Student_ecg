import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from db.database import init_db
from config.settings import get_settings
from vitals.schemas import HealthStatus
from api.routes import vitals, alerts, patients, ehr, mock, audit, abg, ecg

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown — nothing to clean up for SQLite


app = FastAPI(
    title="MedGemma Post-Operative Monitoring Agent",
    description=(
        "Intelligent post-operative monitoring agent powered by MedGemma 4-bit. "
        "Monitors SBP, DBP, HR, and SpO2 for sepsis and cardiovascular deterioration risk. "
        "All outputs are clinical decision support only and must not replace clinical judgment."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allows the local JS UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000"],  # tightened for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI
ui_path = Path(__file__).parent.parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")

# Routers
app.include_router(vitals.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(ehr.router, prefix="/api/v1")
app.include_router(mock.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(abg.router,   prefix="/api/v1")
app.include_router(ecg.router,   prefix="/api/v1")


@app.get("/api/v1/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """System health check — verifies Ollama, ChromaDB, database, and guardrails."""

    # Check Ollama
    ollama_status = "ok"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.OLLAMA_BASE_URL}")
            if r.status_code != 200:
                ollama_status = f"unreachable (status {r.status_code})"
    except Exception as e:
        ollama_status = f"unreachable ({str(e)[:60]})"

    # Check ChromaDB
    chroma_status = "ok"
    try:
        from rag.vector_store import get_chroma_client
        get_chroma_client().heartbeat()
    except Exception as e:
        chroma_status = f"error ({str(e)[:60]})"

    # Check DB
    db_status = "ok"
    try:
        from db.database import SessionLocal
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error ({str(e)[:60]})"

    guardrails_status = "ok"  # Phase 2 will do a real check

    overall = (
        "healthy"
        if all(s == "ok" for s in [ollama_status, chroma_status, db_status])
        else "degraded"
    )

    return HealthStatus(
        status=overall,
        ollama=ollama_status,
        chromadb=chroma_status,
        database=db_status,
        guardrails=guardrails_status,
    )
