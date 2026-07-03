"""
main.py — FastAPI application entry point.

Run locally:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs:  http://localhost:8000/docs
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import scan, user

# --------------------------------------------------------------------------- #
#  Logging configuration                                                       #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  FastAPI app                                                                 #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "AI-powered food allergen detection for Sri Lankan dishes. "
        "Uses Gemini Vision + ChromaDB RAG + hybrid allergen detection."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --------------------------------------------------------------------------- #
#  Middleware                                                                  #
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production to your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
#  Routers                                                                     #
# --------------------------------------------------------------------------- #
app.include_router(scan.router, prefix=settings.API_V1_STR)
app.include_router(user.router, prefix=settings.API_V1_STR)

# --------------------------------------------------------------------------- #
#  Root & health endpoints                                                     #
# --------------------------------------------------------------------------- #

@app.get("/", tags=["Health"])
async def root():
    """API health check — confirms the service is running."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "0.3.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Lightweight liveness probe for deployment monitoring."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
#  Startup / shutdown events                                                   #
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def on_startup():
    """Initialize SQLite tables and pre-warm the ChromaDB connection on startup."""
    logger.info("Starting %s…", settings.PROJECT_NAME)
    
    # Initialize SQLite database
    try:
        from app.core.sqlite_db import Base, engine
        import app.models  # Register models in metadata
        Base.metadata.create_all(bind=engine)
        logger.info("SQLite database initialized successfully.")
    except Exception as exc:
        logger.error("Failed to initialize SQLite database: %s", exc)

    # Pre-warm ChromaDB
    try:
        from app.core.database import get_chroma_client
        client = get_chroma_client()
        text_collection = client.get_collection(settings.COLLECTION_NAME)
        image_collection = client.get_collection(settings.IMAGE_COLLECTION_NAME)
        logger.info(
            "ChromaDB ready — text collection '%s' has %d documents, image collection '%s' has %d documents.",
            settings.COLLECTION_NAME, text_collection.count(),
            settings.IMAGE_COLLECTION_NAME, image_collection.count()
        )
    except Exception as exc:
        logger.warning("ChromaDB pre-warm failed (non-fatal): %s", exc)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("%s shutting down.", settings.PROJECT_NAME)


# --------------------------------------------------------------------------- #
#  Local dev entry point                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)