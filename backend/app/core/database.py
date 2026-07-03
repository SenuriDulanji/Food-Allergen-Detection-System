"""
database.py — ChromaDB connection manager (singleton pattern).

Uses a single persistent client throughout the application lifecycle
to avoid re-opening the database on every request.
"""

from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

# --------------------------------------------------------------------------- #
#  Singleton holder — initialised once on first call                          #
# --------------------------------------------------------------------------- #
_chroma_client: Optional[chromadb.PersistentClient] = None

def get_chroma_client() -> chromadb.PersistentClient:
    """Return (cached) persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.PERSIST_DIRECTORY,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client