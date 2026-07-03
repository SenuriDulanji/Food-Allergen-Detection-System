"""
config.py — Application settings loaded from the .env file.

Uses pydantic-settings so every value is type-validated at startup.
The .env file must sit at the **project root** (one level above /backend).
"""

from pathlib import Path
from pydantic_settings import BaseSettings

# File path: <project_root>/backend/app/core/config.py
# parents[0]=core, parents[1]=app, parents[2]=backend, parents[3]=project_root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    #  General                                                             #
    # ------------------------------------------------------------------ #
    PROJECT_NAME: str = "Sri Lankan Allergen Detector"
    API_V1_STR: str = "/api/v1"

    # ------------------------------------------------------------------ #
    #  Gemini                                                              #
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # ------------------------------------------------------------------ #
    #  ChromaDB                                                            #
    # ------------------------------------------------------------------ #
    COLLECTION_NAME: str = "sri_lankan_recipes_text"
    IMAGE_COLLECTION_NAME: str = "sri_lankan_recipes_images"
    PERSIST_DIRECTORY: str = str(PROJECT_ROOT / "chroma_db")
    SQLITE_DATABASE_URL: str = f"sqlite:///{PROJECT_ROOT}/sri_lankan_allergen_detector.db"

    # ------------------------------------------------------------------ #
    #  RAG tuning                                                          #
    # ------------------------------------------------------------------ #
    # How many recipe chunks to retrieve per query
    RAG_TOP_K: int = 3

    # Maximum image file size accepted by the scan endpoint (bytes)
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        extra = "ignore"


settings = Settings()