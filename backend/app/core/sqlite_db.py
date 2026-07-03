"""
sqlite_db.py — SQLAlchemy database connection manager.

Provides the engine, session local, and base class for SQLite,
along with the get_db dependency for FastAPI endpoints.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# For SQLite, we need connect_args={"check_same_thread": False}
# to allow multi-threaded access in FastAPI development.
engine = create_engine(
    settings.SQLITE_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """Dependency that yields a database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
