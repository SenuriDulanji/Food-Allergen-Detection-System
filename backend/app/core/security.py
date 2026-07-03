"""
core/security.py — Security helpers (password hashing and verification).
"""

from passlib.context import CryptContext

# Use pbkdf2_sha256 for password hashing to avoid bcrypt compatibility issues on Windows/Python 3.10+
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hashed version."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a hash of a plain password."""
    return pwd_context.hash(password)
