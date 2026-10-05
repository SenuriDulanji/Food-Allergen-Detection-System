"""
models/user.py — SQLAlchemy models for User and UserAllergen.
"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.core.sqlite_db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Demographic data
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    province = Column(String, nullable=True)
    blood_type = Column(String, nullable=True)
    dietary_pattern = Column(String, nullable=True)
    medical_conditions = Column(String, nullable=True) # Stored as comma-separated
    lactose_intolerance = Column(Boolean, nullable=True)
    outside_food_frequency = Column(Integer, nullable=True)
    personal_allergy_history = Column(Boolean, nullable=True)
    work_env = Column(String, nullable=True)
    family_has_history = Column(Boolean, nullable=True)
    family_asthma = Column(Boolean, nullable=True)
    family_eczema = Column(Boolean, nullable=True)
    family_allergies = Column(String, nullable=True) # Stored as comma-separated
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    allergens = relationship("UserAllergen", back_populates="user", cascade="all, delete-orphan")


class UserAllergen(Base):
    __tablename__ = "user_allergens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    allergen_category = Column(String, nullable=False)

    # Relationships
    user = relationship("User", back_populates="allergens")
