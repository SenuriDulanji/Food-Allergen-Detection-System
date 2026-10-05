"""
schemas/user.py — Pydantic models for user request validation and response presentation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserBase(BaseModel):
    email: str = Field(..., description="User's unique email address")
    full_name: Optional[str] = Field(default=None, description="User's full name")
    age: Optional[int] = Field(default=None, description="User's age")
    gender: Optional[str] = Field(default=None, description="User's gender")
    province: Optional[str] = Field(default=None, description="User's province (demographic)")
    blood_type: Optional[str] = Field(default=None, description="User's blood type (e.g., O+, A-)")
    dietary_pattern: Optional[str] = Field(default=None, description="Vegetarian, Vegan, Non-Vegetarian, etc.")
    medical_conditions: Optional[list[str]] = Field(default=None, description="Asthma, Eczema, Food Allergy, etc.")
    lactose_intolerance: Optional[bool] = Field(default=None, description="Whether user is lactose intolerant")
    outside_food_frequency: Optional[int] = Field(default=None, description="0-4 scale of how often user eats out")
    personal_allergy_history: Optional[bool] = Field(default=None, description="Personal allergy history")
    work_env: Optional[str] = Field(default=None, description="User's work environment")
    family_has_history: Optional[bool] = Field(default=None, description="Does the user's family have a history of allergies?")
    family_asthma: Optional[bool] = Field(default=None, description="Family history of asthma")
    family_eczema: Optional[bool] = Field(default=None, description="Family history of eczema")
    family_allergies: Optional[list[str]] = Field(default=None, description="List of family food allergies")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v_clean = v.strip()
        if not EMAIL_REGEX.match(v_clean):
            raise ValueError("Invalid email format")
        return v_clean


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User's password (min 6 chars)")
    allergens: list[str] = Field(
        default=[],
        description="List of selected allergen categories (e.g. ['gluten', 'shellfish'])"
    )


class UserUpdate(BaseModel):
    email: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None, min_length=6)
    full_name: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    province: Optional[str] = Field(default=None)
    blood_type: Optional[str] = Field(default=None)
    dietary_pattern: Optional[str] = Field(default=None)
    medical_conditions: Optional[list[str]] = Field(default=None)
    lactose_intolerance: Optional[bool] = Field(default=None)
    outside_food_frequency: Optional[int] = Field(default=None)
    personal_allergy_history: Optional[bool] = Field(default=None)
    work_env: Optional[str] = Field(default=None)
    family_has_history: Optional[bool] = Field(default=None)
    family_asthma: Optional[bool] = Field(default=None)
    family_eczema: Optional[bool] = Field(default=None)
    family_allergies: Optional[list[str]] = Field(default=None)
    allergens: Optional[list[str]] = Field(
        default=None,
        description="List of selected allergen categories (e.g. ['gluten', 'shellfish'])"
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_clean = v.strip()
        if not EMAIL_REGEX.match(v_clean):
            raise ValueError("Invalid email format")
        return v_clean


class UserResponse(UserBase):
    id: int = Field(..., description="Unique database ID")
    is_active: bool = Field(..., description="Whether user account is active")
    allergens: list[str] = Field(default=[], description="List of user's selected allergens")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
