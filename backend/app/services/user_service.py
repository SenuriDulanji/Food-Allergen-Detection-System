"""
services/user_service.py — CRUD operations for User management.
"""

from typing import Optional
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.models.user import User, UserAllergen
from app.schemas.user import UserCreate, UserUpdate


def get_user(db: Session, user_id: int) -> Optional[User]:
    """Retrieve a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieve a user by email."""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user_in: UserCreate) -> User:
    """Create a new user with hashed password and initial allergens list."""
    hashed_pwd = get_password_hash(user_in.password)
    
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_pwd,
        full_name=user_in.full_name,
        age=user_in.age,
        gender=user_in.gender,
        province=user_in.province,
        blood_type=user_in.blood_type,
        dietary_pattern=user_in.dietary_pattern,
        medical_conditions=",".join(user_in.medical_conditions) if user_in.medical_conditions else None,
        lactose_intolerance=user_in.lactose_intolerance,
        outside_food_frequency=user_in.outside_food_frequency,
        personal_allergy_history=user_in.personal_allergy_history,
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Add allergens if specified
    if user_in.allergens:
        for allergen in user_in.allergens:
            allergen_clean = allergen.strip().lower()
            if allergen_clean:
                db_allergen = UserAllergen(
                    user_id=db_user.id,
                    allergen_category=allergen_clean
                )
                db.add(db_allergen)
        db.commit()
        db.refresh(db_user)
        
    return db_user


def update_user(db: Session, db_user: User, user_in: UserUpdate) -> User:
    """Update user profile details and allergens list."""
    update_data = user_in.model_dump(exclude_unset=True)
    
    # Handle password hashing separately
    if "password" in update_data:
        db_user.hashed_password = get_password_hash(update_data.pop("password"))
        
    # Handle allergens list sync
    if "allergens" in update_data:
        new_allergens = update_data.pop("allergens")
        # Clear existing allergens
        db.query(UserAllergen).filter(UserAllergen.user_id == db_user.id).delete()
        # Add new allergens
        if new_allergens:
            for allergen in new_allergens:
                allergen_clean = allergen.strip().lower()
                if allergen_clean:
                    db.add(UserAllergen(user_id=db_user.id, allergen_category=allergen_clean))
                    
    # Handle medical_conditions separately
    if "medical_conditions" in update_data:
        val = update_data.pop("medical_conditions")
        db_user.medical_conditions = ",".join(val) if val else None

    # Update other demographic/profile attributes
    for field, value in update_data.items():
        setattr(db_user, field, value)
        
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def delete_user(db: Session, user_id: int) -> bool:
    """Delete a user from the database."""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False


def format_user_response(db_user: User) -> dict:
    """Helper to convert database User model into schema-compatible dict."""
    return {
        "id": db_user.id,
        "email": db_user.email,
        "full_name": db_user.full_name,
        "age": db_user.age,
        "gender": db_user.gender,
        "province": db_user.province,
        "blood_type": db_user.blood_type,
        "dietary_pattern": db_user.dietary_pattern,
        "medical_conditions": db_user.medical_conditions.split(",") if db_user.medical_conditions else [],
        "lactose_intolerance": db_user.lactose_intolerance,
        "outside_food_frequency": db_user.outside_food_frequency,
        "personal_allergy_history": db_user.personal_allergy_history,
        "is_active": db_user.is_active,
        "allergens": [a.allergen_category for a in db_user.allergens],
        "created_at": db_user.created_at,
        "updated_at": db_user.updated_at,
    }
