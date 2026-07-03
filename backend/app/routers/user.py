"""
routers/user.py — POST /api/v1/users endpoints for user management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.sqlite_db import get_db
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services import user_service

router = APIRouter(prefix="/users", tags=["Users"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a user profile, store demographic info, and select default allergens.",
)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    """Create new user account."""
    db_user = user_service.get_user_by_email(db, email=user_in.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )
    user = user_service.create_user(db, user_in=user_in)
    return UserResponse(**user_service.format_user_response(user))


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user profile",
    description="Retrieve a user's details, demographics, and selected allergens by user ID.",
)
def get_user_profile(user_id: int, db: Session = Depends(get_db)) -> UserResponse:
    """Get user profile details."""
    user = user_service.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return UserResponse(**user_service.format_user_response(user))


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user profile",
    description="Update demographic details, profile fields, or selected allergen preferences.",
)
def update_user_profile(
    user_id: int,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
) -> UserResponse:
    """Update user profile details and allergens."""
    user = user_service.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    
    # If email is updating, check uniqueness
    if user_in.email and user_in.email != user.email:
        existing = user_service.get_user_by_email(db, email=user_in.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists.",
            )

    updated_user = user_service.update_user(db, db_user=user, user_in=user_in)
    return UserResponse(**user_service.format_user_response(updated_user))


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user account",
)
def delete_user_profile(user_id: int, db: Session = Depends(get_db)):
    """Delete a user account."""
    success = user_service.delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return None


@router.get(
    "/",
    response_model=list[UserResponse],
    summary="List all users",
    description="List all registered user profiles and their details (useful for admin/debugging).",
)
def list_users(db: Session = Depends(get_db)) -> list[UserResponse]:
    """Retrieve all users in the system."""
    from app.models.user import User
    users = db.query(User).all()
    return [UserResponse(**user_service.format_user_response(u)) for u in users]
