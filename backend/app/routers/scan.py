"""
routers/scan.py — POST /api/v1/scan endpoint.

This is the main endpoint of the application.  A Flutter client will call it
with a dish photo + the user's selected allergen list and receive back a full
allergen analysis report.

Request:  multipart/form-data
  - image         : UploadFile  (JPEG / PNG / WEBP, max 10 MB)
  - user_allergens: JSON string (list of allergen category strings)
                    Example: '["shellfish","tree_nuts"]'

Response: ScanResponse (JSON)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.sqlite_db import get_db
from app.schemas.scan import AllergenMatch, ClinicalAlert, MLRiskReport, SafetyNetWarning, ScanResponse
from app.services.rag_service import RAGService, get_rag_service
from app.services.allergen_service import run_hybrid_allergen_detection
from app.services.recipe_service import get_ingredients
from app.services import user_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scan", tags=["Scan"])

# Accepted MIME types for uploaded images
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


# --------------------------------------------------------------------------- #
#  POST /api/v1/scan                                                           #
# --------------------------------------------------------------------------- #

@router.post(
    "/",
    response_model=ScanResponse,
    summary="Scan a Sri Lankan dish image for allergens",
    description=(
        "Upload a photo of a Sri Lankan dish. "
        "The system identifies the dish via Gemini Vision + RAG, "
        "then runs a hybrid allergen detection pipeline against your selected allergens."
    ),
    status_code=status.HTTP_200_OK,
)
async def scan_dish(
    image: UploadFile = File(..., description="Dish photo (JPEG / PNG / WEBP, max 10 MB)"),
    user_allergens: str = Form(
        default="[]",
        description=(
            'JSON array of allergen category strings. '
            'Example: \'["shellfish", "tree_nuts", "eggs"]\''
        ),
    ),
    user_id: Optional[int] = Form(
        default=None,
        description="Optional ID of the registered user to fetch saved allergens from."
    ),
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
) -> ScanResponse:
    """
    Full allergen scan pipeline:
    1. Validate image
    2. Gemini Vision → dish name guess
    3. ChromaDB RAG retrieval → best dish + confidence
    4. Load recipe ingredients from JSON
    5. Hybrid allergen detection (rule-based + LLM)
    6. Build and return ScanResponse
    """

    # ── 1. Validate image ─────────────────────────────────────────────────── #

    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{image.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    image_bytes = await image.read()

    if len(image_bytes) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum allowed size of {settings.MAX_IMAGE_BYTES // 1_048_576} MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    # ── 2. Parse user_allergens ───────────────────────────────────────────── #

    parsed_allergens: list[str] = []
    user = None  # will be populated if user_id is provided

    # If user_id is provided, load their registered allergens first
    if user_id is not None:
        user = user_service.get_user(db, user_id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found.",
            )
        parsed_allergens = [a.allergen_category.lower() for a in user.allergens]
        logger.info("Loaded allergens for user_id=%d: %s", user_id, parsed_allergens)

    # Parse and merge form-specified allergens if provided
    form_allergens: list[str] = []
    user_allergens_clean = user_allergens.strip() if user_allergens else ""

    if user_allergens_clean and user_allergens_clean.lower() != "null":
        # Check if it looks like a JSON array
        if user_allergens_clean.startswith("[") and user_allergens_clean.endswith("]"):
            try:
                val = json.loads(user_allergens_clean)
                if isinstance(val, list):
                    form_allergens = [str(item).strip() for item in val]
                else:
                    form_allergens = [str(val).strip()]
            except json.JSONDecodeError:
                # Fallback: Strip brackets and split by comma if JSON parsing fails
                inner = user_allergens_clean[1:-1].strip()
                if inner:
                    form_allergens = [
                        item.strip(" '\"")
                        for item in inner.split(",")
                        if item.strip()
                    ]
        else:
            # Try to load as JSON in case it's a quoted string
            try:
                val = json.loads(user_allergens_clean)
                if isinstance(val, list):
                    form_allergens = [str(item).strip() for item in val]
                else:
                    form_allergens = [str(val).strip()]
            except json.JSONDecodeError:
                # Fallback: Treat as comma-separated values
                form_allergens = [
                    item.strip(" '\"")
                    for item in user_allergens_clean.split(",")
                    if item.strip()
                ]

    # Normalise and merge
    for allergen in form_allergens:
        allergen_clean = allergen.strip().lower()
        if allergen_clean and allergen_clean not in parsed_allergens:
            parsed_allergens.append(allergen_clean)

    logger.info(
        "Scan request | file=%s size=%d bytes | user_allergens=%s",
        image.filename,
        len(image_bytes),
        parsed_allergens,
    )

    # ── Build user profile vector for ML engine ───────────────────────── #
    # Populated from the registered user record (if available); otherwise
    # the ML engine falls back to population-average zero-vector predictions.
    user_profile: dict = {}
    if user_id is not None and user is not None:
        gender_map = {"male": 1, "female": 0, "other": 2}
        user_profile = {
            "age": user.age if user.age is not None else 0,
            "gender": gender_map.get((user.gender or "").lower(), 0),
            "province": user.province or "",
            "blood_type": user.blood_type or "",
            "dietary_pattern": user.dietary_pattern or "",
            "lactose_intolerance": 1.0 if user.lactose_intolerance else 0.0,
            "outside_food_frequency": float(user.outside_food_frequency or 0),
            "personal_allergy_history": 1.0 if user.personal_allergy_history else 0.0,
        }
        
        # Add medical conditions one-hot
        if user.medical_conditions:
            for cond in user.medical_conditions.split(","):
                cond_clean = cond.strip().lower().replace(" ", "_")
                user_profile[f"med_cond_{cond_clean}"] = 1.0

    # ── 3-6. RAG Service: Identify dish and retrieve ingredients ────────── #

    try:
        rag_result = await run_in_threadpool(rag_service.analyze_image, image_bytes)
    except Exception as e:
        error_str = str(e)
        if "503" in error_str or "UNAVAILABLE" in error_str:
            raise HTTPException(
                status_code=503,
                detail="The AI model is currently experiencing high demand. Please try again in a few moments."
            )
        raise HTTPException(
            status_code=500,
            detail=f"Dish analysis failed: {error_str}"
        )
    
    best_dish = rag_result.get("dish_name", "unknown")
    llm_ingredients = rag_result.get("ingredients", [])
    vision_guess = rag_result.get("vision_guess", best_dish)
    confidence = "high"
    rag_context_used = True

    # Ground-truth: Load actual ingredients from JSON to prevent LLM hallucination
    ingredients = []
    if best_dish != "unknown":
        ingredients = get_ingredients(best_dish)

    if not ingredients:
        # Dish not in our database or missing from JSON, fall back to LLM's best guess
        logger.warning("No ground-truth recipe found for '%s'. Falling back to LLM ingredients.", best_dish)
        ingredients = llm_ingredients
        if not ingredients:
            ingredients = [best_dish]  # placeholder so LLM has something to work with
        confidence = "low"
        rag_context_used = False

    # ── 7. Hybrid allergen detection ─────────────────────────────────────── #

    detected_allergens_raw, llm_explanation, ml_risk_report_raw = await run_in_threadpool(
        run_hybrid_allergen_detection,
        dish_name=best_dish,
        ingredients=ingredients,
        user_allergens=parsed_allergens,
        user_profile=user_profile,
    )

    # Convert raw dicts → AllergenMatch schema objects
    detected_allergens = [AllergenMatch(**a) for a in detected_allergens_raw]

    # ── 8. Safety decision ───────────────────────────────────────────────── #

    user_cats = {a.lower() for a in parsed_allergens}

    if user_cats:
        # is_safe = True only if NONE of the detected allergens are in the user's list
        triggered_cats = {a.allergen_category.lower() for a in detected_allergens}
        is_safe = len(triggered_cats & user_cats) == 0
    else:
        # No user allergens declared → cannot assess personal safety just from rule-based
        is_safe = True

    # Build MLRiskReport schema object from raw dict
    ml_risk_report = MLRiskReport(
        clinical_alerts=[
            ClinicalAlert(**alert) for alert in ml_risk_report_raw["clinical_alerts"]
        ],
        safety_net_warnings=[
            SafetyNetWarning(**warn) for warn in ml_risk_report_raw["safety_net_warnings"]
        ],
        models_evaluated=ml_risk_report_raw["models_evaluated"],
        ingredients_checked=ml_risk_report_raw["ingredients_checked"],
    )

    # Also fail safety if there are any ML clinical alerts
    if len(ml_risk_report.clinical_alerts) > 0:
        is_safe = False

    if is_safe:
        safety_message = (
            f"✅ {best_dish.replace('_', ' ').title()} appears safe based on your profile."
        )
    else:
        unsafe_allergens = ", ".join(
            a.allergen_category for a in detected_allergens
            if a.allergen_category.lower() in user_cats
        )
        msg_parts = []
        if unsafe_allergens:
            msg_parts.append(f"contains known allergens: {unsafe_allergens}")
        
        high_risk = [a.ingredient for a in ml_risk_report.clinical_alerts if a.risk_level >= 2]
        if high_risk:
            msg_parts.append(f"ML risk alerts for: {', '.join(high_risk)}")
            
        safety_message = f"⚠️ WARNING: {best_dish.replace('_', ' ').title()} {' and '.join(msg_parts)}. Avoid this dish."

    # ── 9. Build response ────────────────────────────────────────────────── #

    return ScanResponse(
        identified_dish=best_dish,
        confidence=confidence,
        ingredients=ingredients,
        rag_context_used=rag_context_used,
        detected_allergens=detected_allergens,
        is_safe=is_safe,
        safety_message=safety_message,
        llm_explanation=llm_explanation,
        ml_risk_report=ml_risk_report,
        llm_raw_dish_guess=vision_guess if vision_guess != best_dish else None,
    )


# --------------------------------------------------------------------------- #
#  GET /api/v1/scan/allergens — list all known allergen categories             #
# --------------------------------------------------------------------------- #

@router.get(
    "/allergens",
    summary="List all supported allergen categories",
    description="Returns the full list of allergen categories the system can detect.",
)
async def list_allergen_categories() -> dict:
    from app.utils.allergen_kb import ALLERGEN_KB
    return {
        "allergen_categories": sorted(ALLERGEN_KB.keys()),
        "total": len(ALLERGEN_KB),
    }


# --------------------------------------------------------------------------- #
#  GET /api/v1/scan/dishes — list all known dishes                             #
# --------------------------------------------------------------------------- #

@router.get(
    "/dishes",
    summary="List all supported Sri Lankan dishes",
    description="Returns all dish names currently in the recipe database.",
)
async def list_known_dishes() -> dict:
    from app.services.recipe_service import get_all_dish_names
    dishes = get_all_dish_names()
    return {"dishes": dishes, "total": len(dishes)}
