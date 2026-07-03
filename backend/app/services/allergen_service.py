"""
services/allergen_service.py — Hybrid allergen detection pipeline.

Three-layer detection strategy (applied in order):
──────────────────────────────────────────────────
Layer 1 — Rule-based KB scan   (fast, deterministic, no API call)
Layer 2 — LLM cross-check      (catches allergens the KB may miss, e.g. hidden
                                 allergens in compound ingredients like curry powder)
Layer 3 — User-profile merge   (ensures every user-selected allergen category is
                                 present in the result if any ingredient triggers it)

The final result deduplicates across all three layers and preserves the source
of each detection for transparency ("where did this come from?").
Uses the new google-genai SDK (v1+).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from google import genai

from app.core.config import settings
from app.utils.allergen_kb import scan_ingredients
from app.services import ml_risk_service

logger = logging.getLogger(__name__)

# Cached google.genai client
_genai_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Return a cached google.genai Client."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _genai_client

# --------------------------------------------------------------------------- #
#  Layer 2 — LLM allergen cross-check                                         #
# --------------------------------------------------------------------------- #

LLM_ALLERGEN_PROMPT = """\
You are a food safety expert specialising in South Asian cuisine.

Dish: {dish_name}
Ingredients:
{ingredients_list}

User's declared allergens: {user_allergens}

Tasks:
1. Check ALL ingredients (including any hidden allergens inside compound ingredients
   like "curry powder", "curry leaves mix", etc.).
2. Identify ingredients that belong to any of these allergen categories:
3. Pay special attention to the user's declared allergens.

ALLERGEN CATEGORY DEFINITIONS & STRICT RULES:
- milk_dairy: Strictly refers to milk from mammals (cows, buffalo, goats). EXCLUDE coconut milk, soy milk, almond milk, or other plant-based milks.
- tree_nuts: Strictly refers to almonds, walnuts, cashews, hazelnuts, pistachios, etc. EXCLUDE coconut and nutmeg. Coconut must NEVER be classified as a tree nut.
- peanuts: Peanuts are legumes, not tree nuts.
- crustaceans: Refers to prawns, shrimp, crab, lobster. EXCLUDE molluscs.
- molluscs: Refers to squid, octopus, cuttlefish, clams, mussels.

Respond in VALID JSON only (no markdown, no explanation outside the JSON):
{{
  "detected_allergens": [
    {{
      "allergen_category": "<category>",
      "triggered_by": ["<ingredient1>", "<ingredient2>"],
      "source": "llm"
    }}
  ],
  "hidden_allergen_notes": "<any notes about hidden allergens in compound ingredients>"
}}

If no allergens are found, return:
{{"detected_allergens": [], "hidden_allergen_notes": ""}}
"""


def llm_check_allergens(
    dish_name: str,
    ingredients: list[str],
    user_allergens: list[str],
) -> list[dict]:
    """
    Ask Gemini to cross-check the ingredient list for allergens.

    Returns:
        List of allergen dicts with source="llm".
        Empty list on failure (the pipeline continues with rule-based results).
    """
    ingredients_list = "\n".join(f"- {ing}" for ing in ingredients)
    user_allergens_str = ", ".join(user_allergens) if user_allergens else "none specified"

    prompt = LLM_ALLERGEN_PROMPT.format(
        dish_name=dish_name,
        ingredients_list=ingredients_list,
        user_allergens=user_allergens_str,
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
        )
        raw_text = response.text.strip()

        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)
        allergens = data.get("detected_allergens", [])

        logger.info(
            "LLM detected %d allergen(s) in %s. Hidden notes: %s",
            len(allergens),
            dish_name,
            data.get("hidden_allergen_notes", ""),
        )
        return allergens

    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s | Raw: %s", exc, raw_text[:200])
        return []
    except Exception as exc:
        logger.error("LLM allergen check failed: %s", exc, exc_info=True)
        return []


# --------------------------------------------------------------------------- #
#  Layer 3 — Merge & deduplicate results                                      #
# --------------------------------------------------------------------------- #

def merge_allergen_results(
    rule_based: list[dict],
    llm_results: list[dict],
    user_allergens: list[str],
) -> list[dict]:
    """
    Merge rule-based + LLM results into a deduplicated, clean list.

    - If the same category is found by both layers, merge the triggered_by
      lists and label the source "rule_based+llm".
    - User-declared allergens that appear in either result are flagged with
      source "user_profile" appended.

    Returns:
        Deduplicated list sorted by allergen category name.
    """
    # category → {"triggered_by": set, "sources": set}
    merged: dict[str, dict] = {}

    for hit in rule_based + llm_results:
        cat = hit["allergen_category"]
        if cat not in merged:
            merged[cat] = {"triggered_by": set(), "sources": set()}
        merged[cat]["triggered_by"].update(hit.get("triggered_by", []))
        merged[cat]["sources"].add(hit.get("source", "unknown"))

    # Annotate with user_profile source where applicable
    user_cats = {a.lower() for a in user_allergens}
    for cat, data in merged.items():
        if cat.lower() in user_cats:
            data["sources"].add("user_profile")

    result = []
    for cat, data in sorted(merged.items()):
        sources = sorted(data["sources"])
        source_str = "+".join(sources)
        result.append({
            "allergen_category": cat,
            "triggered_by": sorted(data["triggered_by"]),
            "source": source_str,
        })

    return result


# --------------------------------------------------------------------------- #
#  LLM explanation generator                                                   #
# --------------------------------------------------------------------------- #

EXPLANATION_PROMPT = """\
You are a friendly food safety assistant helping someone with food allergies.

Dish: {dish_name}
Ingredients: {ingredients}
Allergens detected: {allergens_summary}
User's declared allergens: {user_allergens}

Write a SHORT (3-4 sentences), friendly, and clear explanation:
1. What the dish is.
2. Which ingredients may cause allergic reactions for this user.
3. A simple safety recommendation.

Speak directly to the user ("you"). Be warm, not clinical.
"""


def generate_explanation(
    dish_name: str,
    ingredients: list[str],
    detected_allergens: list[dict],
    user_allergens: list[str],
) -> str:
    """
    Generate a friendly natural-language allergen explanation using Gemini.

    Returns:
        Plain text explanation string.
        Returns a fallback message on failure.
    """
    if not detected_allergens:
        allergens_summary = "No allergens detected."
    else:
        allergens_summary = "; ".join(
            f"{a['allergen_category']} (from {', '.join(a['triggered_by'])})"
            for a in detected_allergens
        )

    prompt = EXPLANATION_PROMPT.format(
        dish_name=dish_name.replace("_", " ").title(),
        ingredients=", ".join(ingredients),
        allergens_summary=allergens_summary,
        user_allergens=", ".join(user_allergens) if user_allergens else "none",
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
        )
        return response.text.strip()

    except Exception as exc:
        logger.error("Explanation generation failed: %s", exc, exc_info=True)
        return (
            f"This appears to be {dish_name.replace('_', ' ').title()}. "
            "Please review the ingredients carefully before consuming."
        )


# --------------------------------------------------------------------------- #
#  Public API — Full hybrid detection                                          #
# --------------------------------------------------------------------------- #

def run_hybrid_allergen_detection(
    dish_name: str,
    ingredients: list[str],
    user_allergens: list[str],
    user_profile: dict | None = None,
) -> tuple[list[dict], str, dict]:
    """
    Entry point for the hybrid allergen detection pipeline.

    Args:
        dish_name:       Identified dish name (e.g. "fish_curry").
        ingredients:     Ingredient list from the matched recipe.
        user_allergens:  Allergen categories from the user's profile.
        user_profile:    Optional dict of user demographic / medical features
                         used by the ML risk engine for clinical prediction.

    Returns:
        (detected_allergens, explanation, ml_risk_report)
        - detected_allergens: Merged, deduplicated list of AllergenMatch dicts.
        - explanation:        LLM-generated natural language explanation.
        - ml_risk_report:     ML + Apriori risk prediction report dict.
    """
    logger.info(
        "Running hybrid allergen detection for '%s' | user allergens: %s",
        dish_name,
        user_allergens,
    )

    # Layer 1 — Rule-based
    rule_based = scan_ingredients(ingredients, user_allergens)
    logger.info("Rule-based found %d allergen category/categories.", len(rule_based))

    # Layer 2 — LLM cross-check
    llm_results = llm_check_allergens(dish_name, ingredients, user_allergens)
    logger.info("LLM found %d allergen category/categories.", len(llm_results))

    # Layer 3 — Merge
    detected = merge_allergen_results(rule_based, llm_results, user_allergens)

    # Generate explanation (separate LLM call for clean output)
    explanation = generate_explanation(dish_name, ingredients, detected, user_allergens)

    # Layer 4 — ML + Apriori risk prediction
    ml_risk_report = ml_risk_service.predict_dish_safety(
        detected_ingredients=ingredients,
        user_profile=user_profile or {},
    )
    logger.info(
        "ML risk engine complete | clinical_alerts=%d | safety_net_warnings=%d",
        len(ml_risk_report["clinical_alerts"]),
        len(ml_risk_report["safety_net_warnings"]),
    )

    return detected, explanation, ml_risk_report
