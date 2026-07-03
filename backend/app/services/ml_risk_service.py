"""
services/ml_risk_service.py — Allergen Risk Prediction via ML Models + Apriori Rules.

Unified Intelligence Engine (Layer 4 of the detection pipeline):
  • ML Engine  — Loads per-ingredient XGBoost SMOTE pipelines (*.joblib) and predicts
                 clinical risk level (0 = safe, 1-3 = risk) for each detected ingredient
                 based on the user's demographic / medical profile vector.
  • Safety Net — Applies validated Apriori cross-reactivity rules to raise additional
                 warnings when co-reactive ingredients are found in the same dish.

Public API:
  predict_dish_safety(user_profile_vector, detected_ingredients) -> MLRiskReport dict
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Artifact location                                                           #
# --------------------------------------------------------------------------- #
# Resolves to:  <project_root>/ml_model/artifacts/models/
_THIS_FILE   = Path(__file__).resolve()                      # .../backend/app/services/
_BACKEND_DIR = _THIS_FILE.parents[2]                         # .../backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                          # <project_root>/
ARTIFACTS_DIR = _PROJECT_ROOT / "ml_model" / "artifacts" / "models"


# --------------------------------------------------------------------------- #
#  Apriori cross-reactivity rules (validated from mlxtend output)              #
# --------------------------------------------------------------------------- #
# Keys are lowercase ingredient names as they appear after detection.
# Values are the co-reactive ingredients the model associates them with.
SAFETY_RULES: dict[str, list[str]] = {
    "mutton":             ["beef", "pork"],
    "beef":               ["pork"],
    "cuttlefish / squid": ["crab"],
    "crab":               ["cuttlefish / squid"],
    "prawns":             ["cuttlefish / squid", "crab", "pineapple"],
}

# --------------------------------------------------------------------------- #
#  Ingredient synonym normalisation                                            #
# --------------------------------------------------------------------------- #
# Maps common aliases / regional names → the canonical key used in model files
# and SAFETY_RULES. Add new synonyms here as the dataset grows.
INGREDIENT_SYNONYMS: dict[str, str] = {
    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY BLOCK — compound phrases that contain sub-words belonging
    # to a DIFFERENT allergen. These MUST be checked first (dict is ordered).
    # ══════════════════════════════════════════════════════════════════════
    "peanut butter":        "peanuts",   # contains 'butter' → would hit milk
    "peanut oil":           "peanuts",   # contains 'oil'
    "bean curd":            "soy",       # contains 'curd' → would hit milk
    "coconut cream":        "coconut",   # contains 'cream' → would hit milk
    "coconut milk":         "coconut",   # contains 'milk' → would hit milk
    "thick coconut milk":   "coconut",
    "thin coconut milk":    "coconut",
    "coconut oil":          "coconut",
    "coconut butter":       "coconut",
    "soy milk":             "soy",       # contains 'milk' → would hit milk
    "soya milk":            "soy",
    "oat milk":             "wheat",     # contains 'milk' → would hit milk
    "almond milk":          "tree_nuts", # contains 'milk' → would hit milk
    "tuna steak":           "tuna",      # contains 'steak' → would hit beef
    "tuna fish":            "tuna",
    "canned tuna":          "tuna",
    "dried sprats":         "dry_fish",  # contains 'sprats'
    "dried cuttlefish":     "cuttlefish / squid",
    "dried fish":           "dry_fish",
    "dry fish":             "dry_fish",
    "maldive fish":         "dry_fish",
    "ground beef":          "beef",      # keep here to precede 'ground'
    "minced beef":          "beef",

    # NOTE: Longer / more specific phrases MUST come before shorter ones
    # because the synonym loop returns the FIRST match (word-boundary regex).
    # ── Prawns / Shrimp ───────────────────────────────────────────────────
    "shrimp":               "prawns",
    "shrimps":              "prawns",
    "prawn":                "prawns",
    "king prawn":           "prawns",
    "tiger prawn":          "prawns",
    "jumbo prawn":          "prawns",
    "lobster":              "prawns",       # crustacean cross-reactivity

    # ── Cuttlefish / Squid ───────────────────────────────────────────────
    "squid":                "cuttlefish / squid",
    "cuttlefish":           "cuttlefish / squid",
    "calamari":             "cuttlefish / squid",
    "inkfish":              "cuttlefish / squid",
    "octopus":              "cuttlefish / squid",

    # ── Milk / Dairy ─────────────────────────────────────────────────────
    # Compound phrases that would otherwise partially match a shorter key
    # (e.g. 'coconut cream' must not resolve to 'cream→milk'):
    "coconut cream":        "coconut",
    "coconut milk":         "coconut",
    "coconut oil":          "coconut",
    "butter":               "milk",
    "ghee":                 "milk",
    "cream":                "milk",
    "fresh cream":          "milk",
    "double cream":         "milk",
    "whipping cream":       "milk",
    "paneer":               "milk",
    "cheese":               "milk",
    "cheddar":              "milk",
    "mozzarella":           "milk",
    "yogurt":               "milk",
    "yoghurt":              "milk",
    "curd":                 "milk",
    "dairy":                "milk",
    "lactose":              "milk",
    "whey":                 "milk",
    "casein":               "milk",
    "buttermilk":           "milk",

    # ── Eggs (singular forms) ────────────────────────────────────────────
    # The model key is "eggs"; singular "egg" won't match \beggs\b
    "egg":                  "eggs",
    "egg yolk":             "eggs",
    "egg white":            "eggs",
    "egg yolks":            "eggs",
    "egg whites":           "eggs",
    "beaten egg":           "eggs",
    "whole egg":            "eggs",
    "duck egg":             "eggs",
    "quail egg":            "eggs",

    # ── Wheat / Gluten ───────────────────────────────────────────────────
    "flour":                "wheat",
    "plain flour":          "wheat",
    "self-raising flour":   "wheat",
    "maida":                "wheat",       # South Asian refined wheat flour
    "atta":                 "wheat",       # South Asian whole wheat flour
    "bread":                "wheat",
    "bread crumbs":         "wheat",
    "breadcrumbs":          "wheat",
    "naan":                 "wheat",
    "roti":                 "wheat",
    "chapati":              "wheat",
    "pita":                 "wheat",
    "pasta":                "wheat",
    "noodles":              "wheat",
    "semolina":             "wheat",
    "gluten":               "wheat",
    "seitan":               "wheat",

    # ── Peanuts ──────────────────────────────────────────────────────────
    "peanut butter":        "peanuts",  # must precede 'butter'→milk
    "peanut":               "peanuts",     # singular — \bpeanuts\b won't catch it
    "groundnut":            "peanuts",     # British / South Asian common name
    "groundnuts":           "peanuts",
    "groundnut oil":        "peanuts",
    "monkey nut":           "peanuts",
    "monkey nuts":          "peanuts",
    "arachis oil":          "peanuts",     # peanut oil INCI name

    # ── Tree Nuts ────────────────────────────────────────────────────────
    "cashew":               "tree_nuts",
    "cashews":              "tree_nuts",
    "cashew nut":           "tree_nuts",
    "cashew nuts":          "tree_nuts",
    "almond":               "tree_nuts",
    "almonds":              "tree_nuts",
    "walnut":               "tree_nuts",
    "walnuts":              "tree_nuts",
    "pistachio":            "tree_nuts",
    "pistachios":           "tree_nuts",
    "pecan":                "tree_nuts",
    "pecans":               "tree_nuts",
    "hazelnut":             "tree_nuts",
    "hazelnuts":            "tree_nuts",
    "macadamia":            "tree_nuts",
    "pine nut":             "tree_nuts",
    "pine nuts":            "tree_nuts",
    "brazil nut":           "tree_nuts",
    "brazil nuts":          "tree_nuts",

    # ── Soy / Soya ───────────────────────────────────────────────────────
    "bean curd":            "soy",      # must precede 'curd'→milk
    "soy milk":             "soy",      # must precede 'milk' substring match
    "soya milk":            "soy",
    "soya":                 "soy",
    "soybean":              "soy",
    "soybeans":             "soy",
    "soy bean":             "soy",
    "tofu":                 "soy",
    "tempeh":               "soy",
    "edamame":              "soy",
    "miso":                 "soy",
    "natto":                "soy",

    # ── Sesame ───────────────────────────────────────────────────────────
    "gingelly":             "sesame",      # South Asian/Sri Lankan name
    "gingelly oil":         "sesame",
    "til":                  "sesame",      # South Asian name
    "teel":                 "sesame",
    "til seeds":            "sesame",
    "tahini":               "sesame",

    # ── Dry Fish (Sri Lankan specific) ───────────────────────────────────
    "dried sprats":         "dry_fish",  # must precede 'sprats' alone
    "dried fish":           "dry_fish",
    "dry fish":             "dry_fish",
    "maldive fish":         "dry_fish",    # key Sri Lankan ingredient
    "umbalakada":           "dry_fish",    # Sinhala: dried tuna flakes
    "jaadi":                "dry_fish",    # salted dry fish
    "sprat":                "dry_fish",    # small dried fish
    "sprats":               "dry_fish",
    "dried sprats":         "dry_fish",
    "anchovy":              "dry_fish",
    "anchovies":            "dry_fish",

    # ── Tomato ───────────────────────────────────────────────────────────
    # Model key is "tomato"; "tomatoes" won't match \btomato\b exactly
    "tomatoes":             "tomato",

    # ── Tuna ─────────────────────────────────────────────────────────────
    "tuna steak":           "tuna",     # must precede 'steak'→beef
    "tuna fish":            "tuna",
    "canned tuna":          "tuna",

    # ── Beef ─────────────────────────────────────────────────────────────
    "ground beef":          "beef",
    "minced beef":          "beef",
    "steak":                "beef",
    "veal":                 "beef",        # young cattle

    # ── Mutton / Lamb / Goat ─────────────────────────────────────────────
    "lamb":                 "mutton",
    "goat":                 "mutton",
    "goat meat":            "mutton",
    "venison":              "mutton",      # red meat cross-reactivity
}


# --------------------------------------------------------------------------- #
#  Feature columns — must match exactly what was used during training          #
# --------------------------------------------------------------------------- #
# These are the demographic / medical features that the XGBoost models expect.
# Order is critical — it must be identical to `features_to_keep` computed
# during training (Spearman ≥ 0.15 filter).  The common safe subset that
# always survives the filter is listed here; any extra one-hot columns will
# be filled with 0 if absent.
BASE_FEATURE_COLS: list[str] = [
    "age",
    "gender",                         # encoded: 0=Female, 1=Male, 2=Other
    "personal_allergy_history",       # 0/1
    "outside_food_frequency",         # ordinal 0-4
    "lactose_intolerance",            # 0/1
    "med_cond_asthma",
    "med_cond_eczema",
    "med_cond_food_allergy",
    "med_cond_none_of_these",
    "med_cond_not_sure",
    "family_has_history",
    "family_asthma",
    "family_eczema",
    "family_allergy_beef",
    "family_allergy_pork",
    "family_allergy_sausage",
    "family_allergy_red_meat",
    "family_allergy_seafood",
    "family_allergy_dairy",
    "family_allergy_nuts",
    "family_allergy_egg",
    "family_allergy_tomato",
    "family_allergy_pineapple",
    "family_allergy_avocado",
    "family_allergy_breadfruit",
    "family_allergy_flour",
]

# Province one-hot prefixes supported by the training data
_PROVINCE_COLS = [
    "province_central", "province_eastern", "province_north_central",
    "province_north_western", "province_northern", "province_sabaragamuwa",
    "province_southern", "province_uva", "province_western",
]
# Blood-type one-hot prefixes
_BLOOD_TYPE_COLS = [
    "blood_type_a+", "blood_type_a-", "blood_type_ab+", "blood_type_ab-",
    "blood_type_b+", "blood_type_b-", "blood_type_o+", "blood_type_o-",
]
# Dietary pattern one-hot prefixes
_DIETARY_COLS = [
    "dietary_pattern_non_vegetarian", "dietary_pattern_pescatarian",
    "dietary_pattern_vegan", "dietary_pattern_vegetarian",
]
# Work-environment one-hot prefixes
_WORK_ENV_COLS = [
    "work_env_agriculture", "work_env_food_industry",
    "work_env_healthcare", "work_env_office", "work_env_other",
]

ALL_POSSIBLE_FEATURES = (
    BASE_FEATURE_COLS
    + _PROVINCE_COLS
    + _BLOOD_TYPE_COLS
    + _DIETARY_COLS
    + _WORK_ENV_COLS
)


# --------------------------------------------------------------------------- #
#  Model cache & known-key index                                               #
# --------------------------------------------------------------------------- #
_model_cache: dict[str, Any] = {}

# Lazily-built set of all model stems present in ARTIFACTS_DIR
# e.g. {"prawns", "crab", "beef", ...}
_known_model_keys: set[str] | None = None


def _get_known_model_keys() -> set[str]:
    """Return (and cache) the set of ingredient stems for which a model exists."""
    global _known_model_keys
    if _known_model_keys is None:
        _known_model_keys = set()
        if ARTIFACTS_DIR.exists():
            for f in ARTIFACTS_DIR.glob("risk_model_*.joblib"):
                stem = f.stem.removeprefix("risk_model_")
                _known_model_keys.add(stem)
        logger.debug("Known model keys: %s", _known_model_keys)
    return _known_model_keys


def _ingredient_to_model_key(ingredient: str) -> str:
    """
    Convert an ingredient name to the filename stem used when saving models.
    Mirrors the logic in train_backend_models.py:
        clean_name = target_food.lower().replace(' ', '_').replace('/', '_')
    """
    return ingredient.lower().strip().replace(" ", "_").replace("/", "_")


def _resolve_ingredient_key(raw_ingredient: str) -> tuple[str, str] | None:
    """
    Resolve a full ingredient phrase to a known model/rule key.

    Resolution order (first match wins):
      1. Synonym table          – "fresh jumbo prawns" → "prawns" via INGREDIENT_SYNONYMS
      2. Exact key match        – ingredient IS already a known model key
      3. Substring word search  – any known key appears as a whole word inside
                                  the ingredient phrase (e.g. 'prawns' in
                                  'fresh jumbo prawns')

    Returns:
        (raw_ingredient, resolved_key) tuple, or None if no match found.
    """
    norm = raw_ingredient.lower().strip()
    import re

    # 1. Synonym normalisation
    for synonym, canonical in INGREDIENT_SYNONYMS.items():
        # word-boundary match for the synonym inside the ingredient phrase
        if re.search(r'\b' + re.escape(synonym) + r'\b', norm):
            resolved = _ingredient_to_model_key(canonical)
            logger.debug(
                "Synonym match: '%s' → '%s' (via '%s')", raw_ingredient, canonical, synonym
            )
            return raw_ingredient, resolved

    # 2. Exact key match (ingredient name already equals a model key)
    exact_key = _ingredient_to_model_key(norm)
    if exact_key in _get_known_model_keys():
        return raw_ingredient, exact_key

    # 3. Substring word search across all known model keys
    for known_key in _get_known_model_keys():
        # Convert stored key back to a readable form for regex
        readable = known_key.replace("_", " ").replace("  ", " / ")
        pattern = r'\b' + re.escape(readable) + r'\b'
        if re.search(pattern, norm):
            logger.debug(
                "Substring match: '%s' → model key '%s'", raw_ingredient, known_key
            )
            return raw_ingredient, known_key

    return None


def _resolve_apriori_key(raw_ingredient: str) -> tuple[str, str] | None:
    """
    Same resolution logic but against SAFETY_RULES keys instead of model files.
    Returns (raw_ingredient, matched_rule_key) or None.
    """
    norm = raw_ingredient.lower().strip()
    import re

    # 1. Synonym normalisation
    for synonym, canonical in INGREDIENT_SYNONYMS.items():
        if re.search(r'\b' + re.escape(synonym) + r'\b', norm):
            if canonical in SAFETY_RULES:
                logger.debug(
                    "Apriori synonym: '%s' → rule key '%s'", raw_ingredient, canonical
                )
                return raw_ingredient, canonical

    # 2. Exact key match
    if norm in SAFETY_RULES:
        return raw_ingredient, norm

    # 3. Substring word search across rule keys
    for rule_key in SAFETY_RULES:
        pattern = r'\b' + re.escape(rule_key) + r'\b'
        if re.search(pattern, norm):
            logger.debug(
                "Apriori substring: '%s' → rule key '%s'", raw_ingredient, rule_key
            )
            return raw_ingredient, rule_key

    return None


def _load_model(ingredient: str) -> tuple[Any, str] | tuple[None, None]:
    """
    Resolve ingredient to a model key and return (model, resolved_key).
    Returns (None, None) if no model is found.
    """
    resolved = _resolve_ingredient_key(ingredient)
    if resolved is None:
        return None, None

    _, key = resolved
    if key not in _model_cache:
        model_path = ARTIFACTS_DIR / f"risk_model_{key}.joblib"
        if model_path.exists():
            try:
                _model_cache[key] = joblib.load(model_path)
                logger.debug("Loaded ML model for '%s' (key='%s')", ingredient, key)
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", key, exc)
                _model_cache[key] = None
        else:
            _model_cache[key] = None

    return _model_cache[key], key


# --------------------------------------------------------------------------- #
#  User-profile → feature vector                                               #
# --------------------------------------------------------------------------- #

def build_feature_vector(user_profile: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a flat user-profile dict into a single-row DataFrame whose columns
    match the feature set used at training time.

    The caller passes a dict with *any subset* of the known keys.  Missing
    values default to 0 (safe / unknown), which causes the model to predict
    the population-average baseline — a conservative, safe fallback.

    Args:
        user_profile: Dict containing demographic and medical data.
                      Recognised keys are listed in ALL_POSSIBLE_FEATURES.
                      One-hot fields (province, blood_type, etc.) should be
                      pre-expanded by the caller, OR the caller can pass the
                      raw string values under special convenience keys:
                        - "province"        → e.g. "western"
                        - "blood_type"      → e.g. "o+"
                        - "dietary_pattern" → e.g. "non_vegetarian"
                        - "work_env"        → e.g. "office"

    Returns:
        pd.DataFrame with shape (1, N) ready for model.predict().
    """
    row: dict[str, float] = {col: 0.0 for col in ALL_POSSIBLE_FEATURES}

    # ── scalar fields ──────────────────────────────────────────────────── #
    for col in BASE_FEATURE_COLS:
        if col in user_profile:
            val = user_profile[col]
            row[col] = float(val) if val is not None else 0.0

    # ── convenience one-hot expanders ──────────────────────────────────── #
    if "province" in user_profile and user_profile["province"]:
        prov_key = f"province_{user_profile['province'].lower().replace(' ', '_')}"
        if prov_key in row:
            row[prov_key] = 1.0

    if "blood_type" in user_profile and user_profile["blood_type"]:
        bt_key = f"blood_type_{user_profile['blood_type'].lower()}"
        if bt_key in row:
            row[bt_key] = 1.0

    if "dietary_pattern" in user_profile and user_profile["dietary_pattern"]:
        dp_key = f"dietary_pattern_{user_profile['dietary_pattern'].lower().replace(' ', '_')}"
        if dp_key in row:
            row[dp_key] = 1.0

    if "work_env" in user_profile and user_profile["work_env"]:
        we_key = f"work_env_{user_profile['work_env'].lower().replace(' ', '_')}"
        if we_key in row:
            row[we_key] = 1.0

    # Preserve any pre-expanded one-hot columns already in user_profile
    for col in ALL_POSSIBLE_FEATURES:
        if col in user_profile and col not in BASE_FEATURE_COLS:
            val = user_profile.get(col)
            if val is not None:
                row[col] = float(val)

    return pd.DataFrame([row], columns=ALL_POSSIBLE_FEATURES)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def predict_dish_safety(
    detected_ingredients: list[str],
    user_profile: dict[str, Any] | None = None,
) -> dict:
    """
    Unified Intelligence Engine — predicts allergen risk for a list of
    detected dish ingredients against a user's demographic/medical profile.

    Args:
        detected_ingredients: List of ingredient names (strings) found in
                              the dish by the vision + RAG pipeline.
        user_profile:         Dict of user demographic / medical features.
                              Pass None or {} to run with zero-vector
                              (population-average predictions).

    Returns:
        {
          "clinical_alerts": [
            {
              "ingredient": str,      # raw ingredient name
              "model_key": str,       # sanitised file-stem
              "risk_level": int,      # 1 = low, 2 = medium, 3 = high
              "msg": str
            }, ...
          ],
          "safety_net_warnings": [
            {
              "trigger": str,         # ingredient that fired the rule
              "linked_risks": list,   # co-reactive ingredients
              "msg": str
            }, ...
          ],
          "models_evaluated": int,    # how many ML models were found & run
          "ingredients_checked": int  # total ingredients evaluated
        }
    """
    if user_profile is None:
        user_profile = {}

    full_report: dict[str, Any] = {
        "clinical_alerts": [],
        "safety_net_warnings": [],
        "models_evaluated": 0,
        "ingredients_checked": len(detected_ingredients),
    }

    # ── Build feature vector once for all model calls ─────────────────── #
    try:
        X = build_feature_vector(user_profile)
    except Exception as exc:
        logger.error("Failed to build feature vector: %s", exc, exc_info=True)
        X = pd.DataFrame([{col: 0.0 for col in ALL_POSSIBLE_FEATURES}],
                         columns=ALL_POSSIBLE_FEATURES)

    # ── ML Engine — Clinical Prediction ───────────────────────────────── #
    # Track resolved keys to avoid running the same model twice when multiple
    # ingredient phrases map to the same underlying allergen (e.g. "fresh
    # jumbo prawns" and "prawns" both resolve to key "prawns").
    seen_model_keys: set[str] = set()

    for ingredient in detected_ingredients:
        model, resolved_key = _load_model(ingredient)
        if model is None or resolved_key is None:
            continue  # No model for this ingredient
        if resolved_key in seen_model_keys:
            continue  # Already evaluated this allergen via another phrasing
        seen_model_keys.add(resolved_key)

        full_report["models_evaluated"] += 1

        try:
            risk_pred = int(model.predict(X)[0])

            if risk_pred > 0:
                try:
                    proba = float(model.predict_proba(X)[0][1])
                    confidence_pct = f"{proba * 100:.1f}%"
                except Exception:
                    confidence_pct = "N/A"

                full_report["clinical_alerts"].append({
                    "ingredient": ingredient,
                    "model_key": resolved_key,
                    "risk_level": risk_pred,
                    "confidence": confidence_pct,
                    "msg": (
                        f"⚠️ Clinical risk detected for '{ingredient}' "
                        f"(resolved → '{resolved_key}', confidence: {confidence_pct}). "
                        "Your demographic profile suggests a heightened reaction probability."
                    ),
                })
                logger.info(
                    "ML model flagged '%s' (key='%s') | risk=%d | confidence=%s",
                    ingredient, resolved_key, risk_pred, confidence_pct,
                )
        except Exception as exc:
            logger.error(
                "Prediction failed for '%s' (key='%s'): %s",
                ingredient, resolved_key, exc, exc_info=True,
            )

    # ── Apriori Safety Net — Cross-reactivity Warnings ────────────────── #
    seen_rule_keys: set[str] = set()

    for ingredient in detected_ingredients:
        resolved = _resolve_apriori_key(ingredient)
        if resolved is None:
            continue
        _, rule_key = resolved
        if rule_key in seen_rule_keys:
            continue  # Avoid duplicate warnings for same allergen
        seen_rule_keys.add(rule_key)

        linked = SAFETY_RULES[rule_key]
        full_report["safety_net_warnings"].append({
            "trigger": ingredient,
            "linked_risks": linked,
            "msg": (
                f"🔗 Cross-reactivity alert: '{ingredient}' "
                f"(identified as '{rule_key}') is biologically associated "
                f"with {', '.join(linked)}. Individuals reactive to one may react to others."
            ),
        })
        logger.info(
            "Apriori rule fired for '%s' (rule_key='%s') → linked: %s",
            ingredient, rule_key, linked,
        )

    # ── Summary logging ───────────────────────────────────────────────── #
    logger.info(
        "ML risk prediction complete | ingredients=%d | models_run=%d | "
        "clinical_alerts=%d | safety_net_warnings=%d",
        len(detected_ingredients),
        full_report["models_evaluated"],
        len(full_report["clinical_alerts"]),
        len(full_report["safety_net_warnings"]),
    )

    return full_report
