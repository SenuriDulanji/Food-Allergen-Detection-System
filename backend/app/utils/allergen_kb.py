"""
utils/allergen_kb.py — Rule-based allergen knowledge base for Sri Lankan dishes.

Architecture note:
  This is the FIRST layer of the hybrid allergen detection pipeline.
  It maps individual ingredient keywords → allergen categories so we can
  flag known allergens deterministically (no LLM call needed for clear cases).

  The Big-14 EU allergen categories are used as the standard, plus a few
  extras relevant to the Sri Lankan context.

How to extend:
  - Add more ingredient keywords under the relevant category list.
  - Add new categories at the bottom with their ingredient list.
  - Keyword matching is case-insensitive and checks if the keyword is a
    *substring* of the ingredient name.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
#  Knowledge base: allergen_category → list of ingredient keywords            #
# --------------------------------------------------------------------------- #
ALLERGEN_KB: dict[str, list[str]] = {

    # ── Big-14 EU allergens ──────────────────────────────────────────────── #

    "gluten": [
        "wheat", "flour", "bread", "roti", "noodle", "pasta",
        "barley", "rye", "spelt", "semolina", "seitan",
        "soy sauce",          # often contains wheat
    ],

    "crustaceans": [
        "prawn", "shrimp", "crab", "lobster", "crayfish",
        "jumbo prawn", "fresh prawn",
    ],

    "eggs": [
        "egg", "eggs", "omelette",
    ],

    "fish": [
        "fish", "tuna", "salmon", "sardine", "anchovy", "mackerel",
        "herring", "cod", "tilapia", "dried fish", "maldive fish",
        "umbalakada",         # Sri Lankan dried/salted tuna
    ],

    "peanuts": [
        "peanut", "groundnut", "peanut butter",
    ],

    "tree_nuts": [
        "cashew", "almond", "walnut", "pistachio", "pecan",
        "hazelnut", "macadamia", "brazil nut", "pine nut",
        "coconut",            # classified as tree nut by FDA
    ],

    "soy": [
        "soy", "soya", "tofu", "tempeh", "edamame",
        "soy sauce", "miso",
    ],

    "milk_dairy": [
        "milk", "cheese", "butter", "cream", "yogurt", "curd",
        "ghee", "paneer", "whey", "lactose",
        "coconut milk",       # not dairy, but flagged separately — see note below
    ],

    "sesame": [
        "sesame", "tahini", "til",
    ],

    "mustard": [
        "mustard", "mustard seed", "mustard powder", "mustard oil",
    ],

    "celery": [
        "celery", "celeriac",
    ],

    "lupin": [
        "lupin", "lupine",
    ],

    "molluscs": [
        "squid", "octopus", "mussel", "clam", "oyster", "snail",
        "scallop", "abalone",
    ],

    "sulphites": [
        "sulphite", "sulfite", "wine", "dried fruit", "vinegar",
    ],

    # ── Additional categories relevant to Sri Lankan context ──────────────── #

    "shellfish":   # convenience alias combining crustaceans + molluscs
        [
            "prawn", "shrimp", "crab", "lobster", "squid",
            "octopus", "mussel", "clam", "oyster", "scallop",
        ],

    "spices":      # some people are sensitive to specific spices
        [
            "chili", "chilli", "pepper", "fenugreek", "cumin",
            "coriander", "turmeric", "cardamom",
        ],
}

# Re-map "coconut milk" out of dairy for internal use — it's only flagged
# for "tree_nuts" (coconut). We keep it in milk_dairy above because some
# recipes label it loosely, but the LLM layer will contextualise it.

# --------------------------------------------------------------------------- #
#  Helper: check a single ingredient against the KB                           #
# --------------------------------------------------------------------------- #

def detect_allergens_in_ingredient(
    ingredient: str,
    user_allergens: list[str],
) -> list[dict]:
    """
    Check one ingredient string against every allergen category in the KB.

    Args:
        ingredient:      Ingredient name, e.g. "fresh jumbo prawns".
        user_allergens:  Allergen categories selected by the user, e.g. ["shellfish"].

    Returns:
        List of dicts: {"allergen_category": ..., "triggered_by": [ingredient], "source": "rule_based"}
        Empty list if no match.
    """
    ingredient_lower = ingredient.lower()
    found: list[dict] = []

    # Only check categories that the user cares about (if provided).
    # If user_allergens is empty, scan ALL categories (useful for full analysis).
    categories_to_check = (
        [c.lower() for c in user_allergens] if user_allergens else list(ALLERGEN_KB.keys())
    )

    for category, keywords in ALLERGEN_KB.items():
        if category.lower() not in categories_to_check:
            continue
        for kw in keywords:
            if kw.lower() in ingredient_lower:
                found.append({
                    "allergen_category": category,
                    "triggered_by": [ingredient],
                    "source": "rule_based",
                })
                break  # one match per category per ingredient is enough

    return found


# --------------------------------------------------------------------------- #
#  Helper: check a full ingredient list                                        #
# --------------------------------------------------------------------------- #

def scan_ingredients(
    ingredients: list[str],
    user_allergens: list[str],
) -> list[dict]:
    """
    Scan an entire ingredient list and return deduplicated allergen matches.

    Multiple ingredients can trigger the same allergen category.  They are
    merged so each category appears only once with all its trigger ingredients.
    """
    # category → set of triggering ingredients
    aggregated: dict[str, set] = {}

    for ing in ingredients:
        hits = detect_allergens_in_ingredient(ing, user_allergens)
        for hit in hits:
            cat = hit["allergen_category"]
            aggregated.setdefault(cat, set()).update(hit["triggered_by"])

    result = [
        {
            "allergen_category": cat,
            "triggered_by": sorted(triggers),
            "source": "rule_based",
        }
        for cat, triggers in aggregated.items()
    ]
    return result
