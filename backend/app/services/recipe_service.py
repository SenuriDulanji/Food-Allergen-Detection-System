"""
services/recipe_service.py — Load and cache recipe data from JSON files.

The recipe JSONs are the ground-truth ingredient source.  After RAG identifies
the best-matching dish, we load its full ingredient list from the JSON file
rather than parsing it out of a ChromaDB chunk (which may be truncated).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the processed recipes directory (relative to project root)
# File: <project_root>/backend/app/services/recipe_service.py
# parents[0]=services, parents[1]=app, parents[2]=backend, parents[3]=project_root
RECIPES_DIR = Path(__file__).resolve().parents[3] / "dataset" / "recipes" / "processed"


@lru_cache(maxsize=20)
def _load_recipe_cached(dish_name: str) -> dict | None:
    """
    Load and cache a single recipe JSON by dish name.
    Uses lru_cache so the file is only read once per process lifetime.
    """
    json_path = RECIPES_DIR / f"{dish_name}.json"
    if not json_path.exists():
        logger.warning("Recipe file not found: %s", json_path)
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load recipe '%s': %s", dish_name, exc)
        return None


def get_ingredients(dish_name: str) -> list[str]:
    """
    Return the ingredient name list for the given dish.

    Args:
        dish_name: Snake_case dish name, e.g. "fish_curry".

    Returns:
        List of ingredient strings. Empty list if the recipe is not found.
    """
    recipe = _load_recipe_cached(dish_name)
    if recipe is None:
        return []
    return [ing["name"] for ing in recipe.get("ingredients", [])]


def get_all_dish_names() -> list[str]:
    """Return all known dish names from the processed recipes directory."""
    return [p.stem for p in sorted(RECIPES_DIR.glob("*.json"))]
