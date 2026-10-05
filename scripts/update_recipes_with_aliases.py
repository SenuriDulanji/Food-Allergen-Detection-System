import json
from pathlib import Path

# The master aliases dictionary from evaluate_dish_identification.py
ALIASES = {
    "ala_hodhi": "potato_curry",
    "ala_kiriya": "potato_curry",
    "parippu": "dhal_curry",
    "parippu_curry": "dhal_curry",
    "kukul_mas": "chicken_curry",
    "kukul_mas_curry": "chicken_curry",
    "kakulu_mas": "crab_curry",
    "elu_mas": "mutton_curry",
    "kadju_curry": "cashew_curry",
    "biththara_curry": "egg_curry",
    
    # Jackfruit distinctions
    "kos_mallum": "kos_curry",
    "mature_jackfruit_curry": "kos_curry",
    "baby_jackfruit_curry": "polos_curry", 
    "jackfruit_curry": "kos_curry",
    
    # Hopper distinctions
    "appa": "plain_hoppers",
    "appam": "plain_hoppers",
    "biththara_appa": "egg_hoppers",
    "egg_hopper": "egg_hoppers",
    
    # Other mappings
    "kiri_bath": "milk_rice",
    "gotukola_sambol": "gotukola_sambal",
    "karawala_curry": "dry_fish_curry",
    "soya_curry": "soya_meat_curry",
    "bonchi_curry": "green_beans_curry",
    "wambatu_moju": "batu_moju",
    "dallo_curry": "squid_curry",
    "breadfruit_curry": "dell_curry",
    "isso_vada": "isso_vade",
    "isso_vadai": "isso_vade",
    "bitter_gourd_curry": "karawila_curry",
    "ambul_thiyal": "fish_ambulthiyal",
    "maalu_ambulthiyal": "fish_ambulthiyal",
    "fish_abul_thiyal": "fish_ambulthiyal",
    "idiyappam": "string_hoppers",
    "indiappa": "string_hoppers",
    "hbc": "hot_butter_cuttlefish",
    "cuttlefish": "hot_butter_cuttlefish",
    "pol_sambol": "pol_sambal",
    "kottu": "koththu",
    "kottu_roti": "koththu"
}

# Reverse mapping: group all aliases by the canonical dish name
canonical_to_aliases = {}
for alias, canonical in ALIASES.items():
    if canonical not in canonical_to_aliases:
        canonical_to_aliases[canonical] = []
    canonical_to_aliases[canonical].append(alias)

def update_recipes(recipes_dir: Path):
    if not recipes_dir.exists():
        print(f"[ERROR] Directory not found: {recipes_dir}")
        return

    updated_count = 0
    for json_file in recipes_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dish_name = data.get("dish_name")
        if not dish_name:
            continue

        aliases = canonical_to_aliases.get(dish_name, [])
        if aliases:
            # Inject aliases if they exist for this dish
            data["aliases"] = aliases
        else:
            # If no aliases exist, we can optionally add an empty list or leave it out.
            # Adding an empty list keeps schema consistent.
            data["aliases"] = []

        # Re-order the dictionary so 'aliases' appears directly under 'dish_name'
        new_data = {}
        for k, v in data.items():
            new_data[k] = v
            if k == "dish_name":
                new_data["aliases"] = data["aliases"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        updated_count += 1
        print(f"Updated {json_file.name} with aliases: {data['aliases']}")

    print(f"\nSuccessfully updated {updated_count} recipe files.")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "dataset" / "recipes" / "processed"
    update_recipes(processed_dir)
