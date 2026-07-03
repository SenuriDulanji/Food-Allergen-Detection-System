import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path

# ==========================================
# 1. SETUP PATHS & LOAD DATA
# ==========================================
# Resolves paths relative to where the script is executed
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'preprocessed_data.csv'


print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ==========================================
# DEFINE TARGETS
# ==========================================
grid_food_cols = [
    'milk', 'eggs', 'peanuts', 'tree_nuts', 'wheat', 'soy', 'sesame', 'prawns',
    'crab', 'cuttlefish / squid', 'dry_fish', 'tuna', 'beef', 'pork', 'mutton', 'coconut',
    'spicy_oily'
]
other_food_cols = [
    'jackfruit', 'cassava', 'moringa', 'pineapple', 'tomato', 'rambutan',
    'sarana', 'avocado', 'breadfruit', 'calms', 'gotukola',
    'mustard', 'fenugreek', 'cumin'
]
identified_allergens = grid_food_cols + other_food_cols

y = df[identified_allergens]

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Prepare Data
# Ensure we are using Boolean (True/False)
basket = y.applymap(lambda x: True if x > 0 else False)

# 2. Run Apriori
# min_support=0.1 filters out noise (must appear in at least 10% of 90 people)
frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True, max_len=2)

# 3. Generate Rules
# Lift > 2.0 ensures we only keep strong biological/culinary associations
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2.0)

# 4. Filter and Sort
# We prioritize Confidence (Predictive Accuracy) and then Support (Generalizability)
correct_rules = rules.sort_values(by=['confidence', 'support'], ascending=[False, False])

print("--- VALIDATED CROSS-REACTIVITY RULES ---")
for index, row in correct_rules.iterrows():
    ant = list(row['antecedents'])[0].upper()
    cons = list(row['consequents'])[0].upper()
    
    print(f"RULE: [ {ant} ] -> [ {cons} ]")
    print(f"  Confidence : {row['confidence']*100:.1f}%")
    print(f"  Lift       : {row['lift']:.2f} (Strength)")
    print(f"  Support    : {row['support']*100:.1f}% of total population")
    print("-" * 40)