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

eval_dir = BASE_DIR / 'eval_results'
eval_dir.mkdir(parents=True, exist_ok=True)
csv_path = eval_dir / 'apriori_rules.csv'
txt_path = eval_dir / 'apriori_summary.txt'

# Save CSV (converting frozensets to strings for CSV compatibility)
correct_rules_csv = correct_rules.copy()
correct_rules_csv['antecedents'] = correct_rules_csv['antecedents'].apply(lambda x: ', '.join(list(x)))
correct_rules_csv['consequents'] = correct_rules_csv['consequents'].apply(lambda x: ', '.join(list(x)))
correct_rules_csv.to_csv(csv_path, index=False)

output_lines = ["--- VALIDATED CROSS-REACTIVITY RULES ---"]
print(output_lines[0])

for index, row in correct_rules.iterrows():
    ant = list(row['antecedents'])[0].upper()
    cons = list(row['consequents'])[0].upper()
    
    chunk = (f"RULE: [ {ant} ] -> [ {cons} ]\n"
             f"  Confidence : {row['confidence']*100:.1f}%\n"
             f"  Lift       : {row['lift']:.2f} (Strength)\n"
             f"  Support    : {row['support']*100:.1f}% of total population\n"
             + "-" * 40)
    print(chunk)
    output_lines.append(chunk)

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines) + "\n")

print(f"\nResults saved to:\n- {csv_path}\n- {txt_path}")