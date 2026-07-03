import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==========================================
# 1. SETUP PATHS & LOAD DATA
# ==========================================
# Resolves paths relative to where the script is executed
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'preprocessed_data.csv'
ARTIFACTS_DIR = BASE_DIR / 'artifacts' / 'models'

# Ensure the artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ==========================================
# 2. DEFINE FEATURES AND TARGETS
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

feature_cols = [
    'age', 'gender', 'personal_allergy_history',
    'outside_food_frequency', 'lactose_intolerance',
    'med_cond_asthma', 'med_cond_eczema', 'med_cond_food_allergy',
    'med_cond_none_of_these', 'med_cond_not_sure',
    'family_has_history', 'family_asthma', 'family_eczema',
    'family_allergy_beef', 'family_allergy_pork', 'family_allergy_sausage',
    'family_allergy_red_meat', 'family_allergy_seafood', 'family_allergy_dairy',
    'family_allergy_nuts', 'family_allergy_egg', 'family_allergy_tomato',
    'family_allergy_pineapple', 'family_allergy_avocado',
    'family_allergy_breadfruit', 'family_allergy_flour',
]

# Dynamically add one-hot encoded features
feature_cols += [col for col in df.columns if col.startswith('province_')]
feature_cols += [col for col in df.columns if col.startswith('blood_type_')]
feature_cols += [col for col in df.columns if col.startswith('dietary_pattern_')]
feature_cols += [col for col in df.columns if col.startswith('work_env_')]


# 1. Calculate the full Spearman correlation matrix
full_corr = df[feature_cols + identified_allergens].corr(method='spearman')

# 2. Isolate Features (Rows) vs Targets (Columns)
feature_target_corr = full_corr.loc[feature_cols, identified_allergens]

# 3. Find the MAXIMUM absolute correlation for each feature across ALL 31 targets
# (If a feature is useless for every single food, its max score will be very close to 0)
max_correlations = feature_target_corr.abs().max(axis=1)

# 4. Set your Noise Threshold
# 0.15 is a standard starting point for survey data. 
# Anything below 0.15 is generally considered statistical noise.
THRESHOLD = 0.15

# 5. Automatically separate the "Keepers" from the "Trash"
features_to_keep = max_correlations[max_correlations >= THRESHOLD].index.tolist()
features_to_drop = max_correlations[max_correlations < THRESHOLD].index.tolist()


X = df[features_to_keep]
y = df[identified_allergens]

print(f"Feature matrix X prepared: {X.shape[0]} rows x {X.shape[1]} features")

# ==========================================
# 3. MASS TRAINING & ARTIFACT SAVING LOOP
# ==========================================
print("\n🚀 STARTING MASS TRAINING FOR ALL 31 ALLERGENS...\n")

success_count = 0

for target_food in identified_allergens:
    # Binarize the target: 1 if > 0 (Risk), 0 if 0 (Safe)
    y_single_target = y[target_food].apply(lambda x: 1 if x > 0 else 0)
    positive_cases = y_single_target.sum()
    
    # SAFETY CHECK: Must have at least 2 cases for stratify to work
    if positive_cases < 2:
        print(f"⏭️ Skipping {target_food.upper()}: Only {positive_cases} case(s). Not enough data to train.")
        continue
        
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_single_target, 
        test_size=0.2, random_state=42, stratify=y_single_target
    )
    
    # Use Logistic Regression with balanced class weights for optimal recall
    # It outperforms tree-based models on sparse, highly-imbalanced survey data.
    model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    
    # Train and Save
    try:
        model.fit(X_train, y_train)
        
        # Clean the food name so it saves as a valid file name (e.g., 'cuttlefish / squid' -> 'cuttlefish___squid')
        clean_name = target_food.lower().replace(' ', '_').replace('/', '_')
        file_path = ARTIFACTS_DIR / f"risk_model_{clean_name}.joblib"
        
        # Save the trained model artifact to disk
        joblib.dump(model, file_path)
        
        success_count += 1
        print(f"✅ Trained & Saved: {target_food.upper()} (Cases: {positive_cases}) -> {file_path.name}")
        
    except Exception as e:
        print(f"❌ Failed to train {target_food.upper()}: {e}")

print(f"\n🎉 TRAINING COMPLETE! Successfully saved {success_count} model artifacts to {ARTIFACTS_DIR}.")