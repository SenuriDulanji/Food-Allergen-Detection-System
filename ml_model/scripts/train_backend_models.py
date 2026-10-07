import sys
import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Safely configure standard output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. SETUP PATHS & LOAD DATA
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'preprocessed_data.csv'
ARTIFACTS_DIR = BASE_DIR / 'artifacts' / 'models'

# Ensure the artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

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

print(f"Total candidate features: {len(feature_cols)}")

# ==========================================
# 3. TARGET-SPECIFIC TRAINING & ARTIFACT SAVING LOOP
# ==========================================
print("\n🚀 STARTING TARGET-SPECIFIC MASS TRAINING FOR ALL 24 ALLERGENS (FULL DATASET)...\n")

THRESHOLD = 0.15
success_count = 0
trained_metadata = []

y = df[identified_allergens]

for target_food in identified_allergens:
    # Binarize the target: 1 if > 0 (Risk), 0 if 0 (Safe)
    y_single_target = y[target_food].apply(lambda x: 1 if x > 0 else 0)
    positive_cases = int(y_single_target.sum())
    
    # SAFETY CHECK: Must have at least 2 cases
    if positive_cases < 2:
        print(f"⏭️ Skipping {target_food.upper()}: Only {positive_cases} case(s). Insufficient data to train.")
        continue
    
    # Target-specific feature selection: correlate strictly with this target
    corr = df[feature_cols + [target_food]].corr(method='spearman')
    feature_corr = corr.loc[feature_cols, target_food].abs().fillna(0)
    features_to_keep = feature_corr[feature_corr >= THRESHOLD].index.tolist()
    if len(features_to_keep) < 2:
        features_to_keep = feature_corr.nlargest(3).index.tolist()
        
    X_target = df[features_to_keep]
    
    # Logistic Regression with balanced class weights selected for clinical interpretability,
    # transparent odds ratios, and deterministic low-latency edge inference
    model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    
    # Train and Save on full dataset
    try:
        model.fit(X_target, y_single_target)
        
        # Clean the food name for filename
        clean_name = target_food.lower().replace(' ', '_').replace('/', '_')
        file_path = ARTIFACTS_DIR / f"risk_model_{clean_name}.joblib"
        
        # Save the trained model artifact to disk
        joblib.dump(model, file_path)
        
        success_count += 1
        trained_metadata.append({
            "allergen": target_food,
            "artifact_file": file_path.name,
            "positive_cases": positive_cases,
            "total_samples": len(df),
            "features_count": len(features_to_keep),
            "selected_features": features_to_keep
        })
        print(f"✅ Trained & Saved: {target_food.upper():20s} (Cases: {positive_cases:2d}/{len(df)}, Features: {len(features_to_keep):2d}) -> {file_path.name}")
        
    except Exception as e:
        print(f"❌ Failed to train {target_food.upper()}: {e}")

# Save metadata json for downstream service transparency
metadata_path = ARTIFACTS_DIR / "model_metadata.json"
metadata_content = {
    "total_training_samples": len(df),
    "candidate_features_count": len(feature_cols),
    "candidate_features": feature_cols,
    "feature_selection_method": "Target-specific Spearman correlation (|rho| >= 0.15)",
    "threshold": THRESHOLD,
    "models_trained_count": success_count,
    "models": trained_metadata
}
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_content, f, indent=2)

# Save scaler reference parameters (min and max age) for inference transparency
scaler_info = {
    "feature": "age",
    "min_age": 14.0,
    "max_age": 82.0,
    "scale_range": 68.0
}
with open(ARTIFACTS_DIR / "scaler_params.json", 'w', encoding='utf-8') as f:
    json.dump(scaler_info, f, indent=2)

print(f"\n🎉 TRAINING COMPLETE! Successfully saved {success_count} model artifacts to {ARTIFACTS_DIR}.")
print(f"Model metadata written to: {metadata_path}")