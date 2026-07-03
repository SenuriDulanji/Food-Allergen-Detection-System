import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP PATHS & LOAD DATA
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'preprocessed_data.csv'

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

feature_cols += [col for col in df.columns if col.startswith('province_')]
feature_cols += [col for col in df.columns if col.startswith('blood_type_')]
feature_cols += [col for col in df.columns if col.startswith('dietary_pattern_')]
feature_cols += [col for col in df.columns if col.startswith('work_env_')]

# Same feature selection logic as train_backend_models.py
full_corr = df[feature_cols + identified_allergens].corr(method='spearman')
feature_target_corr = full_corr.loc[feature_cols, identified_allergens]
max_correlations = feature_target_corr.abs().max(axis=1)
THRESHOLD = 0.15
features_to_keep = max_correlations[max_correlations >= THRESHOLD].index.tolist()

X = df[features_to_keep]
y = df[identified_allergens]

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
print("\nEvaluating Models: XGBoost vs Random Forest vs Logistic Regression\n")

results = []

for target_food in identified_allergens:
    y_single_target = y[target_food].apply(lambda x: 1 if x > 0 else 0)
    positive_cases = y_single_target.sum()
    
    if positive_cases < 2:
        continue
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_single_target, 
        test_size=0.2, random_state=42, stratify=y_single_target
    )
    
    # Define models to compare
    models_to_test = {
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    }
    
    for model_name, clf in models_to_test.items():
        if positive_cases >= 4 and model_name == 'XGBoost':
            model = Pipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=2)),
                ('clf', clf)
            ])
        elif model_name == 'XGBoost':
            weight = (len(y_train) - positive_cases) / positive_cases
            clf.set_params(scale_pos_weight=weight)
            model = clf
        else:
            model = clf

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results.append({
                'Allergen': target_food,
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1_Score': f1
            })
        except Exception as e:
            pass

results_df = pd.DataFrame(results)

# Aggregate results
agg_results = results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Precision': 'mean',
    'Recall': 'mean',
    'F1_Score': 'mean'
}).reset_index()

print("=== AVERAGE PERFORMANCE ACROSS ALL ALLERGENS ===")
print(agg_results.to_string(index=False))
print("\n")

# Save detailed results to CSV
out_path = BASE_DIR / 'artifacts' / 'models' / 'evaluation_results.csv'
results_df.to_csv(out_path, index=False)
print(f"Detailed evaluation results saved to: {out_path}")
