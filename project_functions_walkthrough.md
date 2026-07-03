# Code Walkthrough: AI-Based Food Allergen Detection System

This document provides a complete guide to every function, notebook, and module in the project. The system has evolved from a simple rule-based and LLM-based checker into a comprehensive, personalized **Unified Intelligence Engine** that leverages deep machine learning and Apriori association rules.

---

## 🗺️ High-Level System Flow

```mermaid
graph TD
    A[Client / POST scan] --> B[scan_dish Router]
    B --> C[identify_dish_from_image]
    C -->|Gemini Vision Guess| D[retrieve_recipe_context]
    D -->|Similarity Search| E[pick_best_dish]
    E -->|Best Dish Match| F[get_ingredients]
    F -->|Load Recipe JSON| G[run_hybrid_allergen_detection]
    G -->|Layer 1: Rule-Based| H[scan_ingredients]
    G -->|Layer 2: LLM Check| I[llm_check_allergens]
    G -->|Layer 3: Merge & Deduplicate| J[merge_allergen_results]
    
    F -->|Ingredients + User Profile| K[ML Risk Service]
    K -->|Layer 4: Clinical XGBoost Models| L[predict_dish_safety]
    K -->|Layer 5: Apriori Cross-Reactivity| M[Safety Net Rules]
    
    J --> N[Consolidate Safety Warnings]
    L --> N
    M --> N
    N -->|LLM Explanation| O[generate_explanation]
    O --> P[Build & Return ScanResponse]
```

---

## 1. Machine Learning & Data Pipeline (`ml_model/`)

The machine learning pipeline predicts clinical risk for specific allergens based on a user's demographic and medical profile.

### `notebooks/preprocessing_data.ipynb`
* **Purpose**: Cleans raw survey data to prepare it for model training.
* **Logic**: 
  1. Loads raw CSV data and standardizes column names.
  2. Parses free-text and categorical columns (e.g., medical conditions, dietary patterns, provinces).
  3. Binarizes allergen severity (e.g., converting text severity descriptors into numerical risk values: 0 for Safe, 1-3 for Risk).
  4. One-hot encodes all categorical variables (Blood Type, Province, Diet, Medical Conditions).
  5. Exports the cleaned, numerically encoded dataset to `data/processed/preprocessed_data.csv`.

### `notebooks/model training.ipynb` & `scripts/train_backend_models.py`
* **Purpose**: Trains an independent, optimized machine learning model for *each* identified food allergen.
* **Logic**:
  1. **Feature Selection (Spearman Correlation)**: Calculates a Spearman correlation matrix between all features (demographics, medical conditions) and the 31 food allergen targets. Features with a maximum correlation below `0.15` (statistical noise threshold) are discarded.
  2. **Dynamic Architecture Generation**: Iterates through each of the 31 allergens.
     - **SMOTE + XGBoost**: If an allergen has $\ge$ 4 positive (risk) instances in the dataset, it utilizes a Pipeline combining SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, followed by an `XGBClassifier`.
     - **Weighted XGBoost**: If there are very few positive instances (< 4), it bypasses SMOTE and instead applies `scale_pos_weight` to an `XGBClassifier` to heavily penalize false negatives.
  3. **Artifact Serialization**: The trained model pipeline for each food is saved to disk as a `.joblib` file (e.g., `risk_model_prawns.joblib`).

### `scripts/apriory.py`
* **Purpose**: Discovers hidden association rules between food allergies (cross-reactivity).
* **Logic**: Uses the `mlxtend` library's Apriori and Association Rules algorithms on the binarized allergen dataset. It extracts rules with high confidence and lift (e.g., "If allergic to cuttlefish, highly likely allergic to crab"). These rules are hardcoded into the backend as the "Safety Net".

---

## 2. ML Risk Service (`backend/app/services/ml_risk_service.py`)

The Unified Intelligence Engine that evaluates personalized clinical risk at runtime.

### `build_feature_vector(user_profile: dict) -> pd.DataFrame`
* **Purpose**: Translates the user's saved profile into the exact 1D feature vector required by the trained XGBoost models.
* **Logic**: Initializes a dictionary with default values (`0.0`) for all columns identified during the feature selection phase. It then maps the `user_profile` values (like age, gender, one-hot encoded medical conditions, and blood types) into the correct columns and returns a Pandas DataFrame.

### `predict_dish_safety(user_profile: dict, detected_ingredients: list) -> dict`
* **Purpose**: The main ML evaluation function.
* **Logic**: 
  1. **Clinical Prediction (XGBoost)**: Loops through every single ingredient in the dish one by one. It normalizes each detected ingredient name and checks if a specific trained `.joblib` model exists for it. If a model exists, it uses the user's `feature_vector` to predict the risk. Crucially, the system only generates a `ClinicalAlert` if the model predicts a high risk (output of `1` or a probability crossing the threshold). If the model predicts an ingredient is safe for the specific user, or if there is no model trained for that ingredient (e.g., "garlic"), it stays completely silent to avoid cluttering the results.
  2. **Apriori Safety Net**: Cross-references the detected ingredients against the `SAFETY_RULES` dictionary. If the dish contains an ingredient known to co-react with another allergen, a `SafetyNetWarning` is generated (e.g., "Contains prawns, which cross-reacts with cuttlefish/crab").
  3. Returns a consolidated `MLRiskReport`.

---

## 3. Router Layer (`backend/app/routers/scan.py`)

Defines the endpoints exposed to clients (such as a Next.js or Flutter app).

### `scan_dish(image: UploadFile, user_id: int) -> ScanResponse`
* **Purpose**: `POST /api/v1/scan/` — The main orchestrator endpoint for the food scanning pipeline.
* **Logic**:
  1. **User Profile Retrieval**: Fetches the full user profile (demographics, medical conditions, saved allergens) from the SQLite DB using `user_id`.
  2. **Vision Identification**: Identifies the dish from the image via `identify_dish_from_image`.
  3. **RAG Context Retrieval**: Queries ChromaDB for recipe documents containing similar text to the vision guess via `retrieve_recipe_context`.
  4. **Recipe Lookup**: Selects the best dish name and loads ground-truth ingredients from recipe JSON files.
  5. **Hybrid Allergen Detection**: Matches ingredients against rules and verifies hidden ingredients via Gemini using `run_hybrid_allergen_detection`.
  6. **ML Risk Evaluation**: Calls `predict_dish_safety` with the user's comprehensive profile vector.
  7. **Safety Decision**: Overrides the `is_safe` boolean to `False` if *either* the rule-based checker flags a known user allergen OR the ML engine raises a clinical risk alert. Generates a dynamic warning message explaining the risks.

---

## 4. Multimodal RAG Service (`backend/app/services/rag_service.py`)

Handles image visual analysis and similarity searches to determine the correct dish.

### `identify_dish_from_image(image_bytes: bytes) -> str`
* **Purpose**: Performs Gemini Vision analysis to guess the dish name.
* **Logic**: Optimizes image bytes using PIL. Sends the image along with `VISION_PROMPT` to the `gemini-2.5-flash` model to respond solely with a snake_case name of a known dish.

### `retrieve_recipe_context(dish_guess: str, top_k: int) -> list[dict]`
* **Purpose**: Queries ChromaDB using the vision-guessed dish name.
* **Logic**: Calls `similarity_search_with_relevance_scores` on the Chroma vector store. Converts the results into list dicts containing: `content` (page content) and `score` (cosine similarity score).

### `pick_best_dish(...) -> tuple[str, str]`
* **Purpose**: Decides on the final dish name and calculates a confidence score (`high`, `medium`, or `low`) based on vector similarity thresholds.

---

## 5. Hybrid Allergen Service (`backend/app/services/allergen_service.py`)

Executes the multi-layered hybrid allergen detection system.

### `llm_check_allergens(dish_name: str, ingredients: list[str], user_allergens: list[str]) -> list[dict]`
* **Purpose**: Employs an LLM to cross-examine ingredients for hidden/compound allergens (e.g. checking if "curry powder" contains mustard seeds).

### `merge_allergen_results(rule_based: list, llm_results: list, user_allergens: list) -> list[dict]`
* **Purpose**: Deduplicates allergen categories detected across different layers and tags the sources (e.g. `rule_based+llm`).

### `generate_explanation(...) -> str`
* **Purpose**: Generates a friendly, natural-language warning or safety summary to display directly to the user.

---

## 6. Data Ingestion & Indexing (Google Colab - Multimodal RAG)

Prepares and populates the vector database (`ChromaDB`) with both recipe text and images. This ingestion process is executed separately in Google Colab to leverage a T4 GPU for high-speed CLIP embeddings.

### `populate_multimodal_rag()`
* **Purpose**: Initializes the persistent Chroma DB client and populates two separate vector collections: `sri_lankan_recipes_text` and `sri_lankan_recipes_images`.
* **Models Used**:
  - **Text Embeddings**: `GoogleGeminiEmbeddingFunction` utilizing the `gemini-embedding-2-preview` model.
  - **Image Embeddings**: `CLIPModel` and `CLIPProcessor` (Hugging Face) utilizing the `openai/clip-vit-base-patch32` model (512-dimension vector).
* **Logic**: 
  1. Reads JSON recipes and formats them into text chunks (800 char size, 100 overlap) using `RecursiveCharacterTextSplitter`.
  2. Embeds text chunks and saves them into the `sri_lankan_recipes_text` collection.
  3. Processes corresponding dish images, generating vectors using the GPU-accelerated CLIP model.
  4. Saves image vectors to the `sri_lankan_recipes_images` collection inside the persistent `chroma_db/` directory.

---

## 7. Core Database Layer (`backend/app/core/database.py`)

Handles singleton instances of ChromaDB and embedding client connections to ensure efficiency.

### `get_embeddings()` & `get_chroma_client()` & `get_vectorstore()`
* **Purpose**: Instantiates and caches the Gemini embedding model client and persistent ChromaDB connections to prevent connection overhead during scans.

---

## 8. Allergen Knowledge Base (`backend/app/utils/allergen_kb.py`)

A fast, local dictionary-based matcher.

### `detect_allergens_in_ingredient(ingredient: str, user_allergens: list[str]) -> list[dict]`
* **Purpose**: Evaluates a single ingredient string against keyword rules in `ALLERGEN_KB`.
* **Logic**: Quickly matches lowercase substrings (e.g. if the ingredient contains "wheat", it triggers the "gluten" allergen). To optimize performance, it only scans categories of interest to the user if a profile is provided.


Viewed rag_service.py:97-97
Viewed rag_service.py:110-249
Edited rag_service.py

I have upgraded the backend logging so you can trace exactly what the AI is thinking at each step! 

Whenever an image is uploaded and processed, you can now look at your `uvicorn` terminal, and you will see a detailed, step-by-step trace of the **Multimodal Dish Identification** process. 

Here is what happens during the identification, and exactly what will be logged in your terminal:

1. **Vision Initial Guess (Gemini 2.5 Flash)**: 
   The AI looks at the raw image first and makes a zero-shot guess at what Sri Lankan dish it might be.
   > 📝 *Log: `Vision model guessed dish: 'kottu_roti'`*

2. **Text RAG Retrieval (ChromaDB)**: 
   The system takes that guess and queries your text vector database, retrieving the closest matching recipe and ingredient contexts to anchor the AI in verified data.
   > 📝 *Log: `Text RAG retrieved 3 chunks for query 'kottu_roti'. Matches: ['kottu_roti', 'kottu_roti', 'chicken_kottu']`*

3. **Image RAG Retrieval (CLIP + ChromaDB)**: 
   In parallel, the system passes the uploaded image through the CLIP neural network, converts it to a mathematical vector, and searches your image database for visually similar dish photos. It retrieves the closest visual matches and calculates how similar they are (the lower the distance, the more identical the image).
   > 📝 *Log: `Image RAG retrieved 3 matches. Visual matches: ['chicken_kottu (dist: 0.2314)', 'vegetable_kottu (dist: 0.3129)', 'string_hopper_kottu (dist: 0.4501)']`*

4. **Final Hybrid Reasoning (Gemini 2.5 Flash)**: 
   The system gives the LLM all three pieces of context—its original visual guess, the text recipe matches, and the visually similar CLIP matches—and asks it to make a final, highly-confident conclusion about the dish.
   > 📝 *Log: `Final reasoning result: 'chicken_kottu' (Original Vision Guess: 'kottu_roti')`*

You can test this right now by scanning a dish on the frontend. Check your backend terminal window immediately after clicking "Scan for Allergens," and you'll see this entire four-step thought process printed out in real-time!