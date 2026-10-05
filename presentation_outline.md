# Full Presentation: A Multimodal AI Framework for Food Allergen Detection and Personalized Risk Prediction in Sri Lankan Cuisine

## Slide 1: Title Slide

- **Project Title:** A Multimodal AI Framework for Food Allergen Detection and Personalized Risk Prediction in Sri Lankan Cuisine
- **Presenter Name:** P.S.Dulanji
- **Index No:** 212240
- **Supervisor:** Mrs. R.P.T.H. Gunasekara (Senior Lecturer II, Dept. of Computing and Information Systems)
- **Second Supervisor:** Dr. (Mrs.) A.M. Namalie Thakshila Adikari (Senior Lecturer, Dept. of Nutrition and Dietetics)
- **Date / Event:** [Date]

---

## Slide 2: Introduction

### Background

- **The Evolving Landscape:** Digital health monitoring and food safety are increasingly relying on the intersection of computer vision, NLP, and medical informatics.
- **The Global Health Challenge:** Food allergies affect approximately 1% to 10% of the global population, posing life-threatening risks (e.g., IgE-mediated anaphylaxis).
- **The Sri Lankan Context:** Sri Lankan cuisine relies heavily on aromatic spices, legumes, and coconut bases, creating a high prevalence of "hidden" allergens that cannot be identified visually.

---

## Slide 3: The Problem Space

### The Current Challenges

- **Shortcomings of Existing AI:** Most existing AI food recognition tools are purely vision-only models designed primarily to detect calories or track diets, not to handle critical safety tasks like identifying hidden allergens.
- **Manual Tracking Failures:** Everyday prevention relies on manual tracking (e.g., reading labels), which is completely ineffective in environments lacking standardized allergen labeling (e.g., local Sri Lankan restaurants and street food cultures).
- **The Culinary Complexity:** The heavily spiced and mixed nature of Sri Lankan cuisine makes it extremely difficult for traditional vision-only systems to analyze effectively.

---

## Slide 4: Project Objectives

### Our Goals

- **Objective 1:** To develop a robust system that accurately identifies complex Sri Lankan dishes using a multimodal AI approach (bridging visual and textual data).
- **Objective 2:** To engineer a unified intelligence engine combining rule-based checks, Large Language Model (LLM) reasoning, and personalized Machine Learning models.
- **Objective 3:** To provide personalized, real-time allergen risk assessments based on a user's unique medical profile, moving beyond generic allergen warnings.

---

## Slide 5: Literature Review

### Existing Research Summarized

- **CNNs & Object Detection:** High categorization accuracy on standard datasets (e.g., Food-101), but struggle with fine-grained food images exhibiting high intra-class diversity.
- **Vision Transformers (ViTs):** Achieve state-of-the-art accuracy on controlled benchmarks but suffer significant performance drops on unseen, cross-cultural data (e.g., dropping to 72.92%).
- **Multimodal Fusion:** Advanced systems fuse visual features with textual documentation, achieving 92%-94% accuracy, but introduce significant computational complexities unsuitable for mobile deployment.
- **ML Classifiers for Risk:** Studies show high raw diagnostic accuracy (up to 98% for Decision Trees), but often evaluate models on generalized EHR data without addressing real-time visual input.

---

## Slide 6: Identified Research Gaps

Our literature review revealed four critical gaps that this research aims to fill:

1. **Absence of Localized Datasets:** A stark lack of multimodal datasets capturing the complex, hidden allergens unique to Sri Lankan cuisine (mainstream models suffer from Cultural Data Bias).
2. **Lack of Multimodal Dish Identification:** Fusing textual recipe data and computer vision to map hidden ingredients in local dishes has been neglected.
3. **The Clinical Disconnect:** No unified system bridges the evaluated food (computer vision) and the consuming individual (user demographic profiling).
4. **Imbalanced Demographic Profiling:** Existing ML models miss hidden cross-reactivities because they struggle to algorithmically balance minority classes (e.g., rare anaphylaxis) without relying on synthetic data distortion.

---

## Slide 7: The Proposed Solution

### A Multimodal Unified Intelligence Engine

Instead of a simple visual classifier, we propose a multi-layered framework:

- **Layer 1 (The Eyes):** A zero-shot visual classifier (Gemini Vision) combined with a RAG vector database (ChromaDB) to accurately identify complex Sri Lankan dishes without hallucination.
- **Layer 2 (The Brain):** A hybrid text engine (Rule-based + LLM) to break down retrieved recipes and resolve hidden ingredients.
- **Layer 3 (The Failsafe):** A personalized machine learning risk predictor (Logistic Regression) paired with an Apriori cross-reactivity safety net to evaluate the food against the user's specific clinical profile.

---

## Slide 8: Methodology Framework (Phases 1 & 2)

- **[Insert Methodology Flowchart Diagram Here]**

### Phase 1: Data Acquisition

- **Survey Data Collection:** Gathering demographic profiles, medical history, and allergy data.
- **Sri Lankan Recipes:** Curating textual data and ingredient lists from traditional recipe sources.
- **Dish Images:** Collecting visual representations of the targeted complex dishes.

### Phase 2: Data Preprocessing

- **Survey Data:** Data preprocessing and cleaning, followed by categorical encoding and binarization (translating raw data into 1D user profile vectors).
- **Textual Data:** Applying text chunking to recipe documents (e.g., 800 char size, 100 overlap) for efficient retrieval.
- **Visual Data:** Initial CLIP image processing and optimization for embedding generation.

---

## Slide 9: Methodology Framework (Phases 3 & 4)

### Phase 3: Model Training & Knowledge Base

- **Machine Learning Pipeline (from Survey Data):**
  - _Branch A (Risk Models):_ Feature selection (Spearman correlation) $\rightarrow$ Training class-weighted Logistic Regression models $\rightarrow$ Saving model weights as serialized `.joblib` files.
  - _Branch B (Safety Net):_ Applying the Apriori algorithm on binarized data to extract hidden cross-reactivity rules.
- **Vector Database Population (ChromaDB):**
  - _Text:_ Using Gemini models for text embeddings to populate the Text Vector DB.
  - _Images:_ Using Hugging Face models for CLIP image embeddings to populate the Image Vector DB.

### Phase 4: Integration & Evaluation

- **System Integration:** Fusing the trained `.joblib` models, Apriori rules, and dual ChromaDB vector stores into a unified **Multimodal Reasoning Engine**.
- **System Evaluation & Validation:** Rigorously testing the final system by measuring Precision, Recall, and F1-Score for the clinical risk models, and assessing RAG Top-K accuracy for the dish identification pipeline.

---

## Slide 10: Multimodal Dish Identification (The 4-Step Process)

- **Step 1: Vision Initial Guess.** The Gemini 3.1 Flash Lite model performs a zero-shot visual analysis of the uploaded image to generate an initial hypothesis (e.g., guessing `kottu_roti`).
- **Step 2: Text RAG Retrieval.** The system queries the ChromaDB text vector database with the vision guess, retrieving the closest matching traditional recipes and ingredient lists.
- **Step 3: Image RAG Retrieval.** The uploaded image is passed through a CLIP neural network. The resulting vector searches the image database for visually similar dish photos, calculating mathematical similarity distances.
- **Step 4: Final Hybrid Reasoning.** The system provides the LLM with all contexts—the original visual guess, text recipe matches, and visually similar CLIP matches—to synthesize a highly confident, final conclusion (e.g., correcting the guess to `chicken_kottu`).

---

## Slide 11: Advanced Ingredient Resolution (3-Tier System)

Before any risk is predicted, ingredients are rigorously normalized to prevent false positives and prepare for ML evaluation:

1. **Synonym Mapping:** Translates complex phrases using a prioritized dictionary (e.g., "peanut butter" -> "peanuts", "coconut milk" -> "coconut") to avoid false partial matches.
2. **Exact Match:** Checks if the ingredient name precisely aligns with known model keys.
3. **Substring Search:** Looks for known allergen model keys embedded within the ingredient phrase (e.g., identifying "prawns" inside "fresh jumbo prawns").

---

## Slide 12: Hybrid Allergen Detection Pipeline

- **Layer 1: Knowledge Base Matcher:** A fast, rule-based dictionary checker scans the normalized ingredients for direct, known allergen keywords.
- **Layer 2: LLM Hidden Allergen Analysis:** The LLM is deployed to cross-examine ingredients for hidden or compound allergens.
  - _Example:_ It identifies that "curry powder" may contain "mustard seeds," a correlation the rule-based system might miss.
- **Consolidation:** The system merges results from both layers, deduplicates categories, and tags the sources (e.g. `rule_based+llm`).

---

## Slide 13: Personalized ML Risk Engine

- **Independent Targeting:** Trained an independent machine learning model for each identified food allergen (skipping targets with < 2 positive cases), resulting in 24 serialized `.joblib` models.
- **Algorithm Choice:** We utilized Class-Weighted Logistic Regression (`class_weight='balanced'`).
- **Why Logistic Regression?:**
  - **Interpretability:** Highly valued in healthcare applications where decisions must be explainable.
  - **Maximizing Recall:** Outperforms complex tree-based models on sparse, highly-imbalanced survey data, successfully prioritizing minority positive cases (anaphylaxis) and minimizing dangerous false negatives.
- **Clinical Action:** A `ClinicalAlert` is generated only if the personalized model predicts a high probability of risk based on the user's specific 1D profile vector.

---

## Slide 14: Apriori Association Rules (The Safety Net)

- **The Failsafe Mechanism:** To provide a safety net for users who may not explicitly list all their related allergens, we utilized the Apriori algorithm (with a minimum lift threshold of 2.0) to mine our historical allergy dataset for hidden cross-reactivities.
- **Key Validated Cross-Reactivities:**
  - **Red Meat Allergy Cluster:** `[ MUTTON ] -> [ BEEF ]` (Confidence: 94.4%, Lift: 3.15). A user declaring only a "Mutton" allergy is safely warned against Beef and Pork.
  - **Shellfish/Seafood Cross-Reactivity:** `[ CUTTLEFISH / SQUID ] -> [ CRAB ]` (Confidence: 83.3%, Lift: 4.17). This extremely high lift proves a strong biological correlation.
  - **Unexpected Co-occurrences:** `[ PINEAPPLE ] -> [ PRAWNS ]` (Lift: 3.67). Potential link to combined histamine intolerance common in Sri Lankan diets.

---

## Slide 15: Results: Dish Identification Efficacy

### RAG Pipeline Efficacy (Dish Identification)

| Methodology         | Test Set (Top-1) | Test Set (Top-3) | Unseen Data (Top-1) |
| ------------------- | ---------------- | ---------------- | ------------------- |
| **CNN Model**       | 34.00%           | 60.00%           | 0.00%               |
| **LLM-Only Vision** | 44.00%           | 44.00%           | 13.33%              |
| **Multimodal RAG**  | **76.00%**       | **82.00%**       | **40.00%**          |

- **CNN Failure:** Memorizes training features, completely failing (0%) on unseen dishes.
- **LLM-Only Limitations:** Freely hallucinates un-normalized aliases without a vector database to provide grounded, canonical recipe names.
- **Multimodal RAG Superiority:** Achieved a high 76% Top-1 (and 82% Top-3) accuracy on the test set, demonstrating significant robustness by generalizing to completely unseen classes (40%).

---

## Slide 16: Results: ML Risk Prediction & System Latency

### ML Model Metrics (Allergen Risk Prediction)

- **[Insert Confusion Matrix Image Here (Highlighting True Positives vs False Negatives to emphasize Recall)]**

| Model                   | Accuracy | Precision | Recall     | F1-Score |
| ----------------------- | -------- | --------- | ---------- | -------- |
| **Logistic Regression** | 83.33%   | 21.40%    | **24.03%** | 21.73%   |
| **Random Forest**       | 90.05%   | 23.61%    | 12.08%     | 15.28%   |
| **XGBoost**             | 84.49%   | 17.43%    | 18.26%     | 16.63%   |

- **Accuracy vs. Recall Tradeoff:** Tree-based models achieved higher raw accuracy by overfitting to the non-allergic majority class, resulting in dangerously low recall. Logistic Regression successfully prioritized Recall (24.03%) and achieved the highest overall F1-Score (21.73%).
- **System Latency:** The end-to-end inference time averaged **4.24 seconds**. This minor delay is an acceptable tradeoff for high diagnostic safety and canonical recipe matching.

---

## Slide 17: Discussion: Comparison with Existing Systems

### 1. Food Recognition Architectures

| System Architecture | Performance Metric | Generalization (Unseen Data) | Deployment Feasibility |
| :--- | :--- | :--- | :--- |
| **Traditional CNN** | 96.88% (Easy Global Classification) | Poor (Drops to 20.66%) | Moderate |
| **Vision Transformers (ViT)** | 96.40% (Easy Global Classification) | Moderate (Drops to 72.92%) | **Poor** (Requires heavy server costs) |
| **Multimodal Fusion** | 92.0% - 94.0% (Classification) | High | **Poor** (Causes massive mobile bottlenecks) |
| **Our Proposed System (RAG)** | **76.00% (Complex Fine-Grained Retrieval)** | **Excellent** (LLM Override) | **Excellent** (Lightweight, ~4.24s latency) |

### 2. Allergen Risk Prediction Models

| Metric / Context | Existing Literature (e.g., Decision Trees) | Our Proposed System (Logistic Regression) |
| :--- | :--- | :--- |
| **Diagnostic Accuracy** | 98.00% Accuracy | 83.33% Accuracy |
| **Recall / Sensitivity** | Extremely low (e.g., 11% for RF) | **24.03% Recall** (Class-Weighted) |
| **Clinical Implication** | Dangerous false negatives | **Minimizes false negatives for safety** |

---

## Slide 18: Discussion: Unexpected Results & Challenges

### Explaining Unexpected Challenges

- **Visual Similarities:** Extreme similarities between Sri Lankan curries (e.g., beef vs. mutton curry) initially confused visual models. This was mitigated by our RAG architecture which cross-references text and imagery to avoid relying purely on superficial cues.
- **Out-of-Distribution Data:** Handling unseen dishes required deterministic prompt engineering to force the LLM to reject retrieved images if they heavily contradicted its initial visual hypothesis.

---

## Slide 19: Discussion: Scope & System Limitations

### System Limitations

- **Demographic Bias:** Personalized risk models trained on 90 user profiles (primarily university students); expanding this would enhance generalizability.
- **Data Scope:** The vector database is currently constrained to 35 core dish images and one traditional recipe book.
- **Network Dependency:** Reliance on the Gemini API introduces an external network dependency, unlike offline edge-computing models.

---

## Slide 20: System Architecture & Live Demo

### Architecture Overview

- **[Insert Implementation Diagram Here]**
- **Frontend (Next.js & React):** Premium glassmorphism UI for user profiling.
- **Backend (FastAPI):** Manages ChromaDB singleton instances, executes ML models, and traces step-by-step LLM reasoning.

### Live Demo Flow

- Walkthrough of the User Profiling form.
- Scanning a complex dish (e.g., Kottu or Kiribath).
- Breakdown of the resulting **Expandable Safety Report** (Contrasting detected ingredients with personalized ML Risk and Apriori warnings).

---

## Slide 21: Conclusion & Future Work

- **Significance & Implications:** We successfully developed a robust, personalized safety tool that bridges zero-shot visual reasoning with a localized vector database. By prioritizing sensitivity (Recall) and deploying an Apriori safety net, the system places user safety above pure computational accuracy, proving vital for complex cuisines.
- **Future Directions:**
  - Expanding the multimodal dataset to encompass other complex regional cuisines.
  - Deploying a native mobile application for on-the-go scanning.
  - Moving inference to the edge for complete offline capabilities.

---

## Slide 22: References

- Adithya, B., et al. (2025). Deep Multimodal Fusion for Ingredient Prediction from Food Images and Recipe Descriptions. _Journal of Information Systems Engineering and Management_, 10(32s).
- Ahmad, Z., et al. (2025). Food Allergy Detection Using Machine Learning Approach. _Kashf Journal of Multidisciplinary Research_, 2(4), 116-127.
- Bolaños, M., Ferrà, A., & Radeva, P. (2017). Food Ingredients Recognition through Multi-label Learning. _Lecture Notes in Computer Science_, 10742, 319-330. Springer.
- Liu, D., et al. (2025). Deep Learning in Food Image Recognition: A Comprehensive Review. _Applied Sciences_, 15, 7626.
- Louro, J., Fidalgo, F., & Oliveira, Â. (2024). Recognition of Food Ingredients—Dataset Analysis. _Applied Sciences_, 14(13), 5448.
- Reddy, B. D., et al. (2025). Food Image Analysis for Ingredient Identification Using Deep-Learning. _Global College of Engineering and Technology (GCET)_.

---

## Slide 23: Q&A

- Thank you for your time and attention.
- We will now open the floor for questions.
