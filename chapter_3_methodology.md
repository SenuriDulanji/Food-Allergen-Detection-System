## Chapter 3: Methodology

3.1 Introduction

The methodological framework employed in this research has been detailed in this chapter. A multimodal Artificial Intelligence (AI) pipeline has been designed to address the complexities of Sri Lankan cuisine and the nuances of personalized allergen risk assessment. The procedures for data collection, preprocessing, model training, and system integration have been thoroughly documented. 

3.2 Overall System Architecture

The system architecture has been divided into four primary phases: Data Acquisition, Data Preprocessing, Model Training & Knowledge Base formulation, and System Integration & Evaluation. The integration of image analysis, vector databases, and machine learning models has been utilized to create a cohesive, real-time allergen detection mechanism.

*[INSERT Figure 3.1: Full Methodology Diagram HERE]*

3.3 Data Collection and Dataset Preparation

3.3.1 Image Acquisition of Sri Lankan Dishes
A specialized dataset encompassing 35 specific Sri Lankan dishes has been curated. A total of 554 dish images have been collected from public repositories, primarily utilizing Google Images. To facilitate efficient data loading during the model training and database initialization phases, the dataset was strictly organized within a local directory structure. Specifically, images were stored within a primary `data/images/` directory, sub-categorized into 35 distinct folders, where each folder name corresponded to the exact class label of the dish (e.g., `data/images/chicken_curry/`). Visually similar dishes, such as beef, pork, and mutton curries, have been deliberately included to capture real-world culinary complexities and test the system's limits.

*[INSERT Figure 3.2: Sri Lankan Dish Image Dataset Sample HERE]*

3.3.2 Textual Data Gathering (Recipes, Ingredients, and Allergen Profiles)
Thirty-five traditional Sri Lankan recipes, corresponding exactly to the selected dishes, have been gathered. Additionally, clinical and demographic survey data from 90 individuals have been collected using a structured online Google Form (available at: `[INSERT YOUR GOOGLE FORM LINK HERE]`). This survey data has included features such as age, gender, personal allergy history, dietary patterns, and family medical histories.

*[INSERT Figure 3.3: Google Form Questionnaire for Clinical Profiling HERE]*

*[INSERT Figure 3.4: Raw Clinical Data Sample HERE]*

3.3.3 Ethical Considerations and Data Anonymization
Due to the medical nature of the collected dietary and allergy profiles, strict ethical considerations have been adhered to during the data acquisition phase. It was explicitly stated in the header of the Google Form that all submitted data would be utilized exclusively for academic research purposes, thereby securing informed consent. Furthermore, the data collection procedure was completely anonymized; no personally identifiable information (PII) or email addresses were recorded, ensuring the total privacy and security of the 90 participants.

3.4 Data Preprocessing and Exploratory Data Analysis (EDA)

3.4.1 Exploratory Data Analysis (EDA) and Feature Optimization
Prior to model training, a rigorous Exploratory Data Analysis (EDA) has been conducted on the raw clinical dataset. To identify critical relationships between demographic variables, dietary habits, and food allergies, a correlation matrix has been generated. To prevent visual clutter and facilitate pattern recognition, a correlation heatmap has been visualized without numerical annotations, allowing major feature clusters to be identified. Through this analysis, features with zero variance or a maximum absolute correlation below a threshold of 0.15 against the target food allergens have been classified as statistical noise. This feature optimization process successfully reduced the original feature count from 49 to 41 optimal features. A total of 8 features (e.g., specific geographical provinces with no statistical variance, and rare family allergies such as 'family_allergy_nuts') were subsequently dropped to prevent model overfitting.

3.4.2 Categorical Encoding and Feature Scaling
The raw clinical features have been systematically transformed into a machine-readable numerical format, ultimately expanding the feature space dimensionality. Binary encoding has been applied in-place to categorical features such as Gender, Personal Allergy History, and Lactose Intolerance. Ordinal encoding has been utilized to quantify ordered variables, such as 'Allergy Awareness' and 'Outside Food Frequency'. Furthermore, One-Hot Encoding has been executed for nominal variables, including Province, Blood Type, and Dietary Pattern, while Multi-Label Binarization has been employed to flatten complex, comma-separated responses regarding Work Environment and Medical Conditions. Finally, continuous variables, specifically the 'Age' column, have been normalized utilizing a MinMaxScaler to ensure algorithmic stability.

3.4.3 Handling Imbalanced Dietary Data via Class-Weighted Algorithms
Clinical allergy data has been inherently imbalanced, as allergic reactions to specific foods constitute a minority class. Instead of relying on synthetic data distortion techniques, class-weighted algorithms have been implemented to balance the datasets algorithmically. By heavily penalizing the misclassification of the minority allergy class, the predictive models have been trained to prioritize clinical recall, ensuring that life-critical allergens are not ignored.

3.5 Machine Learning Pipeline for Allergen Risk Assessment

3.5.1 Model Selection (e.g., Logistic Regression)
A predictive model has been required to assess personalized allergen risks. Logistic Regression models, optimized with balanced class weights, have been trained on the filtered survey data. These models have been successfully utilized to predict individual exposure risks for 24 distinct allergens. Following training, the model weights have been saved as `.joblib` artifacts for backend integration.

3.5.2 Association Rule Mining for Allergen Cross-Reactivity (Apriori)
To uncover hidden cross-reactivities, the Apriori algorithm has been applied to the encoded Boolean dataset. To ensure statistical significance and filter out noise, the minimum support threshold has been set to 0.1, mandating that any itemset must appear in at least 10% of the surveyed population. The maximum rule length has been restricted to 2 to analyze direct pairwise biological associations. Subsequently, deterministic rules have been extracted utilizing a minimum lift threshold of 2.0, thereby ensuring only strong culinary and biological associations are retained. Through this process, a robust cross-reactivity safety net has been established alongside the probabilistic models, with the final rules prioritized by predictive confidence.

3.6 Multimodal Retrieval-Augmented Generation (RAG) System Design

3.6.1 Image Embeddings (e.g., CLIP)
Visual features from the 554 dish images have been extracted using the specific `openai/clip-vit-base-patch32` architecture of CLIP (Contrastive Language-Image Pre-training). The visual embeddings have been processed utilizing T4 Graphics Processing Unit (GPU) acceleration to handle the large matrix computations efficiently. These generated embeddings have been systematically stored in a persistent ChromaDB image vector database collection (`sri_lankan_recipes_images`).

3.6.2 Text Embeddings and Vector Database Configuration (e.g., ChromaDB)
The 35 gathered recipes have been parsed from JSON format and processed using a RecursiveCharacterTextSplitter, by which text chunks of 800 characters with a 100-character overlap have been generated. These text chunks have been vectorized using Google's `gemini-embedding-2-preview` multimodal embedding function and subsequently stored in a dedicated ChromaDB text vector database collection (`sri_lankan_recipes_text`).

*[INSERT Figure 3.5: JSON Recipe Structure HERE]*

3.6.3 Hybrid Reasoning Pipeline (Vision, Text RAG, Image RAG, LLM Synthesis)
A multimodal reasoning engine has been developed. When a dish image has been submitted, initial visual features have been extracted. Text and image database matches have been retrieved via vector similarity searches. A Large Language Model (LLM), specifically Gemini, has been utilized to synthesize these hybrid retrievals, and a final, accurate dish identification and ingredient list have been generated.

*[INSERT Figure 3.6: Multimodal Retrieval-Augmented Generation (RAG) Pipeline Diagram HERE]*

3.7 Evaluation Metrics and Validation Strategies

The multimodal RAG pipeline has been quantitatively evaluated. While the knowledge base has encompassed 35 dishes, the recognition pipeline has been tested on a representative subset of 10 known dishes (totaling 50 images) to assess in-distribution accuracy. Furthermore, 3 unseen, out-of-distribution dishes have been tested to evaluate system robustness. Classification performance has been measured using precision, recall, and F1-score, while retrieval accuracy has been evaluated using Top-1 and Top-3 RAG metrics.
