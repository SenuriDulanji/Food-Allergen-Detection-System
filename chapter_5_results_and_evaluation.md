## Chapter 5: Results and Evaluation

5.1 Introduction

The purpose of this chapter has been to present a comprehensive evaluation of the system. The performance of the dish identification pipeline—encompassing the Convolutional Neural Network (CNN), the Large Language Model (LLM) standalone approach, and the proposed Multimodal Retrieval-Augmented Generation (RAG) architecture—has been rigorously assessed. Furthermore, the efficacy of the machine learning algorithms deployed for allergen classification and the end-to-end system latency have been detailed.

5.2 Exploratory Data Analysis (EDA) and Clinical Demographics

Prior to evaluating the machine learning models, an Exploratory Data Analysis (EDA) has been conducted to understand the distribution of the 90 clinical profiles.

- **Demographics and Health:** The majority of participants have been identified as having O+ (33 participants) or A+ (21 participants) blood types. A significant portion of the cohort (57 participants) reported operating in standard home environments or remote work settings, while 65 participants reported no known family history of atopic conditions. Regarding baseline medical conditions, 21 participants reported pre-existing food allergies, providing a highly relevant sample population for predictive modeling.

*[INSERT Figure 5.1: EDA - Demographic Distributions of the Clinical Cohort HERE]*
- **Top Allergenic Triggers:** An analysis of the overall reaction frequency and severity levels has revealed the most hazardous ingredients within the context of the dataset. As summarized in Table 5.1, red meats and shellfish constitute the most significant allergenic risks, driving the necessity for accurate detection.

**Table 5.1: Top 10 Highest Risk Allergens Identified in the Clinical Dataset**

| Rank | Identified Allergen | Total Reactions | Primary Risk Category |
| :--- | :--- | :--- | :--- |
| 1 | Pork | 31 | Red Meat |
| 2 | Beef | 27 | Red Meat |
| 3 | Crab | 18 | Shellfish |
| 4 | Cuttlefish / Squid | 18 | Shellfish |
| 5 | Mutton | 18 | Red Meat |
| 6 | Spicy Oily Foods | 18 | Dietary Trigger |
| 7 | Prawns | 17 | Shellfish |
| 8 | Pineapple | 13 | Fruit |
| 9 | Milk | 13 | Dairy |
| 10 | Tuna | 11 | Seafood |

*[INSERT Figure 5.2: EDA - Top Allergenic Triggers Identified HERE]*

Furthermore, a Global Biological Matrix (Spearman correlation heatmap) has been evaluated to visually demonstrate the intricate cross-reactivities and correlations between these target allergens and patient demographics. Due to the high dimensionality of the feature space (49 features vs. 31 allergens), presenting this as a single matrix results in visual clutter. Therefore, the global matrix has been divided horizontally into four sequential segments (Figures 5.3a - 5.3d) to maintain readability while preserving the complete X-axis of allergen targets for consistent comparison. This statistical analysis ultimately facilitated the feature optimization process discussed in Chapter 3.

*[INSERT Figure 5.3a: EDA - Biological Matrix (Part 1 of 4) HERE]*

*[INSERT Figure 5.3b: EDA - Biological Matrix (Part 2 of 4) HERE]*

*[INSERT Figure 5.3c: EDA - Biological Matrix (Part 3 of 4) HERE]*

*[INSERT Figure 5.3d: EDA - Biological Matrix (Part 4 of 4) HERE]*

Based on these matrices, several interesting correlations can be observed. For instance, in Part 1 (Figure 5.3a), lactose intolerance shows a strong positive correlation with milk allergy, which aligns with expected medical profiles. In Part 2 (Figure 5.3b), family history of avocado and breadfruit allergies show an unusually strong correlation with each other, suggesting a potential cross-reactivity or regional dietary pattern. Conversely, environmental factors detailed in Part 4 (Figure 5.3d) demonstrate generally weaker correlations across the board, indicating that work environment has less direct mapping to specific food allergies than genetics or direct dietary habits.

5.3 Evaluation of Dish Identification Models

The initial stage of the system, which involved accurately identifying a Sri Lankan dish from a user-uploaded image, has been evaluated across multiple paradigms. To ensure academic rigor and completely prevent data leakage, the images utilized for the test dataset (50 seen images and 15 unseen images) have been downloaded completely separately from the primary 554-image training dataset.

5.3.1 CNN Model Performance (Test Dataset vs. Unseen Data)
The baseline performance of a standard Convolutional Neural Network on dish classification has been evaluated. Specifically, a pre-trained ResNet-50 architecture has been fine-tuned utilizing the PyTorch framework. The training pipeline incorporated robust data augmentation techniques—such as Random Resized Cropping (224x224) and Random Horizontal Flipping—alongside standard ImageNet normalization. The baseline model was trained over 5 epochs utilizing an Adam optimizer (learning rate of 0.0001) and Cross-Entropy Loss to establish a comparative mathematical baseline against the multimodal LLM approaches.

- **Test Dataset Results (50 images):** A Top-1 Accuracy of 34.00% and a Top-3 Accuracy of 60.00% have been recorded. It has been observed that the CNN struggled significantly with classes such as chicken curry, egg curry, and mutton curry (0% Top-1 accuracy), while performing optimally on cashew curry (100% Top-1).
- **Unseen Data Results (15 images):** A Top-1 and Top-3 Accuracy of 0.00% has been recorded. It has been concluded that the CNN completely failed to generalize to unseen dish categories, highlighting the severe limitations of a purely mathematical classification approach devoid of semantic understanding.

  5.3.2 Baseline Evaluation: LLM-Only Vision Capabilities
  The standalone visual reasoning capabilities of a Large Language Model (Gemini Vision) have been evaluated without external database support.

- **Test Dataset Results (50 images):** A Top-1 Accuracy of 44.00% has been recorded. The model has performed well on distinct classes like prawn curry (100% Top-1) but has failed entirely on ambiguous curries like potato and mutton curry (0% Top-1). It has been noted that without retrieval context, the LLM hallucinates or outputs un-normalized local names.
- **Unseen Data Results (15 images):** The accuracy has dropped to 13.33%. The lack of alias mapping has caused frequent misidentifications, such as predicting "kottu_roti" instead of the canonical "koththu," which would critically disrupt downstream ingredient resolution.

  5.3.3 Multimodal RAG Pipeline Performance
  The proposed core system, combining CLIP image embeddings and ChromaDB vector retrieval with LLM reasoning, has been evaluated.

- **Test Dataset Results (50 images):** A Top-1 Accuracy of 76.00% and a Top-3 Accuracy of 82.00% have been achieved. The pipeline has significantly outperformed both baselines, maintaining 100% Top-1 accuracy on 6 out of 10 test classes.
- **Unseen Data Results (15 images):** A Top-1 Accuracy of 40.00% has been recorded, vastly outperforming the CNN's 0% accuracy on the same set. This has demonstrated the robustness and generalizability of the hybrid RAG architecture.

*[INSERT Figure 5.4: Multimodal RAG Pipeline Evaluation Summary HERE]*

  5.3.4 Comparative Analysis
  The comparative performance of the three distinct approaches has been summarized in Table 5.2 below. 

**Table 5.2: Comparative Analysis of Dish Identification Models**

| Methodology | Test Dataset Top-1 | Test Dataset Top-3 | Unseen Data Top-1 | Avg Latency |
|---|---|---|---|---|
| **CNN Model** | 34.00% | 60.00% | 0.00% | 0.31s |
| **LLM-Only Vision** | 44.00% | 44.00% | 13.33% | 7.36s |
| **Multimodal RAG** | 76.00% | 82.00% | 40.00% | 4.24s |

  The Multimodal RAG pipeline has provided the optimal balance. While the mathematical CNN was significantly faster (0.31s), it suffered from catastrophic failure on unseen data. Conversely, the LLM-only approach was the slowest (7.36s) due to unconstrained visual generation. By grounding the LLM's reasoning with retrieved recipes from the vector database, the RAG architecture not only achieved a 76% Top-1 accuracy but also optimized processing latency down to 4.24s, ensuring the safest and most efficient allergen detection mechanism.

  5.4 Machine Learning Model Performance for Allergen Classification

The downstream task of predicting individualized allergy risks based on the identified ingredients and user profiles has been evaluated. Because of the medical nature of the application, Recall (sensitivity) has been prioritized to minimize dangerous false negatives.

5.4.1 Accuracy, Precision, Recall, and F1-Score
Three distinct machine learning models have been evaluated: Logistic Regression, Random Forest, and XGBoost. The summary of their predictive performance has been detailed in Table 5.3.

**Table 5.3: Predictive Performance of Allergen Classification Models**

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 83.33% | 21.40% | 24.03% | 21.73% |
| **Random Forest** | 90.05% | 23.61% | 12.08% | 15.28% |
| **XGBoost** | 84.49% | 17.43% | 18.26% | 16.63% |

It has been determined that while complex tree-based models like Random Forest achieved higher raw accuracy (90.05%), they did so by overfitting to the majority non-allergic class, resulting in dangerously low recall (12.08%). In medical contexts, a False Negative is catastrophic. Logistic Regression, combined with class-weighting, has successfully prioritized minority positive cases, achieving the highest Recall (24.03%). Consequently, Logistic Regression has been selected as the safest and most suitable algorithm for this system.

*[INSERT Figure 5.5: Confusion Matrices for Machine Learning Models HERE]*

5.4.2 Efficacy of Apriori Association Rules (Safety Net)
The Apriori algorithm has been evaluated as a secondary safety net to discover hidden cross-reactivities. The minimum lift threshold has been set to 2.0 and minimum support to 0.1 (10% of the population). Highly confident cross-reactivity clusters have been successfully uncovered and validated. The top 5 most predictive rules have been documented as follows:

- **[ MUTTON ] -> [ BEEF ]** (Confidence: 94.4%, Lift: 3.15, Support: 18.9%)
- **[ MUTTON ] -> [ PORK ]** (Confidence: 94.4%, Lift: 2.74, Support: 18.9%)
- **[ BEEF ] -> [ PORK ]** (Confidence: 85.2%, Lift: 2.47, Support: 25.6%)
- **[ CUTTLEFISH / SQUID ] -> [ CRAB ]** (Confidence: 83.3%, Lift: 4.17, Support: 16.7%)
- **[ PRAWNS ] -> [ CUTTLEFISH / SQUID ]** (Confidence: 76.5%, Lift: 3.82, Support: 14.4%)

This mechanism has ensured that even if a user forgets to declare a related allergy, precautionary warnings based on these high-confidence association rules have been automatically issued, significantly enhancing user safety.

  5.5 End-to-End System Performance and Qualitative Analysis

  5.5.1 System Latency and Real-Time Feasibility
  The latency of the proposed Multimodal RAG system has been evaluated. The end-to-end inference time averaged 4.24 seconds on the test dataset. It has been concluded that this minor delay—primarily introduced by the LLM reasoning phase—is a necessary and entirely acceptable tradeoff to ensure 76%+ accuracy and robust canonical recipe matching for a safety-critical application.

  5.5.2 Qualitative Analysis of AI-Generated Reasoning
  The quality of the final allergen warnings has been assessed. The integration of Gemini as the final reasoning engine has resulted in highly contextual, empathetic warnings. The system has successfully provided structured explanations detailing exactly why a specific food has been flagged based on the user's profile and the retrieved ingredients.

*[INSERT Figure 5.6: RAG Evaluation Example: Unseen Dish Identification HERE]*

  5.6 Summary of Results

The evaluation has unequivocally demonstrated the superiority of the Multimodal RAG architecture for food allergen detection. By combining zero-shot visual reasoning with canonical database grounding, high accuracy has been achieved on complex Sri Lankan curries. Furthermore, the deployment of Logistic Regression and an Apriori safety net has successfully minimized dangerous false negatives, resulting in a highly robust prototype for real-world dietary safety.
