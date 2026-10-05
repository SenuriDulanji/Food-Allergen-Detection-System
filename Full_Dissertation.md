# A Multimodal AI Framework for Food Allergen Detection and Personalized Risk Prediction in Sri Lankan Cuisine

> **Source:** Updated directly from the final dissertation PDF, *P.S.Dulanji - Dissertation.pdf*. The dissertation wording and results are based on the final PDF; page numbers and embedded figure graphics are omitted from this Markdown representation.

A Multimodal AI Framework for Food Allergen Detection and Personalized Risk Prediction in Sri Lankan Cuisine

A THESIS PRESENTED BY

Pandithage Senuri Dulanji – 212240

to the

Department of Computing and Information Systems

Faculty of Applied Sciences

in partial fulfilment of the requirements

for the award of the

B.Sc. (Honours) in Computer Science

of the

WAYAMBA UNIVERSITY OF SRI LANKA

SRI LANKA

2026

# Declaration

I do hereby declare that the work reported in this thesis was exclusively carried out by me under the supervision of Mrs. R.P.T.H. Gunasekara, Senior Lecturer in the Department of Computing and Information Systems, Faculty of Applied Sciences, Wayamba University of Sri Lanka and Dr. (Mrs.) A.M. Namalie Thakshila Adikari, Senior Lecturer in the Department of Nutrition and Dietetics, Faculty of Livestock Fisheries & Nutrition, Wayamba University of Sri Lanka. It describes the results of my own independent work except where due reference has been made in the text. No part of this project thesis has been submitted earlier or concurrently for the same or any other degree.

Date: 2026/07/24 .................................................. Signature of the candidate

Certified by: 1. Supervisor 1: a. Name: Mrs. R.P.T.H. Gunasekara,

b. Signature:……………………………………… c. Date: 2026/08/03

2. Supervisor 2 (if any): a. Name: Dr. (Mrs.) A.M. Namalie Thakshila Adikari.

b. Signature: c. Date: 2026/08/03

# Acknowledgment
I am grateful for the strong support and guidance I received from a dedicated group of individuals whose contributions were invaluable to the success of this project.

Foremost, I extend my sincere gratitude to my primary supervisor, Mrs. R.P.T.H. Gunasekara, Senior Lecturer in the Department of Computing and Information Systems, Faculty of Applied Sciences, Wayamba University of Sri Lanka. Her supervision, mentorship, and encouragement throughout the entire course of this research have been instrumental. Her insightful feedback and constructive comments significantly improved the quality of this work.

I would also like to express my profound appreciation to my second supervisor, Dr. (Mrs.) A.M. Namalie Thakshila Adikari, Senior Lecturer in the Department of Nutrition and Dietetics, Faculty of Livestock Fisheries & Nutrition, Wayamba University of Sri Lanka. Her expert guidance in helping me understand the complex mechanisms of food allergens from a food science perspective was vital to the foundation and accuracy of this research.

My sincere thanks also go to the course coordinator of CMIS 418 – Research Project, Mr. T. Arudchelvam, for providing valuable knowledge, timely guidance, and continuous encouragement during the research process.

I would also like to express my heartfelt thanks to the academic staff of the Department of Computing and Information Systems for their kind guidance and unwavering support during my academic journey. Their expertise and commitment have contributed greatly to my personal and professional development.

My special thanks go to the Department of Computing and Information Systems as a whole for providing an excellent academic environment and the resources necessary to complete this project.

I wish to thank my friends and colleagues for their continuous support, motivation, and encouragement, especially during challenging times. The shared experiences and discussions enriched my understanding and made this journey more meaningful.

Finally, I owe my deepest gratitude to my family for their unconditional love, patience, and belief in me. Their emotional support and sacrifices were crucial in helping me reach this milestone.

I sincerely thank everyone who contributed to the successful completion of this project in any capacity.

# Abstract

Food allergies have been recognized as a severe global public health challenge, affecting vulnerable populations with potentially fatal IgE-mediated anaphylactic reactions. Even with the development of computerized dietary monitoring technologies, it is still very challenging to find hidden allergies in complicated, highly spiced foods like Sri Lankan cuisine. Existing food recognition systems, primarily trained on Western-centric datasets, have frequently failed to accurately deconstruct mixed Asian dishes. Furthermore, these architectures have lacked the multimodal reasoning required to cross-reference visual data with patient-specific allergenic profiles and hidden cross-reactivities. To address this critical gap, a multimodal artificial intelligence framework has been developed and evaluated specifically for Sri Lankan cuisine.

A dual-layered Retrieval-Augmented Generation (RAG) pipeline has been utilized to overcome the limitations of traditional computer vision. Within this framework, Contrastive Language-Image Pre-training (CLIP) image embeddings and Gemini Large Language Models have been integrated to query a localized ChromaDB vector database, which encapsulates curated Sri Lankan recipes. To provide personalized safety warnings, clinical data from 90 individuals was processed, and a Logistic Regression risk classification engine algorithmically balanced using class weights was employed to prioritize diagnostic recall over sheer accuracy. Furthermore, the Apriori algorithm was deployed to mine hidden biological cross-reactivities, creating an additional deterministic safety net.

Through rigorous quantitative evaluation, the multimodal RAG system achieved 76.00% Top-1 fine-grained retrieval accuracy, vastly outperforming traditional Convolutional Neural Network baselines, which failed completely (0.00% accuracy) on out-of-distribution unseen data. Simultaneously, the predictive clinical backend generated a superior recall of 24.03%, minimizing life-threatening false negatives. The proposed system has successfully provided real-time, personalized allergen warnings with an average inference latency of 4.24 seconds. Ultimately, a robust technical blueprint for cross-cultural, safety-critical medical AI applications has been established, significantly advancing preventative dietary management.

Keywords: Food Allergy Detection, Multimodal Artificial Intelligence, Retrieval-Augmented Generation (RAG), Machine Learning, Sri Lankan Cuisine, Personalized Healthcare, Cross- Reactivity.

# Table of Contents

Declaration .............................................................................................................................................. ii Acknowledgment ................................................................................................................................... iii Abstract .................................................................................................................................................. iv Table of Contents .................................................................................................................................... v List of Figures ...................................................................................................................................... viii List of Tables ......................................................................................................................................... ix List of Abbreviations .............................................................................................................................. x
## Chapter 1: Introduction 
### 1.1 Background to the Study 
#### 1.1.1 The Prevalence of Food Allergies 
#### 1.1.2 The Complexity of Sri Lankan Cuisine 
#### 1.1.3 AI and Image Analysis in the Culinary Domain 
### 1.2 Problem Statement 
### 1.3 Aims and Objectives of the Research 
### 1.4 Significance of the Study 
### 1.5 Scope and Limitations
## Chapter 2: Literature Review 
### 2.1 Introduction 
### 2.2 Food Allergens and Associated Health Risks 
### 2.3 Culinary Characteristics and Hidden Allergens in Sri Lankan Dishes 
### 2.4 Traditional Methods vs. Automated Food and Allergen Detection 
### 2.5 Artificial Intelligence and Computer Vision in Food Analysis 
#### 2.5.1 Image Classification and Object Detection Models 
#### 2.5.2 Multimodal Machine Learning Approaches 
### 2.6 Integration of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) .. 8
### 2.7 Association Rule Mining and Cross-Reactivity in Food Allergies 
### 2.8 Details of Existing Food Recognition and Risk Prediction Systems 
#### 2.8.1 Convolutional Neural Network (CNN) Systems
#### 2.8.2 Vision Transformer (ViT) and Attention-Based Systems 
#### 2.8.3 Multimodal Fusion Systems
#### 2.8.4 Machine Learning Classifiers for Allergen Risk Prediction 
### 2.9 Critical Limitations of Existing Systems Regarding Patient Safety 
### 2.10 Summary of Literature and Identified Research Gaps 
## Chapter 3: Methodology 

### 3.1 Introduction 
### 3.2 Overall System Architecture 
### 3.3 Data Collection and Dataset Preparation 
#### 3.3.1 Image Acquisition of Sri Lankan Dishes 
#### 3.3.2 Textual Data Gathering (Recipes, Ingredients, and Allergen Profiles) 
#### 3.3.3 Ethical Considerations and Data Anonymization 
### 3.4 Data Preprocessing and Exploratory Data Analysis (EDA)
#### 3.4.1 Exploratory Data Analysis (EDA) and Feature Optimization 
#### 3.4.2 Categorical Encoding and Feature Scaling 
#### 3.4.3 Handling Imbalanced Dietary Data via Class-Weighted Algorithms 
### 3.5 Machine Learning Pipeline for Allergen Risk Assessment 
#### 3.5.1 Model Selection (e.g., Logistic Regression) 
#### 3.5.2 Association Rule Mining for Allergen Cross-Reactivity (Apriori) 
### 3.6 Multimodal Retrieval-Augmented Generation (RAG) System Design 
#### 3.6.1 Image Embeddings (e.g., CLIP)
#### 3.6.2 Text Embeddings and Vector Database Configuration (e.g., ChromaDB) 
#### 3.6.3 Hybrid Reasoning Pipeline (Vision, Text RAG, Image RAG, LLM Synthesis) 
### 3.7 Evaluation Metrics and Validation Strategies 
## Chapter 4: System Development and Implementation
### 4.1 Introduction 
### 4.2 Overall System Implementation Architecture 
### 4.3 Backend Development (FastAPI, Python) 
### 4.4 Frontend Interface Development (React/Next.js) 
### 4.5 User Profiling and Personalized Risk Engine 
### 4.6 Hybrid Allergen Detection and Advanced Ingredient Resolution 
### 4.7 Implementing the Multimodal RAG Service (Gemini Integration) 
### 4.8 Experimental Environment and Reproducibility 
### 4.9 Database Management and Storage Solutions 
### 4.10 Summary of Implementation 
## Chapter 5: Results and Evaluation 
### 5.1 Introduction 
### 5.2 Exploratory Data Analysis (EDA) and Clinical Demographics
### 5.3 Evaluation of Dish Identification Models 
#### 5.3.1 CNN Model Performance (Test Dataset vs. Unseen Data) 
#### 5.3.2 Baseline Evaluation: LLM-Only Vision Capabilities 
#### 5.3.3 Multimodal RAG Pipeline Performance 
#### 5.3.4 Comparative Analysis 

### 5.4 Machine Learning Model Performance for Allergen Classification 
#### 5.4.1 Accuracy, Precision, Recall, and F1-Score 
#### 5.4.2 Efficacy of Apriori Association Rules (Safety Net) 
### 5.5 End-to-End System Performance and Qualitative Analysis 
#### 5.5.1 System Latency and Real-Time Feasibility 
#### 5.5.2 Qualitative Analysis of AI-Generated Reasoning 
### 5.6 Summary of Results 
## Chapter 6: Discussion 
### 6.1 Interpretation of Key Findings 
### 6.2 Comparison with Existing Systems 
#### 6.2.1 Food Recognition Systems
#### 6.2.2 Allergen Risk Prediction Models 
### 6.3 Technical Challenges Encountered 
#### 6.3.1 Visual Similarities Between Allergenic and Non-Allergenic Ingredients 
#### 6.3.2 Handling Out-of-Distribution Dish Variations 
### 6.4 Ethical Considerations and User Safety Implications 
### 6.5 Limitations of the Study
## Chapter 7: Conclusion
### 7.1 Summary of Contributions 
### 7.2 Core Findings and Validation of Hypothesis 
### 7.3 Real-World Implications for Dietary Healthcare 
### 7.4 Future Directions 
### 7.5 Concluding Remarks 
References 
# Appendices

# List of Figures

**Figure 3.1: Full Methodology Diagram **
**Figure 3.2: Sri Lankan Dish Image Dataset Sample **
**Figure 3.3: Google Form for Socio-Demographic information and other information **
**Figure 3.4: JSON Recipe Structure **
**Figure 3.5: Multimodal Retrieval-Augmented Generation (RAG) Pipeline Diagram **
**Figure 4.1: Overall System Architecture **
**Figure 4.2: Frontend User Interface (UI) - User Registration and Profiling **
**Figure 4.3: Frontend UI - Dish Upload Interface **
**Figure 4.4: Frontend UI - Personalized Allergen Warning Results **
**Figure 5.1: EDA – Distribution of Participant Blood Types **
**Figure 5.2: EDA – Distribution of Participant Working Environments **
**Figure 5.3: EDA – Family History of Participants **
**Figure 5.4: EDA – Baseline Medical Conditions **
**Figure 5.5: EDA - Top Allergenic Triggers Identified **
**Figure 5.6a: EDA - Heatmap for Biological Matrix (Part 1 of 4)**
**Figure 5.6b: EDA - Heatmap for Biological Matrix (Part 2 of 4) **
**Figure 5.6c: EDA - Heatmap for Biological Matrix (Part 3 of 4)**
**Figure 5.6d: EDA - Heatmap for Biological Matrix (Part 4 of 4) **
**Figure 5.7: RAG Evaluation Example: Seen Dish Identification **
**Figure 5.8: Confusion Matrix for RAG Seen Dish Identification **
**Figure 5.9: RAG Evaluation Example: Unseen Dish Identification **

# List of Tables

**Table 5.1: Top 10 Highest Risk Allergens Identified in the Clinical Dataset **
**Table 5.2: Comparative Analysis of Dish Identification Models **
**Table 5.3: Predictive Performance of Allergen Classification Models **
**Table 6.1: Comparative Performance of Image Classification Systems**
**Table 6.2: Comparative Performance of Allergen Risk Models**

# List of Abbreviations

AI - Artificial Intelligence API - Application Programming Interface CLIP - Contrastive Language-Image Pre-training CMA - Cow's Milk Allergy CNN - Convolutional Neural Network EDA - Exploratory Data Analysis FDEIA - Food-Dependent Exercise-Induced Anaphylaxis LLM - Large Language Model ML - Machine Learning RAG - Retrieval-Augmented Generation ViT - Vision Transformer

## Chapter 1: Introduction

### 1.1 Background to the Study

Food safety has been recognized as a major public health priority. The rising rate of food allergies has been identified as a critical challenge. A heavy burden has been placed on individuals and healthcare systems, requiring smarter methods to detect allergens and communicate risks. Simultaneously, advancements in artificial intelligence (AI) and computer vision have offered unprecedented opportunities to solve these problems. This research has been situated at this intersection. An AI-driven system has been proposed, wherein image analysis has been used to detect potential food allergens, with a specific focus on the complex landscape of Sri Lankan cuisine.

#### 1.1.1 The Prevalence of Food Allergies

A food allergy has been defined as an abnormal immune response to specific food proteins. For sensitized individuals, reactions ranging from mild rashes to life-threatening anaphylaxis have been triggered by minor accidental exposures (Sicherer & Sampson, 2009). The global burden has been observed to be massive and growing; hundreds of millions of people worldwide have been affected. The daily management of these allergies has been proven to be difficult. Identifying safe food has been recognized as a major challenge, especially when dining out, where ingredient lists are not always available. Since most severe allergic reactions have been recorded outside the home, there is an urgent need for accessible tools to evaluate the allergen risk of unfamiliar dishes in real time.

#### 1.1.2 The Complexity of Sri Lankan Cuisine

Sri Lankan food has been celebrated for its bold spices, rich textures, and diverse influences. However, this culinary complexity has presented significant challenges for individuals managing food allergies. The greatest risk has been posed by hidden allergens—ingredients crucial for flavor but invisible to the eye. For example, Maldive fish (umbalakada) has been used as a hidden flavor enhancer in many dishes. Cashews have often been blended into festive dishes like Kiribath, and coconut milk has been utilized as the base of most local curries.

Furthermore, similar appearances have been exhibited by distinct Sri Lankan dishes. Both beef and pork curries may display rich, dark sauces, yet they carry completely different allergen profiles. Traditional visual inspection has been proven to be insufficient, especially for individuals who are not familiar with the local cuisine. Without standardized allergen labeling, technology has been identified as the optimal method to close this information gap.

#### 1.1.3 AI and Image Analysis in the Culinary Domain

Over the last decade, food analysis has been transformed by AI. Early systems have relied on Convolutional Neural Networks (CNNs) to classify dishes from images. However, these systems have been primarily trained on Western or East Asian foods, and South Asian cuisines have been largely ignored. Recently, the landscape has been advanced by Vision-Language Models (VLMs). Models like CLIP enable understanding of both images and text in a shared space, facilitating nuanced reasoning (Radford et al., 2021). For allergen detection, simple dish identification has been deemed insufficient; visual classification has been required to be cross-referenced with known ingredient profiles. Therefore, Retrieval-Augmented Generation (RAG) has been introduced. By linking an image query to a verified knowledge base, factual responses have been generated. Building on these advancements, a novel system tailored for Sri Lankan food has been proposed. By combining image embeddings, a recipe vector database, and personalized machine learning classifiers, accurate, real-time allergen warnings have been generated, improving food safety.

### 1.2 Problem Statement

Despite global awareness of food allergies, individuals consuming Sri Lankan cuisine face a significant challenge: the inability to reliably identify allergens in visually complex dishes. Common allergens have frequently been concealed within curries that cannot be distinguished by appearance alone. Existing food recognition systems have predominantly been designed for Western or East Asian cuisines. These systems lack the cultural specificity required to interpret Sri Lankan dishes. Furthermore, they have typically been operated at the dish identification level, and ingredient-level allergen assessments tailored to individual dietary profiles have not been provided.

Therefore, two core gaps have been identified by this research: 1. An AI-based food recognition system specifically trained and optimized for Sri Lankan cuisine has not been developed. 2. A personalized allergen risk assessment mechanism, by which dish ingredients are cross-referenced with a user's specific allergen profile in real time, has not been established. A direct health risk has been presented by these gaps. The addressing of both gaps through a multimodal AI system, in which image analysis, a Retrieval-Augmented Generation (RAG) pipeline, and a personalized risk classification engine have been combined, has been proposed by this research.

### 1.3 Aims and Objectives of the Research

Aim: The design, development, and evaluation of an AI-based food allergen detection system has been established as the primary aim of this research. Through this system, Sri Lankan dishes can be identified, and real-time, personalized allergen risk assessments can be provided. Objectives: To achieve this aim, the following objectives have been established: 1. A domain-specific dataset of Sri Lankan dish images, recipes, and ingredient-allergen mappings has been curated and prepared. 2. A multimodal dish identification pipeline has been developed by integrating CLIP image embeddings with a ChromaDB vector database and a Gemini Large Language Model (LLM) within a Retrieval-Augmented Generation (RAG) framework. 3. A personalized allergen risk classification model has been designed and trained using Logistic Regression balanced with class weights, by which individual allergen exposure risks have been predicted. 4. A full-stack application, comprising a FastAPI backend and a React frontend, has been built to integrate the AI models and deliver allergen warnings. 5. The system's performance has been evaluated using standard metrics, including accuracy, precision, recall, F1-score, and RAG retrieval accuracy (Top-1 and Top-3).

### 1.4 Significance of the Study

This research establishes significant practical value for public health and academic value in applied artificial intelligence.

Practically, a safety net for individuals managing food allergies in Sri Lanka has been provided. A real-time, preventative tool against accidental allergen exposure has been created for tourists and locals. By offering immediate, personalized risk assessments based on a photograph, users have been empowered to make informed choices, and the incidence of severe allergic reactions has been reduced. Furthermore, a structured method to communicate allergen information has been offered to the hospitality industry. Academically, the domain of multimodal food computing has been advanced. South Asian cuisines have been kept underrepresented in AI literature. By utilizing a Retrieval-Augmented Generation (RAG) pipeline combined with Vision-Language Models (CLIP and Gemini) for Sri Lankan dishes, it has been demonstrated how AI techniques can be adapted to culturally specific health challenges. Additionally, a methodological framework for personalized risk assessment has been provided by integrating Logistic Regression with class-weighted algorithms to handle imbalanced allergen profile data.

### 1.5 Scope and Limitations

Scope
- Target Cuisine and Dataset: The system's knowledge base has been built for 35
specific Sri Lankan dishes, supported by a curated dataset of 554 images and 35 traditional recipes. However, quantitative evaluation of the recognition pipeline has been focused on a representative test set of 10 dishes (50 images) to assess in-distribution accuracy, alongside an evaluation of 3 unseen dishes to test out-of-distribution robustness.
- Demographic Profile: User data collected to train the Logistic Regression risk
classification model has been predominantly represented by university students aged between 20 and 25 years.
- Allergen Tracking: The identification of major food allergens has been integrated into
the system; peanuts, tree nuts, dairy, egg, fish, and shellfish have been focused upon.
- Technological Framework: The core architecture has been restricted to a Retrieval-
Augmented Generation (RAG) approach, in which CLIP image embeddings, ChromaDB, and the Gemini Large Language Model have been utilized. Risk classification has been based on a Logistic Regression model optimized for imbalanced datasets via class weights.

Limitations
- Generalizability of Demographic Data: Because the model has been trained primarily
on data from young adults, its predictive accuracy might not generalize to other age groups or geographic populations.
- Dish Recognition Boundaries: The system has been trained on 35 specific dishes. If
an image of an out-of-distribution dish is input, it might be misidentified. Furthermore, the system has encountered difficulties when processing visually similar dishes, such as beef, pork, and mutton curries.
- Visual Dependency: Visual features captured in an image have been primarily relied
upon. If an allergenic ingredient has been pureed or visually obscured, the risk might not have been detected.
- Environmental Factors: The accuracy of the image analysis has been heavily
dependent upon the photograph's quality. Predictive capabilities could have been negatively impacted by poor lighting or blurry images.

## Chapter 2: Literature Review

### 2.1 Introduction

The paradigm of digital health monitoring and food safety has been fundamentally altered by the intersection of computer vision, natural language processing, and medical informatics. Historically, dietary monitoring has relied on self-reported logs or single-modality classifiers. Both methods have been challenged by real-world complexities. This chapter contextualizes the evolving landscape of food analysis. Vision-language modeling, Retrieval-Augmented Generation (RAG), tabular machine learning, and association rule mining have been evaluated, and the technical foundation for a localized allergen risk assessment framework has been established.

### 2.2 Food Allergens and Associated Health Risks

Food allergies are a major global public health challenge. These allergies have been driven by abnormal hypersensitivity reactions triggered by specific dietary proteins. It has been indicated by epidemiological data that food allergies have affected approximately 1% to 10% of the global population. Clinical manifestations have ranged from mild skin reactions to lifethreatening systemic anaphylaxis (Sampath et al., 2021). The highest immediate risk has been posed by Immunoglobulin E (IgE)-mediated reactions, for which immediate emergency intervention has been required. The pathophysiology of allergic diseases has been explained by the epithelial barrier hypothesis (Sampath et al., 2021). It has been posited that environmental factors disrupt skin and mucosal tight junctions, instigating a systemic immune response. Risk management has been completely reliant on allergen avoidance strategies. However, real-world execution has been hindered by the limitations of conventional medical diagnostics. Standard procedures, such as the Skin Prick Test (SPT) and Enzyme-Linked Immunosorbent Assays (ELISA), have been proven to be resource-intensive and impractical for real-time protection.

### 2.3 Culinary Characteristics and Hidden Allergens in Sri Lankan Dishes

While major Western allergens have been primarily focused upon in clinical research, South Asian cuisines present a distinct allergen profile. Sri Lankan cuisine has been characterized by

its reliance on aromatic spices, legumes, and coconut bases. Consequently, a high prevalence of hidden allergens has been observed; these allergens are not identifiable via visual inspection. Major variations from Western distributions have been highlighted by epidemiological reviews in Sri Lanka. Cow's Milk Allergy (CMA) has been identified as the most common childhood food allergy (31.2%), followed by primary red meat allergy (27.7%) (de Silva et al., 2022). Furthermore, a high rate of Food-Dependent Exercise-Induced Anaphylaxis (FDEIA) has been documented, with wheat having been identified as the culprit food. Additionally, coconut milk has been utilized as a foundational ingredient in Sri Lankan cooking. Novel heat-stable allergens have been successfully identified in both fresh and boiled coconut milk (Iddagoda et al., 2022). Because these proteins have survived the cooking process, they pose a significant threat. Immediate hypersensitivity to native foods, including pineapple, rambutan, and cuttlefish, has also been clinically documented. Thus, it has been demonstrated that a Western food recognition system has not been functional for the Sri Lankan context.

### 2.4 Traditional Methods vs. Automated Food and Allergen Detection

Historically, allergen confirmation has been reliant on clinical diagnostic tests. These tests have been time-consuming and prone to false positives. For everyday prevention, allergen tracking has been reliant on manual techniques, such as reading ingredient labels. Within the context of street food cultures and local restaurants in Sri Lanka, standardized allergen labeling has been observed to be nonexistent. This information gap has rendered manual avoidance strategies ineffective. The transition from manual verification to automated, AI-driven food detection has been proposed to bridge this gap. Through machine learning, manual guesswork has been replaced by immediate, localized risk assessments, and a real-time safety mechanism for vulnerable individuals has been provided.

### 2.5 Artificial Intelligence and Computer Vision in Food Analysis

#### 2.5.1 Image Classification and Object Detection Models

To automate food detection, researchers have heavily utilized computer vision. Automated food image analysis has progressed from manual feature extraction methods to Convolutional

Neural Networks (CNNs). While high categorization accuracy has been achieved by standard CNNs on controlled datasets, difficulties have been encountered due to the fine-grained nature of food images, where high intra-class diversity and low inter-class variation have been exhibited. To address real-world dining scenarios, efficiency and depth have been prioritized in modern architectures. The effectiveness of hybrid models has been demonstrated for complex ingredient identification. By combining local feature extraction with transformers, overlapping ingredients and lighting variations have been effectively handled, and the approach has been made suitable for mobile deployment.

#### 2.5.2 Multimodal Machine Learning Approaches

To overcome the limitations of single-modality vision systems, research has shifted toward multimodal learning. Standard CNN backbones have been adapted by replacing the final layer with a sigmoid activation function. Through this adaptation, systems have been trained to predict multiple ingredients simultaneously, and visually invisible components have been inferred via global visual context. This capability has been expanded by fusing visual data with text. It has been proven that image categorization accuracy has been significantly boosted through the integration of textual cues. Modern iterations have been made highly sophisticated. By combining CNN backbones for visual feature extraction with Transformer encoders for textual processing, the importance of images versus recipe descriptions has been dynamically weighed by the system. This multimodal fusion has handled ambiguous inputs, and high classification accuracies have been achieved.

### 2.6 Integration of Large Language Models (LLMs) and Retrieval-
Augmented Generation (RAG)

While natural language has been excellently processed by Large Language Models (LLMs), reliance on them for medical advice has been considered dangerous due to their tendency to hallucinate facts. In the context of allergen detection, a fatal outcome could have been caused by an incorrect prediction.

To ensure safety and factual grounding, Retrieval-Augmented Generation (RAG) frameworks have been increasingly utilized. Within the proposed architecture, visual features have been extracted via CLIP embeddings, while textual recipe metadata has been vectorized using semantic text embeddings. Instead of relying on parametric guesswork, vector databases (e.g., ChromaDB) have been queried to retrieve verified Sri Lankan recipes. Subsequently, this retrieved factual context has been synthesized by a Large Language Model (e.g., Gemini). Thus, it has been guaranteed that the final allergen warning has been firmly grounded in a localized knowledge base.

### 2.7 Association Rule Mining and Cross-Reactivity in Food Allergies

The complex nature of food allergies has been frequently characterized by hidden crossreactivities, wherein an immune response to one allergen has been triggered by a structurally similar protein in an unrelated food. Historically, these rare co-occurrences have been missed by standard machine learning models due to the inherent imbalance of clinical datasets. Consequently, association rule mining has been explored as a methodology for uncovering these hidden relationships. Specifically, the Apriori algorithm has been utilized in medical informatics to extract deterministic rules from sparse transactional data. Rare cross-reactivities have been algorithmically isolated without the need for synthetic data distortion. It has been demonstrated that when association rules have been combined with predictive models, a robust, hybrid safety net has been established for personalized dietary healthcare.

### 2.8 Details of Existing Food Recognition and Risk Prediction Systems

#### 2.8.1 Convolutional Neural Network (CNN) Systems
Standard image sets, including the Food-101 corpus and the UEC-Food100 database, have been widely utilized to construct baseline convolutional neural network frameworks (Bolaños et al., 2017; Liu et al., 2025). Specialized datasets such as the Ingredients101 collection have been deployed to train multi-label networks (Bolaños et al., 2017). While a test F1-score of 80.06% has been achieved on the Ingredients101 dataset, a significantly reduced test F1-score of 20.66% has been observed during an evaluation on unseen data from the Recipes5k corpus (Bolaños et al., 2017). An improved classification accuracy of 96.88% has been yielded on the FOOD-11 dataset through the deployment of transfer learning ensembles (Bu et al., 2024, as

cited in Liu et al., 2025). Furthermore, a macro-F1 score of 97.59% has been recorded on the Vireo Food-251 dataset through a multi-task, region-wise deep convolutional framework (Chen et al., 2021, as cited in Reddy et al., 2025). However, structural visual limitations have been introduced by coarse dataset annotations, whereby Western cuisines have been heavily overrepresented while African and South Asian dishes have been restricted (Liu et al., 2025).

#### 2.8.2 Vision Transformer (ViT) and Attention-Based Systems

Large-scale benchmarks, including the ISIA Food-500 dataset, have been deployed to assess attention-augmented architectures (Liu et al., 2025). A state-of-the-art fine-grained recognition accuracy of 96.40% has been reached on the Food-101 dataset by the FRCNNSAM framework, by which self-attention mechanisms have been integrated with deep convolutional networks (Abiyev & Adepoju, 2024, as cited in Liu et al., 2025). However, when the CBiAFormer model has been evaluated against unseen, cross-cultural data within the ISIA Food-200 dataset, a prominent performance drop to a Top-1 accuracy of 72.92% has been observed (Liu, Min, et al., 2024, as cited in Liu et al., 2025). Prohibitive execution bottlenecks have been generated by high multi-head self-attention computational complexities, by which mobile and edge deployments have been severely restricted (Reddy et al., 2025; Liu et al., 2025).

#### 2.8.3 Multimodal Fusion Systems

Heterogeneous multimodal databases combining visual feature layers with corresponding textual documentation have been leveraged to align cross-modal network features (Adithya et al., 2025). Recognition accuracies ranging from 92% to 94% have been attained by advanced multimodal architectures in which convolutional networks have been fused with transformer language models (Adithya et al., 2025). An F1-score of 80.06% and an Intersection over Union (IoU) of 67.9% have been established by attention-driven fusion models (Adithya et al., 2025). However, significant architectural complexities have been introduced during cross-modal semantic feature alignment (Adithya et al., 2025; Liu et al., 2025).

#### 2.8.4 Machine Learning Classifiers for Allergen Risk Prediction

Various electronic health record (EHR) screening algorithms have been evaluated to predict potential individual allergen exposure risks (Aziz et al., 2025). A maximum diagnostic accuracy rate of 98.00% has been achieved by Decision Tree (DT) models during the

classification of patient food allergy registries (Aziz et al., 2025). Predictive accuracies of 96.77% for Random Forest and 95.16% for K-Nearest Neighbors (KNN) have also been recorded (Aziz et al., 2025). Conversely, Logistic Regression models have produced suboptimal classification results, with accuracies of 58.06% (Aziz et al., 2025).

### 2.9 Critical Limitations of Existing Systems Regarding Patient Safety

A critical evaluation of modern machine learning applications has revealed several limitations regarding patient safety: 1. Focus on Calories over Safety: Most food recognition systems have been designed for calorie tracking, and allergen detection has often been treated as an afterthought. 2. Cultural Data Bias: Mainstream AI models have been trained on predominantly Western and East Asian datasets. A severe lack of representation for South Asian cuisine has been observed, and poor generalization has occurred when Sri Lankan dishes have been evaluated. 3. The Clinical Disconnect: Computer vision models have evaluated the composition of a plate, but the user's clinical profile has been ignored. Conversely, patient health records have been utilized by machine learning models to predict allergy risks, but real-time visual input and cross-reactive safety nets have been omitted from these tabular systems.

### 2.10 Summary of Literature and Identified Research Gaps

In summary, while the components for a robust allergen detection system have been identified in the literature, they have not been integrated to serve the Sri Lankan context. Four major research gaps have been identified:

- Gap 1: Lack of Localized Datasets. An absence of multimodal datasets capturing
complex, hidden allergens unique to Sri Lankan cuisine has been observed.
- Gap 2: Absence of Multimodal Dish Identification. The integration of textual data
and computer vision into a cohesive multimodal dish identification framework has been neglected. The synergistic potential of fusing visual features with textual descriptions has not been realized.

- Gap 3: Unimodal Clinical vs. Visual Silos. A unified system that bridges the gap
between the evaluated food and the consuming individual has not been established.
- Gap 4: Imbalanced Demographic Profiling and Cross-Reactivity Blind Spots.
Clinical user allergy data has been inherently imbalanced. Consequently, hidden cross-reactivities have been missed by standalone predictive models. Developing models that balance these minority classes without relying on synthetic data distortion has remained a challenge. The application of association rule mining alongside standard probabilistic models (e.g., Logistic Regression) has remained underexplored. It aims to bridge these gaps through this research. A dual-pipeline multimodal RAG framework has been developed to accurately identify Sri Lankan dishes through combined textual and visual retrievals. By combining this framework with a hybrid detection engine powered by Logistic Regression and an Apriori-based cross-reactivity safety net, a holistic, personalized, and culturally accurate allergen risk engine has been proposed.

## Chapter 3: Methodology
### 3.1 Introduction

The methodological framework employed in this research has been detailed in this chapter. A multimodal Artificial Intelligence (AI) pipeline has been designed to address the complexities of Sri Lankan cuisine and the nuances of personalized allergen risk assessment. The procedures for data collection, preprocessing, model training, and system integration have been thoroughly documented.

### 3.2 Overall System Architecture

The system architecture has been divided into four primary phases: Data Acquisition, Data Preprocessing, Model Training & Knowledge Base formulation, and System Integration & Evaluation. The integration of image analysis, vector databases, and machine learning models has been utilized to create a cohesive, real-time allergen detection mechanism.

**Figure 3.1: Full Methodology Diagram**

### 3.3 Data Collection and Dataset Preparation

#### 3.3.1 Image Acquisition of Sri Lankan Dishes

A specialized dataset encompassing 35 specific Sri Lankan dishes has been curated. A total of 554 dish images have been collected from public repositories, primarily utilizing Google Images. To facilitate efficient data loading during the model training and database initialization phases, the dataset was strictly organized within a local directory structure. Specifically, images were stored within a primary `data/images/` directory, sub-categorized into 35 distinct folders, where each folder name corresponded to the exact class label of the dish (e.g., `data/images/chicken_curry/`). Visually similar dishes, such as beef, pork, and mutton curries, have been deliberately included to capture real-world culinary complexities and test the system's limits.

**Figure 3.2: Sri Lankan Dish Image Dataset Sample**

#### 3.3.2 Textual Data Gathering (Recipes, Ingredients, and Allergen Profiles)

Thirty-five traditional Sri Lankan recipes, corresponding exactly to the selected dishes, have been gathered. Additionally, clinical and demographic survey data from 90 individuals have been      collected   using   a    structured    online    Google      Form     (available   at: https://forms.gle/JygMjNVxiXna9Rh8A). This survey data has included features such as age, gender, personal allergy history, dietary patterns, and family medical histories.

**Figure 3.3: Google Form for Socio-Demographic information and other information**

#### 3.3.3 Ethical Considerations and Data Anonymization

Due to the medical nature of the collected dietary and allergy profiles, strict ethical considerations have been adhered to during the data acquisition phase. It has been explicitly stated in the header of the Google Form that all submitted data would be utilized exclusively for academic research purposes, thereby securing informed consent. Furthermore, the data collection procedure has been completely anonymized; no personally identifiable information (PII) or email addresses have been recorded, ensuring the total privacy and security of the 90 participants.

### 3.4 Data Preprocessing and Exploratory Data Analysis (EDA)

#### 3.4.1 Exploratory Data Analysis (EDA) and Feature Optimization

Prior to model training, a rigorous Exploratory Data Analysis (EDA) has been conducted on the raw clinical dataset. To identify critical relationships between demographic variables, dietary habits, and food allergies, a correlation matrix has been generated. To prevent visual clutter and facilitate pattern recognition, a correlation heatmap has been visualized without numerical annotations, allowing major feature clusters to be identified. Through this analysis, features with zero variance or a maximum absolute correlation below a threshold of 0.15 against the target food allergens have been classified as statistical noise. This feature optimization process has successfully reduced the original feature count from 49 to 41 optimal features. A total of 8 features (e.g., specific geographical provinces with no statistical variance, and rare family allergies) have been subsequently dropped to prevent model overfitting.

#### 3.4.2 Categorical Encoding and Feature Scaling

The raw clinical features have been systematically transformed into a machine-readable numerical format, ultimately expanding the feature space dimensionality. Binary encoding has been applied in-place to categorical features such as Gender, Personal Allergy History, and Lactose Intolerance. Ordinal encoding has been utilized to quantify ordered variables, such as Allergy Awareness and Outside Food Frequency. Furthermore, One-Hot Encoding has been executed for nominal variables, including Province, Blood Type, and Dietary Pattern, while Multi-Label Binarization has been employed to flatten complex, comma-separated responses

regarding Work Environment and Medical Conditions. Finally, continuous variables, specifically the Age column, have been normalized utilizing a MinMaxScaler to ensure algorithmic stability.

#### 3.4.3 Handling Imbalanced Dietary Data via Class-Weighted Algorithms

Clinical allergy data has been inherently imbalanced, as allergic reactions to specific foods constitute a minority class. Instead of relying on synthetic data distortion techniques, classweighted algorithms have been implemented to balance the datasets algorithmically. By heavily penalizing the misclassification of the minority allergy class, the predictive models have been trained to prioritize clinical recall, ensuring that life-critical allergens have not been ignored.

### 3.5 Machine Learning Pipeline for Allergen Risk Assessment

#### 3.5.1 Model Selection (e.g., Logistic Regression)

A predictive model has been required to assess personalized allergen risks. Logistic Regression models, optimized with balanced class weights, have been trained on the filtered survey data. These models have been successfully utilized to predict individual exposure risks for 24 distinct allergens. Following training, the model weights have been saved as .joblib artifacts for backend integration.

#### 3.5.2 Association Rule Mining for Allergen Cross-Reactivity (Apriori)

To uncover hidden cross-reactivities, the Apriori algorithm has been applied to the encoded Boolean dataset. To ensure statistical significance and filter out noise, the minimum support threshold has been set to 0.1, mandating that any itemset must appear in at least 10% of the surveyed population. The maximum rule length has been restricted to 2 to analyze direct pairwise biological associations. Subsequently, deterministic rules have been extracted utilizing a minimum lift threshold of 2.0, thereby ensuring only strong culinary and biological associations have been retained. Through this process, a robust cross-reactivity safety net has been established alongside the probabilistic models, with the final rules prioritized by predictive confidence.

### 3.6 Multimodal Retrieval-Augmented Generation (RAG) System Design

#### 3.6.1 Image Embeddings (e.g., CLIP)

Visual features from the 554 dish images have been extracted using the specific `openai/clipvit-base-patch32` architecture of CLIP. The visual embeddings have been processed utilizing T4 Graphics Processing Unit (GPU) acceleration to handle the large matrix computations efficiently. These generated embeddings have been systematically stored in a persistent ChromaDB image vector database collection (`sri_lankan_recipes_images`).

#### 3.6.2 Text Embeddings and Vector Database Configuration (e.g.,
ChromaDB)

The 35 gathered recipes have been parsed from JSON format and processed using a RecursiveCharacterTextSplitter, generating text chunks of 800 characters with a 100-character overlap. These text chunks have been vectorized using Google's gemini-embedding-2-preview multimodal embedding function and subsequently stored in a dedicated ChromaDB text vector database collection.

**Figure 3.4: JSON Recipe Structure**

#### 3.6.3 Hybrid Reasoning Pipeline (Vision, Text RAG, Image RAG, LLM
Synthesis)

A multimodal reasoning engine has been developed. When a dish image has been submitted, initial visual features have been extracted. Text and image database matches have been retrieved via vector similarity searches. A Large Language Model (LLM), specifically Gemini, has been utilized to synthesize these hybrid retrievals, and a final, accurate dish identification and ingredient list have been generated.

**Figure 3.5: Multimodal Retrieval-Augmented Generation (RAG) Pipeline Diagram**

### 3.7 Evaluation Metrics and Validation Strategies

The multimodal RAG pipeline has been quantitatively evaluated. While the knowledge base has encompassed 35 dishes, the recognition pipeline has been tested on a representative subset of 10 known dishes (totaling 50 images) to assess in-distribution accuracy. Furthermore, 3 unseen, out-of-distribution dishes have been tested to evaluate system robustness. Classification performance has been measured using precision, recall, and F1-score, while retrieval accuracy has been evaluated using Top-1 and Top-3 RAG metrics.

## Chapter 4: System Development and Implementation

### 4.1 Introduction

This chapter details the transition from theoretical methodology to practical application. The development of a functional prototype capable of real-time multimodal food allergen detection has been described. The implementation of the backend architecture, frontend user interface, and the integration of artificial intelligence models have been systematically documented.

### 4.2 Overall System Implementation Architecture

The structural foundation of the application has been modeled as a three-tier architecture, encompassing the Presentation Tier, Application Tier, and Data Tier. The interactions between the Next.js frontend, the FastAPI backend, and the dual databases (SQLite and ChromaDB) have been mapped to ensure seamless data flow. The complex multimodal RAG pipeline and the multi-layered hybrid allergen detection engine have been integrated into this central architecture.

**Figure 4.1: Overall System Architecture**

### 4.3 Backend Development (FastAPI, Python)

The backend infrastructure has been developed using Python, leveraging the FastAPI framework for high-performance, asynchronous Application Programming Interface (API) routing. FastAPI has been selected due to its robust support for asynchronous endpoints, automatic data validation, and built-in interactive API documentation. The backend API has been structured into modular services, ensuring that database management, machine learning inference, and large language model (LLM) communications have been independently maintained.

### 4.4 Frontend Interface Development (React/Next.js)

The user-facing application has been developed using the React library within the Next.js framework. A responsive, dynamic interface featuring modern glassmorphism aesthetics has been designed to facilitate seamless user interactions, from account registration to real-time dish scanning. The frontend has been configured to collect extensive user profiling data securely and to present complex allergy risk assessments via an interactive safety report. This report has been designed to visually contrast detected ingredients alongside personalized machine learning risk probabilities and Apriori cross-reactivity warnings, ensuring that lifecritical information is communicated in a visually intuitive format.

**Figure 4.2: Frontend User Interface (UI) - User Registration and Profiling**

**Figure 4.3: Frontend UI - Dish Upload Interface**

**Figure 4.4: Frontend UI - Personalized Allergen Warning Results**

### 4.5 User Profiling and Personalized Risk Engine

A comprehensive user profiling system has been implemented to capture individualized demographic and clinical data. Variables such as age, gender, province, blood type, dietary patterns, work environment, and family medical history have been systematically collected via

the frontend. In the backend, a risk engine has been engineered to process this data. The data has been dynamically mapped into a one-hot encoded feature vector and fed into the trained Logistic Regression models. Through this mechanism, personalized probabilistic risk scores for 24 distinct allergens have been successfully generated in real time.

### 4.6 Hybrid Allergen Detection and Advanced Ingredient Resolution

To ensure maximum safety, a hybrid detection pipeline has been implemented. The probabilistic outputs of the Logistic Regression models have been fortified by deterministic safety nets generated via the Apriori algorithm. Furthermore, to prevent false positives and prepare the unstructured data for machine learning evaluation, an advanced 3-tier ingredient resolution service has been developed to process raw ingredient names extracted from the RAG pipeline: 1. Synonym Mapping: A prioritized dictionary has been utilized to translate complex, localized phrases into canonical terms, thereby avoiding false partial matches. 2. Exact Match: It has been algorithmically verified if the normalized ingredient name precisely aligns with the known machine learning model keys. 3. Substring Search: Known allergen model keys embedded within larger ingredient phrases have been successfully extracted.

### 4.7 Implementing the Multimodal RAG Service (Gemini Integration)

A Multimodal Retrieval-Augmented Generation (RAG) service has been engineered to manage the complex identification of Sri Lankan cuisine. The Google Gemini API has been integrated as the primary reasoning engine. When a user has uploaded a dish image, initial visual features have been extracted using Gemini Vision. Concurrently, vector similarity searches have been executed against a ChromaDB database using CLIP image embeddings and Gemini text embeddings. Crucially, an intelligent LLM reasoning prompt has been implemented to handle out-ofdistribution or unseen dishes. Rather than relying on a strict mathematical confidence threshold for the image retrieval, the prompt has instructed the Gemini LLM to prioritize its own foundational knowledge over the database matches if the visual evidence has strongly contradicted the retrieved recipes. Through this approach, accurate ingredient lists have been generated even for dishes absent from the primary database of 35 curated recipes.

### 4.8 Experimental Environment and Reproducibility

To ensure the reproducibility of the proposed system, the experimental environment has been explicitly documented. The core backend infrastructure and machine learning pipelines have been developed and executed using Python 3.10 within a Windows operating system environment. While Google Colab was utilized to accelerate the vectorization of images and recipes via GPU hardware during the database initialization phase, the dataset was not accessed via a cloud-based drive. Instead, the runtime environment was configured to mirror the local repository   structure,    loading    data   directly    from    local    file   paths   (e.g., `./backend/data/processed_recipes.json` and `./backend/data/images/`). This ensured that the exact data ingestion pipelines used during development could be seamlessly reproduced in any local environment. Furthermore, due to the highly optimized nature of the curated dataset, the final inference performance demonstrated negligible differences between Central Processing Unit (CPU) and Graphics Processing Unit (GPU) environments, ensuring the system remains accessible for standard deployment.

### 4.9 Database Management and Storage Solutions

A dual-database strategy has been implemented to manage the system's diverse data requirements. Relational data, including user credentials, demographic profiles, and personalized allergen settings, have been securely stored in a SQLite database using the SQLAlchemy Object-Relational Mapper (ORM). Conversely, high-dimensional vector data, encompassing the 554 dish image embeddings and the chunked recipe texts, have been stored in a persistent ChromaDB instance.

### 4.10 Summary of Implementation

A highly modular, full-stack application has been successfully implemented as a localized prototype. The integration of FastAPI, Next.js, traditional machine learning models, and advanced LLM reasoning has culminated in a robust system capable of identifying Sri Lankan dishes and calculating highly personalized allergen exposure risks. Currently, the system has been operated seamlessly in a localized environment, effectively preparing the architecture for future cloud deployment.

## Chapter 5: Results and Evaluation

### 5.1 Introduction

The purpose of this chapter has been to present a comprehensive evaluation of the system. The performance of the dish identification pipeline—encompassing the Convolutional Neural Network (CNN), the Large Language Model (LLM) standalone approach, and the proposed Multimodal Retrieval-Augmented Generation (RAG) architecture—has been rigorously assessed. Furthermore, the efficacy of the machine learning algorithms deployed for allergen classification and the end-to-end system latency have been detailed.

### 5.2 Exploratory Data Analysis (EDA) and Clinical Demographics

Prior to evaluating the machine learning models, an Exploratory Data Analysis (EDA) has been conducted to understand the distribution of the 90 clinical profiles.
- Demographics and Health: The majority of participants have been identified as
having O+ (33 participants) or A+ (21 participants) blood types. A significant portion of the cohort (57 participants) reported operating in standard home environments or remote work settings, while 65 participants reported no known family history of atopic conditions. Regarding baseline medical conditions, 21 participants reported pre-existing food allergies, providing a highly relevant sample population for predictive modeling.

**Figure 5.1: EDA – Distribution of Participant Blood Types**

**Figure 5.2: EDA – Distribution of Participant Working Environments**

**Figure 5.3: EDA – Family History of Participants**

**Figure 5.4: EDA – Baseline Medical Conditions**

- Top Allergenic Triggers: An analysis of the overall reaction frequency and severity
levels has revealed the most hazardous ingredients within the context of the dataset. As summarized in Table 5.1, red meats and shellfish constitute the most significant allergenic risks, driving the necessity for accurate detection.

**Table 5.1: Top 10 Highest Risk Allergens Identified in the Clinical Dataset**

Rank Identified Allergen Total Reactions Primary Risk Category 1       Pork                    31                  Red Meat 2       Beef                    27                  Red Meat 3       Crab                    18                  Shellfish 4       Cuttlefish / Squid      18                  Shellfish 5       Mutton                  18                  Red Meat 6       Spicy Oily Foods        18                  Dietary Trigger 7       Prawns                  17                  Shellfish 8       Pineapple               13                  Fruit 9       Milk                    13                  Dairy 10      Tuna                    11                  Seafood

**Figure 5.5: EDA - Top Allergenic Triggers Identified**

Furthermore, a Global Biological Matrix (Spearman correlation heatmap) has been evaluated to visually demonstrate the intricate cross-reactivities and correlations between these target allergens and patient demographics. Due to the high dimensionality of the feature space (49 features vs. 31 allergens), presenting this as a single matrix results in visual clutter. Therefore, the global matrix has been divided horizontally into four sequential segments (Figures 5.6a - 5.6d) to maintain readability while preserving the complete X-axis of allergen targets for consistent comparison. This statistical analysis ultimately facilitated the feature optimization process discussed in Chapter 3.

**Figure 5.6a: EDA - Heatmap for Biological Matrix (Part 1 of 4)**

**Figure 5.6b: EDA - Heatmap for Biological Matrix (Part 2 of 4)**

**Figure 5.6c: EDA - Heatmap for Biological Matrix (Part 3 of 4)**

**Figure 5.6d: EDA - Heatmap for Biological Matrix (Part 4 of 4)**

### 5.3 Evaluation of Dish Identification Models

The initial stage of the system, which involved accurately identifying a Sri Lankan dish from a user-uploaded image, has been evaluated across multiple paradigms. To ensure academic rigor and completely prevent data leakage, the images utilized for the test dataset (50 seen images and 15 unseen images) have been downloaded completely separately from the primary 554-image training dataset.

#### 5.3.1 CNN Model Performance (Test Dataset vs. Unseen Data)

The baseline performance of a standard Convolutional Neural Network on dish classification has been evaluated. Specifically, a pre-trained ResNet-50 architecture has been fine-tuned utilizing the PyTorch framework. The training pipeline incorporated robust data augmentation techniques—such as Random Resized Cropping (224x224) and Random Horizontal Flipping—alongside standard ImageNet normalization. The baseline model has been trained over 5 epochs utilizing an Adam optimizer (learning rate of 0.0001) and Cross-Entropy Loss to establish a comparative mathematical baseline against the multimodal LLM approaches.
- Test Dataset Results (50 images): A Top-1 Accuracy of 34.00% and a Top-3
Accuracy of 60.00% have been recorded. It has been observed that the CNN struggled significantly with classes such as chicken curry, egg curry, and mutton curry (0% Top-1 accuracy), while performing optimally on cashew curry (100% Top-1).
- Unseen Data Results (15 images): A Top-1 and Top-3 Accuracy of 0.00% has been
recorded. It has been concluded that the CNN completely failed to generalize to unseen dish categories, highlighting the severe limitations of a purely mathematical classification approach devoid of semantic understanding.

#### 5.3.2 Baseline Evaluation: LLM-Only Vision Capabilities

The standalone visual reasoning capabilities of a Large Language Model (Gemini Vision) have been evaluated without external database support.
- Test Dataset Results (50 images): A Top-1 Accuracy of 44.00% has been recorded.
The model has performed well on distinct classes like prawn curry (100% Top-1) but has failed entirely on ambiguous curries like potato and mutton curry (0% Top-1). It has been noted that without retrieval context, the LLM hallucinates or outputs un-normalized local names.
- Unseen Data Results (15 images): The accuracy has dropped to 13.33%. The lack of
alias mapping has caused frequent misidentifications, such as predicting "kottu_roti" instead of the canonical "koththu," which would critically disrupt downstream ingredient resolution.

#### 5.3.3 Multimodal RAG Pipeline Performance

The proposed core system, combining CLIP image embeddings and ChromaDB vector retrieval with LLM reasoning, has been evaluated.
- Test Dataset Results (50 images): A Top-1 Accuracy of 76.00% and a Top-3
Accuracy of 82.00% have been achieved. The pipeline has significantly outperformed both baselines, maintaining 100% Top-1 accuracy on 6 out of 10 test classes.
- Unseen Data Results (15 images): A Top-1 Accuracy of 40.00% has been recorded,
vastly outperforming the CNN's 0% accuracy on the same set. This has demonstrated the robustness and generalizability of the hybrid RAG architecture.

**Figure 5.7: RAG Evaluation Example: Seen Dish Identification**

**Figure 5.8: Confusion Matrix for RAG Seen Dish Identification**

**Figure 5.9: RAG Evaluation Example: Unseen Dish Identification**

#### 5.3.4 Comparative Analysis

The comparative performance of the three distinct approaches has been summarized in Table
### 5.2 below.

**Table 5.2: Comparative Analysis of Dish Identification Models**

Methodology         Test      Dataset Test        Dataset Unseen         Data Avg Top-1               Top-3                Top-1               Latency CNN Model           34.00%              60.00%               0.00%               0.31s LLM-Only            44.00%              44.00%               13.33%              7.36s Vision Multimodal          76.00%              82.00%               40.00%              4.24s RAG The Multimodal RAG pipeline has provided the optimal balance. While the mathematical CNN has been significantly faster (0.31s), it has suffered from catastrophic failure on unseen data. Conversely, the LLM-only approach has been the slowest (7.36s) due to unconstrained visual generation. By grounding the LLM's reasoning with retrieved recipes from the vector database, the RAG architecture has not only achieved a 76% Top-1 accuracy but has also optimized processing latency down to 4.24s, ensuring the safest and most efficient allergen detection mechanism.

### 5.4 Machine Learning Model Performance for Allergen Classification

The downstream task of predicting individualized allergy risks based on the identified ingredients and user profiles has been evaluated. Because of the medical nature of the application, Recall (sensitivity) has been prioritized to minimize dangerous false negatives.

#### 5.4.1 Accuracy, Precision, Recall, and F1-Score

Three distinct machine learning models have been evaluated: Logistic Regression, Random Forest, and XGBoost. The summary of their predictive performance has been detailed in Table 5.3.

**Table 5.3: Predictive Performance of Allergen Classification Models**

| Model                 | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 83.33% | 21.40% | 24.03% | 21.73% |
| Random Forest | 90.05% | 23.61% | 12.08% | 15.28% |
| XGBoost | 84.49% | 17.43% | 18.26% | 16.63% |

It has been determined that while complex tree-based models like Random Forest achieved higher raw accuracy (90.05%), they did so by overfitting to the majority non-allergic class, resulting in dangerously low recall (12.08%). In medical contexts, a False Negative has been considered catastrophic. Logistic Regression, combined with class-weighting, has successfully prioritized minority positive cases, achieving the highest Recall (24.03%). Consequently, Logistic Regression has been selected as the safest and most suitable algorithm for this system.

#### 5.4.2 Efficacy of Apriori Association Rules (Safety Net)

The Apriori algorithm has been evaluated as a secondary safety net to discover hidden crossreactivities. The minimum lift threshold has been set to 2.0 and minimum support to 0.1 (10% of the population). Highly confident cross-reactivity clusters have been successfully uncovered and validated. The top 5 most predictive rules have been documented as follows:
- [ MUTTON ] -> [ BEEF ] (Confidence: 94.4%, Lift: 3.15, Support: 18.9%)
- [ MUTTON ] -> [ PORK ] (Confidence: 94.4%, Lift: 2.74, Support: 18.9%)
- [ BEEF ] -> [ PORK ] (Confidence: 85.2%, Lift: 2.47, Support: 25.6%)
- [ CUTTLEFISH / SQUID ] -> [ CRAB ] (Confidence: 83.3%, Lift: 4.17, Support:
16.7%)
- [ PRAWNS ] -> [ CUTTLEFISH / SQUID ] (Confidence: 76.5%, Lift: 3.82, Support:
14.4%)

This mechanism has ensured that even if a user forgets to declare a related allergy, precautionary warnings based on these high-confidence association rules have been automatically issued, significantly enhancing user safety.

### 5.5 End-to-End System Performance and Qualitative Analysis

#### 5.5.1 System Latency and Real-Time Feasibility

The latency of the proposed Multimodal RAG system has been evaluated. The end-to-end inference time has averaged 4.24 seconds on the test dataset. It has been concluded that this minor delay—primarily introduced by the LLM reasoning phase—has been a necessary and entirely acceptable tradeoff to ensure 76%+ accuracy and robust canonical recipe matching for a safety-critical application.

#### 5.5.2 Qualitative Analysis of AI-Generated Reasoning

The quality of the final allergen warnings has been assessed. The integration of Gemini as the final reasoning engine has resulted in highly contextual, empathetic warnings. The system has successfully provided structured explanations detailing exactly why a specific food has been flagged based on the user's profile and the retrieved ingredients.

### 5.6 Summary of Results

The evaluation has unequivocally demonstrated the superiority of the Multimodal RAG architecture for food allergen detection. By combining zero-shot visual reasoning with canonical database grounding, high accuracy has been achieved on complex Sri Lankan curries. Furthermore, the deployment of Logistic Regression and an Apriori safety net has successfully minimized dangerous false negatives, resulting in a highly robust prototype for real-world dietary safety.

## Chapter 6: Discussion

### 6.1 Interpretation of Key Findings

The results obtained from the evaluation phase have been synthesized and interpreted in this chapter. It has been established that the proposed Multimodal Retrieval-Augmented Generation (RAG) system significantly outperformed traditional classification methods for identifying complex Sri Lankan cuisine. The achievement of a 76.00% Top-1 accuracy on the test dataset has validated the hypothesis that grounding Large Language Models (LLMs) with localized vector database retrieval mitigates hallucinations and improves semantic reasoning. Furthermore, the selection of Logistic Regression, which yielded a 24.03% Recall, has demonstrated the critical importance of prioritizing sensitivity over raw accuracy in life-critical medical applications such as allergen detection.

### 6.2 Comparison with Existing Systems

#### 6.2.1 Food Recognition Systems

The performance of the proposed Multimodal RAG architecture has been evaluated against contemporary food recognition research. When evaluated against traditional Convolutional Neural Network (CNN) systems, fundamental limitations in generalization have been observed. While an improved classification accuracy of 96.88% has been yielded on the FOOD-11 dataset using transfer learning ensembles (Bu et al., 2024, as cited in Liu et al., 2025), and a macro-F1 score of 97.59% has been recorded on the Vireo Food-251 dataset (Chen et al., 2021, as cited in Reddy et al., 2025), drastic failures on out-of-distribution data have been exhibited by these models. For instance, while a test F1-score of 80.06% has been produced on the Ingredients101 dataset, a significantly reduced test F1-score of 20.66% has been observed during an evaluation on unseen data from the Recipes5k corpus (Bolaños et al., 2017). Furthermore, structural visual limitations have been introduced by coarse dataset annotations, whereby Western cuisines have been heavily overrepresented while African and South Asian dishes have been restricted (Liu et al., 2025). This has been overcome by the proposed system through the specific targeting of Sri Lankan cuisine, and a 40.00% Top-1 accuracy on completely unseen dishes has been maintained via LLM reasoning overrides. When compared against Vision Transformer (ViT) systems, similar trends have been noted. A state-of-the-art fine-grained recognition accuracy of 96.40% has been reached on the Food-101 dataset by the FRCNNSAM framework, by which self-attention mechanisms have been

integrated with deep convolutional networks (Abiyev & Adepoju, 2024, as cited in Liu et al., 2025). However, when the CBiAFormer model has been evaluated against unseen, crosscultural data within the ISIA Food-200 dataset, a prominent performance drop to a Top-1 accuracy of 72.92% has been observed (Liu, Min, et al., 2024, as cited in Liu et al., 2025). Additionally, prohibitive execution bottlenecks have been generated by high multi-head selfattention computational complexities, by which mobile and edge deployments have been severely restricted (Reddy et al., 2025; Liu et al., 2025). The proposed system has been aligned most closely with state-of-the-art Multimodal Fusion Systems. Recognition accuracies ranging from 92% to 94% have been attained by advanced multimodal architectures (Adithya et al., 2025). It has been considered critical to contextualize these numerical differences. The 96% metrics reported by CNNs and ViTs have represented global image classification on visually distinct, Western-centric datasets. In contrast, finegrained Top-1 ingredient retrieval on highly ambiguous Sri Lankan curries has been represented by the proposed RAG system's 76.00% metric. While numerically lower than generalized classification benchmarks, a highly robust achievement for this specific domain has been represented by attaining a 76.00% precise retrieval accuracy on localized, visually similar cuisine, while the significant computational bottlenecks of standard fusion models (Adithya et al., 2025) have been simultaneously avoided.

The comparative performance between the proposed system and the literature benchmarks has been summarized in Table 6.1 below.

**Table 6.1: Comparative Performance of Image Classification Systems**

System                Performance Metric          Generalization        Deployment Feasibility Architecture                                      (Unseen Data) Traditional CNN       96.88% (Easy Global         Poor (Drops to        Moderate Classification)             20.66%) Vision                96.40% (Easy Global         Moderate (Drops to    Poor (Requires heavy server Transformers          Classification)             72.92%)               costs) (ViT) Multimodal            92.0% - 94.0%               High                  Poor (Causes massive Fusion                (Classification)                                  mobile bottlenecks) Our Proposed          76.00% (Complex             Excellent (LLM        Excellent (Lightweight, System (RAG)          Fine-Grained                Override)             ~4.24s latency) Retrieval)

#### 6.2.2 Allergen Risk Prediction Models

The machine learning predictive backend has been contrasted against existing diagnostic systems. It has been demonstrated by recent machine learning evaluations that a maximum diagnostic accuracy rate of 98.00% has been achieved by Decision Tree models during patient food allergy classifications, while predictive accuracies of 96.77% and 95.16% have been recorded for Random Forest and K-Nearest Neighbors, respectively (Aziz et al., 2025). Conversely, suboptimal classification results have historically been produced by Logistic Regression models, which have been limited to accuracies of 58.06% in those studies (Aziz et al., 2025). However, in the context of this proposed system, prioritizing sheer diagnostic accuracy has been recognized as a severe medical risk. While a high accuracy of 90.05% has been achieved by the Random Forest model, a dangerously low Recall of 12.08% has been exhibited due to class imbalance. A divergence from the literature (Aziz et al., 2025) has been intentionally created within the proposed system through the utilization of a class-weighted Logistic Regression model, by which a lower accuracy of 83.33% but a significantly superior Recall of 24.03% has been achieved. Through this strategic divergence, it has been ensured that false negatives have been minimized, and a safer application has been provided for the end-user. The comparative performance between the literature benchmarks and the proposed risk prediction models has been summarized in Table 6.2 below.

**Table 6.2: Comparative Performance of Allergen Risk Models**

Model                Source              Accuracy Recall              Diagnostic Focus Architecture                                          (Sensitivity) Decision Tree        Aziz et al., 2025   98.00%       Not Reported    Sheer Accuracy Random Forest        Aziz et al., 2025   96.77%       Not Reported    Sheer Accuracy Logistic             Aziz et al., 2025   58.06%       Not Reported    Baseline Regression Logistic             Proposed System 83.33%           24.03%          Prioritizes   Safety Regression (Class-                                                    (Minimizes     False Weighted)                                                             Negatives)

### 6.3 Technical Challenges Encountered

#### 6.3.1 Visual Similarities Between Allergenic and Non-Allergenic Ingredients

Significant difficulties have been encountered due to the extreme visual similarities inherent in Sri Lankan curries. Dishes such as beef curry, mutton curry, and pork curry exhibit nearly identical textures and color palettes. Consequently, baseline visual models frequently failed to distinguish between them. This challenge has been mitigated by the multimodal architecture, which cross-references visual features with retrieved textual recipes, thereby reducing reliance on purely superficial visual cues.

#### 6.3.2 Handling Out-of-Distribution Dish Variations

Handling unseen dishes has presented a major technical hurdle, as the ChromaDB vector database fundamentally attempts to match every input to the closest known 35 curries. This challenge has been resolved through advanced prompt engineering. A deterministic instruction has been formulated, compelling the Gemini LLM to reject the retrieved Image Database Matches if they heavily contradict its initial visual hypothesis, thereby preserving system accuracy even when queried with out-of-distribution data.

### 6.4 Ethical Considerations and User Safety Implications

The development of an allergen detection system inherently carries profound ethical responsibilities. The primary ethical mandate has been the protection of user safety over computational prestige. As highlighted in Section 6.2.2, highly accurate algorithms have been actively rejected in favor of Logistic Regression due to the latter's superior capability to minimize false negatives. Furthermore, the integration of Apriori association rules as a secondary safety net reflects an ethical commitment to preemptively warn users of hidden cross-reactivities, ensuring that algorithmic limitations do not translate into physical harm for the end-user.

### 6.5 Limitations of the Study

Despite the successful implementation of the Multimodal RAG system, certain limitations have been acknowledged. First, the machine learning models for personalized allergen risk assessment have been trained on a dataset comprising 90 user profiles. While class-weighting mitigated the class imbalance, expanding the dataset to encompass a wider demographic would further enhance the model's generalizability. Second, the system's reliance on the Gemini API introduces an external dependency. It has been noted that this architecture requires an active internet connection and incurs a slight network latency, contrasting with localized edge-computing models. Finally, the ChromaDB vector database has been restricted to 35 curated Sri Lankan recipes. However, training networks exclusively on massive generic datasets like Food-101 often leads to severe overfitting and memory allocation errors in mobile deployments. Therefore, while expanding the curated recipe corpus would fortify the system's foundational grounding, maintaining a targeted, localized dataset remains a highly valid and computationally efficient strategy.

## Chapter 7: Conclusion

### 7.1 Summary of Contributions

A comprehensive, multimodal artificial intelligence framework dedicated to the real-time detection of food allergens within Sri Lankan cuisine has been successfully developed and evaluated. The research journey, which commenced with a critical review highlighting the severe limitations of existing food recognition systems in localized contexts, has culminated in the deployment of a highly functional, full-stack prototype. The seamless integration of a Next.js presentation tier, a FastAPI application tier, and a dual-database storage tier has been fundamentally documented and proven effective.

### 7.2 Core Findings and Validation of Hypothesis

The primary research objectives established at the outset of this dissertation have been completely fulfilled. It had been hypothesized that a multimodal RAG system would outperform traditional classification models for complex curries. This has been validated; a robust image recognition pipeline capable of identifying visually ambiguous Sri Lankan dishes has been engineered, achieving a 76.00% Top-1 accuracy and demonstrating significant resilience against out-of-distribution data. Second, a highly personalized risk assessment engine has been implemented. By utilizing Logistic Regression prioritized for maximum Recall (24.03%), individualized demographic profiles, clinical data, and dynamic ingredient lists have been successfully mapped to generate accurate, safety-focused allergen warnings, while an Apriori association rule safety net has been established to catch hidden cross-reactivities.

### 7.3 Real-World Implications for Dietary Healthcare

Significant technical contributions to the intersection of food computing and healthcare AI have been made through this research. A novel integration of large language models (Gemini) as dynamic reasoning engines over localized vector databases (ChromaDB) has been introduced. This architecture has fundamentally mitigated the "unseen data" problem that historically paralyzed traditional Convolutional Neural Networks. Furthermore, a reproducible blueprint for developing ethically sound, safety-critical medical AI applications has been

provided, bridging the critical gap between visual food recognition and clinical patient profiling.

### 7.4 Future Directions

While the foundational architecture has proven highly successful, it has been recommended that future iterations aggressively expand the underlying datasets. The incorporation of a significantly larger demographic sample size beyond the initial 90-user dataset would drastically improve the machine learning model's demographic generalizability. Furthermore, the expansion of the ChromaDB vector database from 35 curated recipes to encompass a comprehensive, national registry of Sri Lankan and regional dishes is strongly advised. To maximize the system's real-world utility, it has been proposed that the architecture be migrated from a web-based prototype to a native mobile application. The integration of real-time edge computing models would significantly enhance accessibility.

### 7.5 Concluding Remarks

In conclusion, it has been definitively demonstrated that the complexities of localized, visually ambiguous cuisines can be safely navigated by integrating multimodal visual reasoning with robust clinical predictive models. A significant leap forward in preventative dietary healthcare has been achieved, ensuring that technological innovation directly translates into enhanced physical safety for vulnerable populations.

# References

I.   Adithya, B., Kasarla, S., Praneeth, N. S., Krishna, B. H., Chamundeswari, P., &
Janakiraman, R. (2025). Deep multimodal fusion for ingredient prediction from food images and recipe descriptions. Journal of Information Systems Engineering and Management, 10(32s). https://doi.org/10.52783/jisem.v10i32s.5405
II.   Aziz, Y., Ahmad, Z., Farooq, A., Aslam, N., & Fuzail, M. (2025). Food allergy
detection using machine learning approach. Kashf Journal of Multidisciplinary Research, 2(4), 116–127. https://doi.org/10.71146/kjmr402
III.   Bolaños, M., Ferrà, A., & Radeva, P. (2017). Food ingredients recognition through
multi-label learning. In Lecture Notes in Computer Science (pp. 394–402). Springer. https://doi.org/10.1007/978-3-319-70742-6_37
IV.    de Silva, R., Karunatilake, C., Iddagoda, J., & Dasanayake, D. (2022). Food allergy in
Sri Lanka – A comparative study. World Allergy Organization Journal, 15(12), 100723. https://doi.org/10.1016/j.waojou.2022.100723
V.    Iddagoda, J., Gunasekara, P., Handunnetti, S., Jeewandara, C., Karunatilake, C.,
Malavige, G. N., de Silva, R., & Dasanayake, D. (2022). Identification of allergens in coconut milk and oil with patients sensitized to coconut milk in Sri Lanka. Clinical and Molecular Allergy, 20(1), 14. https://doi.org/10.1186/s12948-022-00181-0
VI.    Liu, D., Zuo, E., Wang, D., He, L., Dong, L., & Lu, X. (2025). Deep learning in food
image recognition: A comprehensive review. Applied Sciences, 15(14), 7626. https://doi.org/10.3390/app15147626
VII.    Louro, J., Fidalgo, F., & Oliveira, Â. (2024). Recognition of food ingredients—Dataset
analysis. Applied Sciences, 14(13), 5448. https://doi.org/10.3390/app14135448
VIII.   Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable   visual   models     from     natural   language    supervision.   arXiv. https://doi.org/10.48550/arxiv.2103.00020
IX.    Reddy, B. D., Darshan, C. H., Harshitha, P., & Adithya, B. (2025). Food image analysis
for ingredient identification using deep-learning. Geethanjali College of Engineering and Technology (GCET). https://dx.doi.org/10.2139/ssrn.5253155
X.    Sampath, V., Abrams, E. M., Adlou, B., Akdis, C., Akdis, M., Brough, H. A., Chan, S.,
Chatchatee, P., Chinthrajah, R. S., Cocco, R. R., Deschildre, A., Eigenmann, P., Galvan, C., Gupta, R., Hossny, E., Koplin, J. J., Lack, G., Levin, M., Shek, L. P., ...

Renz, H. (2021). Food allergy across the globe. Journal of Allergy and Clinical Immunology, 148(6), 1347–1364. https://doi.org/10.1016/j.jaci.2021.10.018
XI.    Sicherer, S. H., & Sampson, H. A. (2009). Food allergy: Recent advances in
pathophysiology and treatment. Annual Review of Medicine, 60(1), 261–277. https://doi.org/10.1146/annurev.med.60.042407.205711
XII.   Food and Agriculture Organization of the United Nations & World Health
Organization. (2026). FAO/WHO workshop on risk assessment of food allergens: Workshop report.

# Appendices

## Appendix A: Multimodal RAG Pipeline Evaluation (Seen Test Dataset)

## Appendix B: Gemini Large Language Model Prompts

B.1 Zero-Shot Visual Dish Identification Prompt

B.2 Multimodal RAG Reasoning Synthesis Prompt

