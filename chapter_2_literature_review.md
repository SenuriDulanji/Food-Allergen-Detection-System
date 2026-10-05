## Chapter 2: Literature Review

2.1 Introduction

The paradigm of digital health monitoring and food safety has been fundamentally altered by the intersection of computer vision, natural language processing, and medical informatics. Historically, dietary monitoring has been relied upon self-reported logs or single-modality classifiers. Both methods have been challenged by real-world complexities. The evolving landscape of food analysis has been contextualized in this chapter. Vision-language modeling, Retrieval-Augmented Generation (RAG), tabular machine learning, and association rule mining have been evaluated, and the technical foundation for a localized allergen risk assessment framework has been established.

2.2 Food Allergens and Associated Health Risks

A major global public health challenge has been presented by food allergies. These allergies have been driven by abnormal hypersensitivity reactions triggered by specific dietary proteins. It has been indicated by epidemiological data that food allergies have affected approximately 1% to 10% of the global population. Clinical manifestations have ranged from mild skin reactions to life-threatening systemic anaphylaxis (Sampath et al., 2021). The highest immediate risk has been posed by Immunoglobulin E (IgE)-mediated reactions, for which immediate emergency intervention has been required.

The pathophysiology of allergic diseases has been explained by the "epithelial barrier hypothesis." It has been posited that skin and mucosal tight junctions have been disrupted by environmental factors, and a systemic immune response has been instigated. Risk management has been completely reliant on allergen avoidance strategies. However, real-world execution has been hindered by the limitations of conventional medical diagnostics. Standard procedures, such as the Skin Prick Test (SPT) and Enzyme-Linked Immunosorbent Assays (ELISA), have been proven to be resource-intensive and impractical for real-time protection.

2.3 Culinary Characteristics and Hidden Allergens in Sri Lankan Dishes

While major Western allergens have been primarily focused upon in clinical research, a distinct allergen profile has been presented by South Asian cuisines. Sri Lankan cuisine has been characterized by its reliance on aromatic spices, legumes, and coconut bases. Consequently, a high prevalence of "hidden" allergens has been created; these allergens have not been identifiable via visual inspection.

Major variations from Western distributions have been highlighted by epidemiological reviews in Sri Lanka. Cow's Milk Allergy (CMA) has been identified as the most common childhood food allergy (31.2%), and it has been followed by primary red meat allergy (27.7%) (de Silva et al., 2022). Furthermore, a high rate of Food-Dependent Exercise-Induced Anaphylaxis (FDEIA) has been documented, with wheat having been identified as the culprit food.

Additionally, coconut milk has been utilized as a foundational ingredient in Sri Lankan cooking. Novel heat-stable allergens have been successfully identified in both fresh and boiled coconut milk (Iddagoda et al., 2022). Because these proteins have survived the cooking process, a significant threat has been posed. Immediate hypersensitivity to native foods, including pineapple, rambutan, and cuttlefish, has also been clinically documented. Thus, it has been demonstrated that a Western food recognition system has not been functional for the Sri Lankan context.

2.4 Traditional Methods vs. Automated Food and Allergen Detection

Historically, allergen confirmation has been reliant on clinical diagnostic tests. These tests have been time-consuming and prone to false positives. For everyday prevention, allergen tracking has been reliant on manual techniques, such as reading ingredient labels. Within the context of street food cultures and local restaurants in Sri Lanka, standardized allergen labeling has been observed to be nonexistent. This information gap has rendered manual avoidance strategies ineffective.

The transition from manual verification to automated, AI-driven food detection has been proposed to bridge this gap. Through machine learning, manual guesswork has been replaced by immediate, localized risk assessments, and a real-time safety mechanism for vulnerable individuals has been provided.

2.5 Artificial Intelligence and Computer Vision in Food Analysis

2.5.1 Image Classification and Object Detection Models
To automate food detection, computer vision has been heavily utilized by researchers. Automated food image analysis has progressed from manual feature extraction methods to Convolutional Neural Networks (CNNs). While high categorization accuracy has been achieved by standard CNNs on controlled datasets, difficulties have been encountered due to the fine-grained nature of food images, where high intra-class diversity and low inter-class variation have been exhibited.

To address real-world dining scenarios, efficiency and depth have been prioritized in modern architectures. The effectiveness of hybrid models has been demonstrated for complex ingredient identification. By combining local feature extraction with transformers, overlapping ingredients and lighting variations have been effectively handled, and the approach has been made suitable for mobile deployment.

2.5.2 Multimodal Machine Learning Approaches
To overcome the limitations of single-modality vision systems, research has been shifted toward multimodal learning. Standard CNN backbones have been adapted by replacing the final layer with a sigmoid activation function. Through this adaptation, systems have been trained to predict multiple ingredients simultaneously, and visually 'invisible' components have been inferred via global visual context.

This capability has been expanded by fusing visual data with text. It has been proven that image categorization accuracy has been significantly boosted through the integration of textual cues. Modern iterations have been made highly sophisticated. By combining CNN backbones for visual feature extraction with Transformer encoders for textual processing, the importance of images versus recipe descriptions has been dynamically weighed by the system. Ambiguous inputs have been handled by this multimodal fusion, and high classification accuracies have been achieved.

2.6 Integration of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG)

While natural language has been excellently processed by Large Language Models (LLMs), reliance on them for medical advice has been considered dangerous due to their tendency to hallucinate facts. In the context of allergen detection, a fatal outcome could have been caused by an incorrect prediction.

To ensure safety and factual grounding, Retrieval-Augmented Generation (RAG) frameworks have been increasingly utilized. Within the proposed architecture, visual features have been extracted via CLIP embeddings, while textual recipe metadata has been vectorized using semantic text embeddings. Instead of relying on parametric guesswork, vector databases (e.g., ChromaDB) have been queried to retrieve verified Sri Lankan recipes. Subsequently, this retrieved factual context has been synthesized by a Large Language Model (e.g., Gemini). Thus, it has been guaranteed that the final allergen warning has been firmly grounded in a localized knowledge base.

2.7 Association Rule Mining and Cross-Reactivity in Food Allergies

The complex nature of food allergies has been frequently characterized by hidden cross-reactivities, wherein an immune response to one allergen has been triggered by a structurally similar protein in an unrelated food. Historically, these rare co-occurrences have been missed by standard machine learning models due to the inherent imbalance of clinical datasets. Consequently, association rule mining has been explored as a methodology for uncovering these hidden relationships. Specifically, the Apriori algorithm has been utilized in medical informatics to extract deterministic rules from sparse transactional data. Rare cross-reactivities have been algorithmically isolated without the need for synthetic data distortion. It has been demonstrated that when association rules have been combined with predictive models, a robust, hybrid safety net has been established for personalized dietary healthcare.

2.8 Details of Existing Food Recognition and Risk Prediction Systems

2.8.1 Convolutional Neural Network (CNN) Systems
Standard image sets, including the Food-101 corpus and the UEC-Food100 database, have been widely utilized to construct baseline convolutional neural network frameworks (Bolaños et al., 2017; Liu et al., 2025). Specialized datasets such as the Ingredients101 collection have been deployed to train multi-label networks (Bolaños et al., 2017). While a test F1-score of 80.06% has been produced on the Ingredients101 dataset, an evaluation on unseen data from the Recipes5k corpus resulted in a significantly reduced test F1-score of 20.66% (Bolaños et al., 2017). An improved classification accuracy of 96.88% has been yielded on the FOOD-11 dataset through the deployment of transfer learning ensembles (Liu et al., 2025). Furthermore, a macro-F1 score of 97.59% has been recorded on the Vireo Food-251 dataset through a multi-task, region-wise deep convolutional framework (Reddy et al., 2025). However, structural visual limitations have been introduced by coarse dataset annotations, whereby Western cuisines have been heavily overrepresented while African and South Asian dishes have been restricted (Liu et al., 2025).

2.8.2 Vision Transformer (ViT) and Attention-Based Systems
Large-scale benchmarks, including the ISIA Food-500 dataset, have been deployed to assess attention-augmented architectures (Liu et al., 2025). A state-of-the-art fine-grained recognition accuracy of 96.40% has been reached on the Food-101 dataset by the FRCNNSAM framework, which integrates self-attention mechanisms with deep convolutional networks (Liu et al., 2025). However, when the CBiAFormer model has been evaluated against unseen, cross-cultural data within the ISIA Food-200 dataset, a prominent performance drop to a Top-1 accuracy of 72.92% has been observed (Liu et al., 2025). Prohibitive execution bottlenecks have been generated by high multi-head self-attention computational complexities, by which mobile and edge deployments have been severely restricted (Reddy et al., 2025; Liu et al., 2025).

2.8.3 Multimodal Fusion Systems
Heterogeneous multimodal databases combining visual feature layers with corresponding textual documentation have been leveraged to align cross-modal network features (Adithya et al., 2025). Recognition accuracies ranging from 92% to 94% have been attained by advanced multimodal architectures in which convolutional networks have been fused with transformer language models (Adithya et al., 2025). An F1-score of 80.06% and an Intersection over Union (IoU) of 67.9% have been established by attention-driven fusion models (Adithya et al., 2025). However, significant architectural complexities have been introduced during cross-modal semantic feature alignment (Adithya et al., 2025; Liu et al., 2025).

2.8.4 Machine Learning Classifiers for Allergen Risk Prediction
Various electronic health record (EHR) screening algorithms have been evaluated to predict potential individual allergen exposure risks (Ahmad et al., 2025). A maximum diagnostic accuracy rate of 98.00% has been achieved by Decision Tree (DT) models during the classification of patient food allergy registries (Ahmad et al., 2025). Predictive accuracies of 96.77% for Random Forest and 95.16% for K-Nearest Neighbors (KNN) have also been recorded (Ahmad et al., 2025). Conversely, suboptimal classification results have been produced by Logistic Regression models, which have been limited to low accuracies of 58.06% (Ahmad et al., 2025).

2.9 Critical Limitations of Existing Systems Regarding Patient Safety

A critical evaluation of modern machine learning applications has revealed several limitations regarding patient safety:

1. Focus on Calories over Safety: Most food recognition systems have been designed for calorie tracking, and allergen detection has often been treated as an afterthought.
2. Cultural Data Bias: Mainstream AI models have been trained on predominantly Western and East Asian datasets. A severe lack of representation for South Asian cuisine has been observed, and poor generalization has occurred when Sri Lankan dishes have been evaluated.
3. The Clinical Disconnect: The composition of a plate has been evaluated by computer vision models, but the user's clinical profile has been ignored. Conversely, patient health records have been utilized by machine learning models to predict allergy risks, but real-time visual input and cross-reactive safety nets have been omitted from these tabular systems.

2.9 Summary of Literature and Identified Research Gaps

In summary, while the components for a robust allergen detection system have been identified in the literature, they have not been integrated to serve the Sri Lankan context. Four major research gaps have been identified:

- Gap 1: Lack of Localized Datasets. An absence of multimodal datasets capturing complex, hidden allergens unique to Sri Lankan cuisine has been observed.
- Gap 2: Absence of Multimodal Dish Identification. The integration of textual data and computer vision into a cohesive multimodal dish identification framework has been neglected. The synergistic potential of fusing visual features with textual descriptions has not been realized.
- Gap 3: Unimodal Clinical vs. Visual Silos. A unified system by which the gap between the evaluated food and the consuming individual is bridged has not been established.
- Gap 4: Imbalanced Demographic Profiling and Cross-Reactivity Blind Spots. Clinical user allergy data has been inherently imbalanced. Consequently, hidden cross-reactivities have been missed by standalone predictive models. The development of models by which these minority classes are balanced without relying on synthetic data distortion has remained a challenge. The application of association rule mining alongside standard probabilistic models (e.g., Logistic Regression) has remained underexplored.

It has been aimed to bridge these gaps through this research. A dual-pipeline multimodal RAG framework has been developed to accurately identify Sri Lankan dishes through combined textual and visual retrievals. By combining this framework with a hybrid detection engine powered by Logistic Regression and an Apriori-based cross-reactivity safety net, a holistic, personalized, and culturally accurate allergen risk engine has been proposed.
