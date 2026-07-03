# Table of Contents

**Abstract**  
**Acknowledgements** 
1. Background & Context (1-2 sentences)
What to consider: Start by setting the scene. Mention the rising concern of food allergies and the specific challenge of identifying hidden allergens in complex, mixed-ingredient cuisines like Sri Lankan food.
Example focus: "Food allergies pose significant health risks, and identifying allergens in complex, heavily spiced Sri Lankan cuisine remains a challenge for both locals and tourists."
2. The Problem Statement (1-2 sentences)
What to consider: What is the exact gap in the current technology that you are trying to solve?
Example focus: "Existing food recognition systems often fail to accurately deconstruct mixed Sri Lankan dishes or lack multimodal reasoning to cross-reference visual data with specific allergenic ingredients."
3. Methodology & Approach (2-3 sentences)
What to consider: How did you build it? Briefly mention the core technologies and techniques you used. Be sure to highlight the unique AI aspects.
Example focus: "This research proposes a multimodal AI framework utilizing computer vision and a Retrieval-Augmented Generation (RAG) pipeline. The system leverages image embeddings (CLIP), an XGBoost risk classification engine balanced with SMOTE, and Large Language Models (Gemini) to analyze dish images against a vector database of Sri Lankan recipes."
4. Key Results & Findings (2-3 sentences)
What to consider: What did you achieve? Provide specific metrics. This is arguably the most important part to prove your system works.
Example focus: State the accuracy, precision, and recall of your ML model. Mention how well the RAG pipeline performed (e.g., Top-1 and Top-3 retrieval accuracy) and comment on the system's ability to accurately flag specific allergens (like nuts, seafood, or dairy) in test dishes.
5. Conclusion & Significance (1-2 sentences)
What to consider: What is the ultimate impact of your work? Why does it matter?
Example focus: "The proposed system successfully provides real-time, accurate allergen warnings, ultimately enhancing food safety and dietary management for individuals consuming Sri Lankan cuisine." 
**Declaration**  
**Table of Contents**  
**List of Figures**  
**List of Tables**  
**List of Abbreviations**  

## Chapter 1: Introduction
1.1 Background to the Study  
&nbsp;&nbsp;&nbsp;&nbsp;*1.1.1 The Prevalence of Food Allergies*  
&nbsp;&nbsp;&nbsp;&nbsp;*1.1.2 The Complexity of Sri Lankan Cuisine*  
&nbsp;&nbsp;&nbsp;&nbsp;*1.1.3 AI and Image Analysis in the Culinary Domain*  
1.2 Problem Statement  
1.3 Aims and Objectives of the Research  
1.4 Significance of the Study  
1.5 Scope and Limitations  
1.6 Structure of the Dissertation  

## Chapter 2: Literature Review
2.1 Introduction  
2.2 Food Allergens and Associated Health Risks  
2.3 Culinary Characteristics and Hidden Allergens in Sri Lankan Dishes  
2.4 Traditional Methods vs. Automated Food and Allergen Detection  
2.5 Artificial Intelligence and Computer Vision in Food Analysis  
&nbsp;&nbsp;&nbsp;&nbsp;*2.5.1 Image Classification and Object Detection Models*  
&nbsp;&nbsp;&nbsp;&nbsp;*2.5.2 Multimodal Machine Learning Approaches*  
2.6 Integration of Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG)  
2.7 Existing Allergen Detection Systems and Their Limitations  
2.8 Summary of Literature and Identified Research Gaps  

## Chapter 3: Methodology
3.1 Introduction  
3.2 Overall System Architecture  
3.3 Data Collection and Dataset Preparation  
&nbsp;&nbsp;&nbsp;&nbsp;*3.3.1 Image Acquisition of Sri Lankan Dishes*  
&nbsp;&nbsp;&nbsp;&nbsp;*3.3.2 Textual Data Gathering (Recipes, Ingredients, and Allergen Profiles)*  
3.4 Data Preprocessing and Feature Engineering  
&nbsp;&nbsp;&nbsp;&nbsp;*3.4.1 Handling Imbalanced Dietary Data (e.g., SMOTE)*  
&nbsp;&nbsp;&nbsp;&nbsp;*3.4.2 Categorical Encoding and Standardization*  
3.5 Machine Learning Pipeline for Allergen Risk Assessment  
&nbsp;&nbsp;&nbsp;&nbsp;*3.5.1 Model Selection (e.g., XGBoost, MultiOutputClassifier)*  
3.6 Multimodal Retrieval-Augmented Generation (RAG) System Design  
&nbsp;&nbsp;&nbsp;&nbsp;*3.6.1 Image Embeddings (e.g., CLIP)*  
&nbsp;&nbsp;&nbsp;&nbsp;*3.6.2 Vector Database Configuration (e.g., ChromaDB)*  
3.7 Evaluation Metrics and Validation Strategies  

## Chapter 4: System Development and Implementation
4.1 Introduction  
4.2 Backend Development (FastAPI, Python)  
4.3 Frontend Interface Development (React/Next.js)  
4.4 User Profiling and Personalized Risk Engine  
4.5 Implementing the Multimodal RAG Service (Gemini Integration)  
4.6 Database Management and Storage Solutions  
4.7 Summary of Implementation  

## Chapter 5: Results and Evaluation
5.1 Introduction  
5.2 Machine Learning Model Performance Metrics  
&nbsp;&nbsp;&nbsp;&nbsp;*5.2.1 Accuracy, Precision, Recall, and F1-Score for Allergen Classification*  
5.3 Evaluation of the Multimodal RAG Pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;*5.3.1 Retrieval Accuracy (Top-1 and Top-3 Metrics)*  
&nbsp;&nbsp;&nbsp;&nbsp;*5.3.2 Qualitative Analysis of AI-Generated Reasoning*  
5.4 End-to-End System Performance and Latency  
5.5 Case Studies: Analyzing Specific Sri Lankan Dishes (e.g., Kottu, Kiribath, Curries)  
5.6 Summary of Results  

## Chapter 6: Discussion
6.1 Interpretation of Key Findings  
6.2 Comparison with Existing Food Recognition Systems  
6.3 Technical Challenges Encountered  
&nbsp;&nbsp;&nbsp;&nbsp;*6.3.1 Visual Similarities Between Allergenic and Non-Allergenic Ingredients*  
&nbsp;&nbsp;&nbsp;&nbsp;*6.3.2 Handling Out-of-Distribution Dish Variations*  
6.4 Ethical Considerations and User Safety Implications  

## Chapter 7: Conclusion and Future Work
7.1 Summary of the Research Journey  
7.2 Fulfillment of Research Objectives  
7.3 Key Contributions to the Field  
7.4 Recommendations for Future Work  
&nbsp;&nbsp;&nbsp;&nbsp;*7.4.1 Expanding the Dataset and Generalizability*  
&nbsp;&nbsp;&nbsp;&nbsp;*7.4.2 Mobile Application Integration and Real-time Edge Computing*  

**References**  

**Appendices**  
Appendix A: Sample Survey Data and Encoded User Allergen Profiles  
Appendix B: Core Code Snippets (RAG Service, XGBoost Pipeline)  
Appendix C: User Interface Walkthrough and Screen Captures  
Appendix D: Detailed Evaluation Logs and Confusion Matrices  
