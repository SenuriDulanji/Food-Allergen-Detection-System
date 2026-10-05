## Chapter 1: Introduction

1.1 Background to the Study

Food safety has been recognized as a major public health priority. The rising rate of food allergies has been identified as a critical challenge. A heavy burden has been placed on individuals and healthcare systems, whereby smarter methods to detect allergens and communicate risks have been required. Simultaneously, unprecedented opportunities to solve these problems have been offered by advancements in artificial intelligence (AI) and computer vision. This research has been situated at this intersection. An AI-driven system has been proposed, wherein image analysis has been used to detect potential food allergens, and the complex landscape of Sri Lankan cuisine has been specifically focused upon.

1.1.1 The Prevalence of Food Allergies
Food allergy has been defined as an abnormal immune response to specific food proteins. For sensitized individuals, reactions ranging from mild rashes to life-threatening anaphylaxis have been triggered by minor accidental exposures (Sicherer & Sampson, 2009). The global burden has been observed to be massive and growing; hundreds of millions of people worldwide have been affected.

The daily management of these allergies has been proven to be difficult. The identification of safe food has been recognized as a major challenge, especially when dining out where ingredient lists have not been made available. Since most severe allergic reactions have been recorded outside the home, an urgent need has been created for accessible tools by which the allergen risk of unfamiliar dishes can be evaluated in real time.

1.1.2 The Complexity of Sri Lankan Cuisine
Sri Lankan food has been celebrated for its bold spices, rich textures, and diverse influences. However, significant challenges have been presented by this culinary complexity for individuals managing food allergies. The greatest risk has been posed by 'hidden' allergens—ingredients crucial for flavor but invisible to the eye. For example, Maldive fish (umbalakada) has been used as a hidden flavor enhancer in many dishes. Cashews have often been blended into festive dishes like Kiribath, and coconut milk has been utilized as the base of most local curries.

Furthermore, similar appearances have been exhibited by distinct Sri Lankan dishes. Rich, dark sauces might be displayed by both prawn and chicken curries, yet completely different allergen profiles have been carried by them. Traditional visual inspection has been proven to be insufficient, especially for individuals by whom the local cuisine has not been familiarized. Without standardized allergen labeling, technology has been identified as the optimal method by which this information gap can be closed.

1.1.3 AI and Image Analysis in the Culinary Domain
Over the last decade, food analysis has been transformed by AI. Early systems have been reliant upon Convolutional Neural Networks (CNNs) to classify dishes from images. However, these systems have been primarily trained on Western or East Asian foods, and South Asian cuisines have been largely ignored.

Recently, the landscape has been advanced by Vision-Language Models (VLMs). The understanding of both images and text in a shared space has been enabled by models like CLIP, by which nuanced reasoning has been facilitated (Radford et al., 2021). For allergen detection, simple dish identification has been deemed insufficient; visual classification has been required to be cross-referenced with known ingredient profiles.

Therefore, Retrieval-Augmented Generation (RAG) has been introduced. By linking an image query to a verified knowledge base, factual responses have been generated. Building on these advancements, a novel system tailored for Sri Lankan food has been proposed. By combining image embeddings, a recipe vector database, and personalized machine learning classifiers, accurate, real-time allergen warnings have been generated, and food safety has been improved.

1.2 Problem Statement

Despite global awareness of food allergies, a significant challenge has been faced by individuals consuming Sri Lankan cuisine: the inability to reliably identify allergens in visually complex dishes. Common allergens have frequently been concealed within curries that cannot be distinguished by appearance alone.

Existing food recognition systems have predominantly been designed for Western or East Asian cuisines. The cultural specificity required to interpret Sri Lankan dishes has not been possessed by these systems. Furthermore, they have typically been operated at the dish identification level, and ingredient-level allergen assessments tailored to individual dietary profiles have not been provided.

Therefore, two core gaps have been identified by this research:
1. An AI-based food recognition system specifically trained and optimized for Sri Lankan cuisine has not been developed.
2. A personalized allergen risk assessment mechanism, by which dish ingredients are cross-referenced with a user's specific allergen profile in real time, has not been established.

A direct health risk has been presented by these gaps. The addressing of both gaps through a multimodal AI system, in which image analysis, a Retrieval-Augmented Generation (RAG) pipeline, and a personalized risk classification engine have been combined, has been proposed by this research.

1.3 Aims and Objectives of the Research

Aim
The design, development, and evaluation of an AI-based food allergen detection system has been established as the primary aim of this research. Through this system, Sri Lankan dishes can be identified, and real-time, personalized allergen risk assessments can be provided.

Objectives
To achieve this aim, the following objectives have been established:
1. A domain-specific dataset of Sri Lankan dish images, recipes, and ingredient-allergen mappings has been curated and prepared.
2. A multimodal dish identification pipeline has been developed by integrating CLIP image embeddings with a ChromaDB vector database and a Gemini Large Language Model (LLM) within a Retrieval-Augmented Generation (RAG) framework.
3. A personalized allergen risk classification model has been designed and trained using Logistic Regression balanced with class weights, by which individual allergen exposure risks have been predicted.
4. A full-stack application, comprising a FastAPI backend and a React frontend, has been built to integrate the AI models and deliver allergen warnings.
5. The system's performance has been evaluated using standard metrics, including accuracy, precision, recall, F1-score, and RAG retrieval accuracy (Top-1 and Top-3).

1.4 Significance of the Study

Significant practical value for public health and academic value in applied artificial intelligence have been established by this research.

Practically, a safety net for individuals managing food allergies in Sri Lanka has been provided. A real-time, preventative tool against accidental allergen exposure has been created for tourists and locals. By offering immediate, personalized risk assessments based on a photograph, users have been empowered to make informed choices, and the incidence of severe allergic reactions has been reduced. Furthermore, a structured method to communicate allergen information has been offered to the hospitality industry.

Academically, the domain of multimodal food computing has been advanced. South Asian cuisines have been kept underrepresented in AI literature. By utilizing a Retrieval-Augmented Generation (RAG) pipeline combined with Vision-Language Models (CLIP and Gemini) for Sri Lankan dishes, it has been demonstrated how AI techniques can be adapted to culturally specific health challenges. Additionally, a methodological framework for personalized risk assessment has been provided by integrating Logistic Regression with class-weighted algorithms to handle imbalanced allergen profile data.

1.5 Scope and Limitations

Scope
- Target Cuisine and Dataset: The system's knowledge base has been built for 35 specific Sri Lankan dishes, supported by a curated dataset of 554 images and 35 traditional recipes. However, quantitative evaluation of the recognition pipeline has been focused on a representative test set of 10 dishes (50 images) to assess in-distribution accuracy, alongside an evaluation of 3 unseen dishes to test out-of-distribution robustness.
- Demographic Profile: User data collected to train the Logistic Regression risk classification model has been predominantly represented by university students aged between 20 and 25 years.
- Allergen Tracking: The identification of major food allergens has been integrated into the system; peanuts, tree nuts, dairy, egg, fish, and shellfish have been focused upon.
- Technological Framework: The core architecture has been restricted to a Retrieval-Augmented Generation (RAG) approach, in which CLIP image embeddings, ChromaDB, and the Gemini Large Language Model have been utilized. Risk classification has been based on a Logistic Regression model optimized for imbalanced datasets via class weights.

Limitations
- Generalizability of Demographic Data: Because the model has been trained primarily on data from young adults, its predictive accuracy might not have been generalized to other age groups or geographic populations.
- Dish Recognition Boundaries: The system has been trained on 35 specific dishes. If an image of an out-of-distribution dish has been inputted, it might have been misidentified. Furthermore, difficulties have been encountered by the system when visually similar dishes, such as beef, pork, and mutton curries, have been processed.
- Visual Dependency: Visual features captured in an image have been primarily relied upon. If an allergenic ingredient has been pureed or visually obscured, the risk might not have been detected.
- Environmental Factors: The accuracy of the image analysis has been heavily dependent upon the photograph's quality. Predictive capabilities could have been negatively impacted by poor lighting or blurry images.

1.6 Structure of the Dissertation

The dissertation has been organized into several chapters. A comprehensive literature review of existing allergen detection systems and AI methodologies has been presented in Chapter 2. The system architecture, data collection, and algorithms have been detailed in Chapter 3. The backend and frontend development processes have been discussed in Chapter 4. The evaluation metrics and system performance have been analyzed in Chapter 5. Finally, the findings have been discussed in Chapter 6, and the research conclusions and future work have been summarized in Chapter 7.
