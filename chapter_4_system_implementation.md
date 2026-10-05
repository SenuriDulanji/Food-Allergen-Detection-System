## Chapter 4: System Development and Implementation

4.1 Introduction

The transition from theoretical methodology to practical application has been detailed in this chapter. The development of a functional prototype capable of real-time multimodal food allergen detection has been described. The implementation of the backend architecture, frontend user interface, and the integration of artificial intelligence models have been systematically documented.

4.2 Overall System Implementation Architecture

The structural foundation of the application has been modeled as a three-tier architecture, encompassing the Presentation Tier, Application Tier, and Data Tier. The interactions between the Next.js frontend, the FastAPI backend, and the dual databases (SQLite and ChromaDB) have been mapped to ensure seamless data flow. The complex multimodal Retrieval-Augmented Generation (RAG) pipeline and the multi-layered hybrid allergen detection engine have been integrated into this central architecture.

![Overall System Implementation Architecture](system_architecture_diagram.png)

*[INSERT Figure 4.1: Overall System Architecture HERE]*

4.3 Backend Development (FastAPI, Python)

The backend infrastructure has been developed using Python, leveraging the FastAPI framework for high-performance, asynchronous Application Programming Interface (API) routing. FastAPI has been selected due to its robust support for asynchronous endpoints, automatic data validation, and built-in interactive API documentation. The backend API has been structured into modular services, ensuring that database management, machine learning inference, and large language model (LLM) communications have been independently maintained.

4.4 Frontend Interface Development (React/Next.js)

The user-facing application has been developed using the React library within the Next.js framework. A responsive, dynamic interface featuring modern glassmorphism aesthetics has been designed to facilitate seamless user interactions, from account registration to real-time dish scanning. The frontend has been configured to collect extensive user profiling data securely and to present complex allergy risk assessments via an interactive, Expandable Safety Report. This report visually contrasts detected ingredients alongside personalized machine learning risk probabilities and Apriori cross-reactivity warnings, ensuring that life-critical information is communicated in a visually intuitive format.

*[INSERT Figure 4.2: Frontend User Interface (UI) - User Registration and Profiling HERE]*

*[INSERT Figure 4.3: Frontend UI - Dish Upload Interface HERE]*

*[INSERT Figure 4.4: Frontend UI - Personalized Allergen Warning Results HERE]*

4.5 User Profiling and Personalized Risk Engine

A comprehensive user profiling system has been implemented to capture individualized demographic and clinical data. Variables such as age, gender, province, blood type, dietary patterns, work environment, and family medical history have been systematically collected via the frontend. In the backend, a risk engine has been engineered to process this data. The data has been dynamically mapped into a one-hot encoded feature vector and fed into the trained Logistic Regression models. Through this mechanism, personalized probabilistic risk scores for 24 distinct allergens have been successfully generated in real-time.

4.6 Hybrid Allergen Detection and Advanced Ingredient Resolution

To ensure maximum safety, a hybrid detection pipeline has been implemented. The probabilistic outputs of the Logistic Regression models have been fortified by deterministic safety nets generated via the Apriori algorithm. Furthermore, to prevent false positives and prepare the unstructured data for machine learning evaluation, an advanced 3-tier ingredient resolution service has been developed to process raw ingredient names extracted from the RAG pipeline:
1. **Synonym Mapping:** A prioritized dictionary has been utilized to translate complex, localized phrases into canonical terms (e.g., mapping "coconut milk" to the base allergen key "coconut"), thereby avoiding false partial matches.
2. **Exact Match:** It has been algorithmically verified if the normalized ingredient name precisely aligns with the known machine learning model keys.
3. **Substring Search:** Known allergen model keys embedded within larger ingredient phrases (e.g., identifying "prawns" inside the unstructured string "fresh jumbo prawns") have been successfully extracted.

4.7 Implementing the Multimodal RAG Service (Gemini Integration)

A Multimodal Retrieval-Augmented Generation (RAG) service has been engineered to manage the complex identification of Sri Lankan cuisine. The Google Gemini API has been integrated as the primary reasoning engine. When a user has uploaded a dish image, initial visual features have been extracted using Gemini Vision. Concurrently, vector similarity searches have been executed against a ChromaDB database using CLIP image embeddings and Gemini text embeddings. 

Crucially, an intelligent LLM reasoning prompt has been implemented to handle out-of-distribution or unseen dishes. Rather than relying on a strict mathematical confidence threshold for the image retrieval, the prompt has instructed the Gemini LLM to prioritize its own foundational knowledge over the database matches if the visual evidence has strongly contradicted the retrieved recipes. Through this approach, accurate ingredient lists have been generated even for dishes absent from the primary database of 35 curated recipes.

4.8 Experimental Environment and Reproducibility

To ensure the reproducibility of the proposed system, the experimental environment has been explicitly documented. The core backend infrastructure and machine learning pipelines have been developed and executed using Python 3.10 within a Windows operating system environment. While Google Colab was utilized to accelerate the vectorization of images and recipes via GPU hardware during the database initialization phase, the dataset was not accessed via a cloud-based drive. Instead, the runtime environment was configured to mirror the local repository structure, loading data directly from local file paths (e.g., `./backend/data/processed_recipes.json` and `./backend/data/images/`). This ensured that the exact data ingestion pipelines used during development could be seamlessly reproduced in any local environment. Furthermore, due to the highly optimized nature of the curated dataset, the final inference performance demonstrated negligible differences between Central Processing Unit (CPU) and Graphics Processing Unit (GPU) environments, ensuring the system remains accessible for standard deployment.

4.9 Database Management and Storage Solutions

A dual-database strategy has been implemented to manage the system's diverse data requirements. Relational data, including user credentials, demographic profiles, and personalized allergen settings, have been securely stored in a SQLite database using the SQLAlchemy Object-Relational Mapper (ORM). Conversely, high-dimensional vector data, encompassing the 554 dish image embeddings and the chunked recipe texts, have been stored in a persistent ChromaDB instance. 

*[INSERT Figure 4.5: Relational Database Entity-Relationship (ER) Diagram HERE]*

4.10 Summary of Implementation

A highly modular, full-stack application has been successfully implemented as a localized prototype. The integration of FastAPI, Next.js, traditional machine learning models, and advanced LLM reasoning has culminated in a robust system capable of identifying Sri Lankan dishes and calculating highly personalized allergen exposure risks. Currently, the system operates seamlessly in a localized environment, effectively preparing the architecture for future cloud deployment.
