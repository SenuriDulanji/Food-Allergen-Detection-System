# Multimodal AI Framework for Food Allergen Detection and Personalized Risk Prediction in Sri Lankan Cuisine

An AI tool designed to identify hidden food allergens in Sri Lankan cuisine and estimate personalized allergy risks. The system operates on a dual-layered pipeline:
1. **Multimodal Allergen Detection**: Combines computer vision (CLIP image embeddings) and Large Language Models (Google Gemini) with a ChromaDB recipe database to identify dishes and resolve their ingredients in real time.
2. **Personalized Risk Assessment**: Uses a class-weighted Logistic Regression classifier alongside the Apriori association rule mining algorithm to predict exposure risks based on an individual's socio-demographic profile and other information, acting as a preventative safety net.

## Overview
This repository contains the codebase and documentation for a multimodal artificial intelligence framework designed specifically for identifying hidden food allergens and providing personalized risk predictions in complex, heavily spiced Sri Lankan cuisine.

Food allergies are a severe global public health challenge. Existing food recognition systems, primarily trained on Western-centric datasets, struggle to accurately deconstruct mixed Asian dishes and lack the multimodal reasoning required to cross-reference visual data with patient-specific allergenic profiles. This project addresses these limitations by introducing a novel, dual-layered system combining advanced computer vision, Retrieval-Augmented Generation (RAG), and personalized machine learning classifiers.

## Key Features
- **Multimodal Dish Identification**: Utilizes a Retrieval-Augmented Generation (RAG) pipeline, integrating CLIP image embeddings and Gemini Large Language Models to query a localized ChromaDB vector database of curated Sri Lankan recipes.
- **Personalized Risk Prediction**: Processes a person's socio-demographic data to power a Logistic Regression risk classification engine. This engine is algorithmically balanced using class weights to prioritize diagnostic recall, minimizing life-threatening false negatives.
- **Cross-Reactivity Detection**: Deploys the Apriori algorithm to mine hidden biological cross-reactivities, creating an additional deterministic safety net for dietary management.
- **Full-Stack Application**: Features a high-performance asynchronous FastAPI backend (Python) and a dynamic, responsive frontend built with React and Next.js, allowing for real-time dish scanning and personalized allergen warnings.

## Architecture
The system architecture follows a three-tier model (Presentation, Application, and Data Tiers) incorporating:
- **Frontend**: React / Next.js
- **Backend**: FastAPI (Python)
- **AI Models**:
  - Image Embeddings: CLIP (`openai/clip-vit-base-patch32`)
  - Text Embeddings & Synthesis: Google Gemini (`gemini-embedding-2-preview` and LLM)
  - Risk Classification: Logistic Regression (class-weighted)
  - Association Rule Mining: Apriori Algorithm
- **Databases**: SQLite (Relational Data) & ChromaDB (Vector Search for Image & Text Embeddings)

## Performance & Results
- **Retrieval Accuracy**: Achieved a 76.00% Top-1 fine-grained retrieval accuracy using the multimodal RAG system, vastly outperforming traditional Convolutional Neural Network (CNN) baselines which failed on out-of-distribution unseen data (0.00% accuracy).
- **Model Recall**: Generated a superior Recall of 24.03% by the predictive socio-demographic backend, ensuring minimization of false negatives.
- **Latency**: Provides real-time, personalized allergen warnings with an average inference latency of just 4.24 seconds.

## Dataset Highlights
- **Image Data**: Curated dataset of 554 images across 35 specific Sri Lankan dishes, including visually similar dishes (e.g., beef, pork, and mutton curries) to test system limits.
- **Textual Data**: 35 corresponding traditional recipes and socio-demographic survey data from 90 individuals.

## Project Structure
- `ml_model/`: Contains Jupyter notebooks for Exploratory Data Analysis (EDA) and machine learning pipeline training.
- `chapter_*.md`: Dissertation chapters detailing methodology, implementation, and results.

## Future Directions
Future research will focus on expanding the dataset to improve generalizability across broader demographic populations and integrating the system into mobile applications with real-time edge computing capabilities.
