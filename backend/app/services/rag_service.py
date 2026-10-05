"""
services/rag_service.py — Multimodal RAG pipeline for dish identification.
"""

import io
import json
import logging
from typing import Optional, Dict, Any

from google import genai
from google.genai import types as genai_types
from PIL import Image
import numpy as np
import torch

# ChromaDB
import chromadb
from chromadb.api.types import EmbeddingFunction, Images
import chromadb.utils.embedding_functions as embedding_functions

from app.core.config import settings
from app.core.database import get_chroma_client

logger = logging.getLogger(__name__)


class TransformersCLIPEmbeddingFunction(EmbeddingFunction[Images]):
    """
    Custom Chroma embedding function using Hugging Face transformers CLIP.
    Directly mirrors the Colab notebook embedding steps for perfect vector matching.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. GPU is required for this model prediction as requested.")
                
            self.device = "cuda"
            self.model.to(self.device)
            logger.info("CLIP model successfully loaded onto GPU (cuda).")
        except ImportError:
            logger.error("transformers package is required for TransformersCLIPEmbeddingFunction")
            raise
        
    def __call__(self, input: Images) -> chromadb.api.types.Embeddings:
        pil_images = [
            Image.fromarray(img.astype('uint8'), 'RGB') if isinstance(img, np.ndarray) else img 
            for img in input
        ]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Safely unpack the tensor depending on the Hugging Face Transformers version
        if hasattr(outputs, 'image_embeds'):
            image_features = outputs.image_embeds
        elif hasattr(outputs, 'pooler_output'):
            image_features = outputs.pooler_output
        else:
            image_features = outputs

        # NOTE: Vector normalization is intentionally omitted here to exactly match
        # the raw embeddings generated and stored via the Colab T4 GPU script.
        return image_features.cpu().numpy().tolist()


class RAGService:
    """
    Multimodal RAG pipeline for dish identification.
    Uses a hybrid approach combining text and image retrievals.
    """
    def __init__(self):
        logger.info("Initializing RAGService...")
        self.chroma_client = get_chroma_client()
        
        # 1. Text Collection Initialization
        # We explicitly use the same Gemini embedding model from the Colab script
        import os
        os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
        self.text_embedding_function = embedding_functions.GoogleGeminiEmbeddingFunction(
            api_key_env_var="GEMINI_API_KEY",
            model_name="gemini-embedding-2-preview" 
        )
        
        try:
            self.text_collection = self.chroma_client.get_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.text_embedding_function
            )
            logger.info("Connected to text collection: %s", settings.COLLECTION_NAME)
        except Exception as e:
            logger.error("Failed to load text collection: %s", e)
        
        # 2. Image Collection Initialization
        self.clip_embedding_function = TransformersCLIPEmbeddingFunction()
            
        try:
            self.image_collection = self.chroma_client.get_collection(
                name=settings.IMAGE_COLLECTION_NAME,
                embedding_function=self.clip_embedding_function
            )
            logger.info("Connected to image collection: %s", settings.IMAGE_COLLECTION_NAME)
        except Exception as e:
            logger.warning("Failed to load image collection with embedding function (falling back to manual): %s", e)
            try:
                self.image_collection = self.chroma_client.get_collection(
                    name=settings.IMAGE_COLLECTION_NAME
                )
                logger.info("Connected to image collection manually: %s", settings.IMAGE_COLLECTION_NAME)
            except Exception as inner_e:
                logger.error("Failed to load image collection entirely: %s", inner_e)
        
        # 3. Gemini LLM Setup for Reasoning and Vision
        self.genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.vision_model = settings.LLM_MODEL
        self.reasoning_model = settings.LLM_MODEL


    def _prepare_image_bytes(self, image_bytes: bytes) -> bytes:
        """Resize and convert image to JPEG to reduce token usage."""
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        max_dim = 1024
        if max(pil_image.size) > max_dim:
            pil_image.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Multimodal Hybrid RAG Analysis:
        1. Uses Gemini Vision for initial analysis
        2. Performs text retrieval from the text collection
        3. Performs CLIP image similarity search from image collection
        4. Combines both results using Gemini for final reasoning
        """
        try:
            processed_bytes = self._prepare_image_bytes(image_bytes)
            pil_image = Image.open(io.BytesIO(processed_bytes))
            
            # -----------------------------------------------------------------------
            # 1. Vision Initial Analysis
            # -----------------------------------------------------------------------
            vision_prompt = "Identify which Sri Lankan dish this is. Respond with ONLY the dish name in snake_case (e.g., fish_curry)."
            try:
                vision_response = self.genai_client.models.generate_content(
                    model=self.vision_model,
                    contents=[
                        genai_types.Part.from_bytes(data=processed_bytes, mime_type="image/jpeg"),
                        vision_prompt,
                    ],
                )
                vision_guess = vision_response.text.strip().lower().replace(" ", "_")
                logger.info("Vision model guessed dish: '%s'", vision_guess)
            except Exception as e:
                logger.error("Vision initial analysis failed: %s", e)
                vision_guess = "unknown"
                
            # -----------------------------------------------------------------------
            # 2. Text Retrieval 
            # -----------------------------------------------------------------------
            text_context = []
            try:
                text_results = self.text_collection.query(
                    query_texts=[vision_guess.replace("_", " ")],
                    n_results=settings.RAG_TOP_K
                )
                if text_results and text_results.get('documents') and text_results['documents'][0]:
                    for doc, meta in zip(text_results['documents'][0], text_results['metadatas'][0]):
                        text_context.append({
                            "content": doc,
                            "dish_name": meta.get("dish_name", "unknown") if meta else "unknown"
                        })
                logger.info(
                    "Text RAG retrieved %d chunks for query '%s'. Matches: %s", 
                    len(text_context), 
                    vision_guess,
                    [ctx['dish_name'] for ctx in text_context]
                )
            except Exception as e:
                logger.error("Text retrieval failed: %s", e)
                
            # -----------------------------------------------------------------------
            # 3. Image Retrieval (CLIP)
            # -----------------------------------------------------------------------
            image_context = []
            try:
                img_array = np.array(pil_image)
                embeddings = self.clip_embedding_function([img_array])
                clip_results = self.image_collection.query(
                    query_embeddings=embeddings,
                    n_results=settings.RAG_TOP_K
                )
                if clip_results and clip_results.get('metadatas') and clip_results['metadatas'][0]:
                    for meta, dist in zip(clip_results['metadatas'][0], clip_results['distances'][0]):
                        image_context.append({
                            "dish_name": meta.get("dish_name", "unknown") if meta else "unknown",
                            "distance": round(dist, 4)
                        })
                logger.info(
                    "Image RAG retrieved %d matches. Visual matches: %s", 
                    len(image_context),
                    [f"{ctx['dish_name']} (dist: {ctx['distance']})" for ctx in image_context]
                )
            except Exception as e:
                logger.error("Image retrieval failed: %s", e)
                
            # -----------------------------------------------------------------------
            # 4. Final Reasoning (Hybrid Combination)
            # -----------------------------------------------------------------------
            reasoning_prompt = f"""
            You are a Sri Lankan cuisine expert. We are trying to identify a dish from an image.
            
            Here is the evidence gathered:
            1. Initial Vision Model Guess: {vision_guess}
            
            2. Text Database Matches (based on vision guess):
            {json.dumps(text_context, indent=2)}
            
            3. Image Database Matches (based on visual similarity via CLIP):
            {json.dumps(image_context, indent=2)}
            
            CRITICAL INSTRUCTION FOR UNKNOWN DISHES:
            Our database only contains 35 specific curries. 
            If the dish in the image is NOT one of those 35 curries (like Koththu), 
            the "Image Database Matches" will be completely wrong because it is just returning the closest looking curry it has. 
            Therefore: If the visual matches contradict the initial Vision Guess, 
            heavily prioritize the Vision Guess! Do not output Pol Sambal or Potato Curry if the vision guess is Koththu.
            
            CRITICAL INSTRUCTION FOR ALIASES:
            If your vision guess is a local alias (e.g. "ala_hodi") and the database matches provide the canonical English name 
            (e.g. "potato_curry"), you MUST output the canonical English "dish_name" exactly as it appears in the database matches.
            
            Based on this hybrid context, identify the dish and list its ingredients.
            
            Respond strictly with a JSON object containing:
            - "dish_name": string (snake_case, e.g. "fish_curry", "potato_curry", or "koththu")
            - "ingredients": list of strings
            """
            
            final_response = self.genai_client.models.generate_content(
                model=self.reasoning_model,
                contents=[reasoning_prompt],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            try:
                final_result = json.loads(final_response.text)
                logger.info(
                    "Final reasoning result: '%s' (Original Vision Guess: '%s')", 
                    final_result.get('dish_name'), 
                    vision_guess
                )
                final_result["vision_guess"] = vision_guess
                
                # Combine top matches from Text RAG and Image RAG for Top-3 Evaluation
                candidates = list(dict.fromkeys(
                    [ctx["dish_name"] for ctx in text_context] +
                    [ctx["dish_name"] for ctx in image_context]
                ))
                final_result["candidates"] = candidates
                
                return final_result
            except json.JSONDecodeError as e:
                logger.error("Failed to parse reasoning output as JSON: %s. Output was: %s", e, final_response.text)
                return {"dish_name": "unknown", "ingredients": [], "vision_guess": vision_guess}
                
        except Exception as exc:
            logger.error("analyze_image failed: %s", exc, exc_info=True)
            raise


from functools import lru_cache

@lru_cache()
def get_rag_service() -> RAGService:
    """Return a singleton instance of RAGService."""
    return RAGService()