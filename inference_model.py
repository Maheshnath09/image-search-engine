"""
CLIP Model Inference
Handles text and image embedding generation
"""
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import Union

# Global model instance (loaded once)
_model = None


def get_model() -> SentenceTransformer:
    """
    Get or initialize the CLIP model (singleton pattern)
    
    Returns:
        SentenceTransformer: CLIP model instance
    """
    global _model
    
    if _model is None:
        print("🔄 Loading CLIP model (clip-ViT-B-32)...")
        _model = SentenceTransformer('clip-ViT-B-32')
        print("✅ CLIP model loaded successfully")
    
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Generate embedding for text query
    
    Args:
        text: text query string
    
    Returns:
        numpy array: embedding vector
    """
    model = get_model()
    
    try:
        embedding = model.encode(text, convert_to_tensor=False)
        return np.array(embedding).astype('float32')
    except Exception as e:
        raise ValueError(f"Error embedding text: {e}")


def embed_image(image: Union[Image.Image, str]) -> np.ndarray:
    """
    Generate embedding for image
    
    Args:
        image: PIL Image object or path to image
    
    Returns:
        numpy array: embedding vector
    """
    model = get_model()
    
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL.Image or file path")
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        embedding = model.encode(image, convert_to_tensor=False)
        return np.array(embedding).astype('float32')
    
    except Exception as e:
        raise ValueError(f"Error embedding image: {e}")