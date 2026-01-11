"""
Configuration file for Image Search Engine
Manages settings for local data, COCO dataset, and external APIs
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Directories
    DATA_DIR: Path = Path("data/images")
    VECTOR_STORE_DIR: Path = Path("vector_store")
    COCO_DIR: Path = Path("data/coco")
    CACHE_DIR: Path = Path("data/cache")
    
    # FAISS Index files
    LOCAL_INDEX: Path = VECTOR_STORE_DIR / "faiss_index.bin"
    LOCAL_METADATA: Path = VECTOR_STORE_DIR / "images.pkl"
    COCO_INDEX: Path = VECTOR_STORE_DIR / "coco_index.bin"
    COCO_METADATA: Path = VECTOR_STORE_DIR / "coco_metadata.pkl"
    COMBINED_INDEX: Path = VECTOR_STORE_DIR / "combined_index.bin"
    COMBINED_METADATA: Path = VECTOR_STORE_DIR / "combined_metadata.pkl"
    
    # API Settings
    ENABLE_EXTERNAL_APIS: bool = True
    API_TIMEOUT: int = 10
    MAX_API_RESULTS: int = 20
    
    # Openverse API (No key required - open API)
    OPENVERSE_API_URL: str = "https://api.openverse.org/v1"
    OPENVERSE_CLIENT_ID: Optional[str] = None  # Optional for rate limit increase
    OPENVERSE_CLIENT_SECRET: Optional[str] = None
    
    # Lexica API (No key required)
    LEXICA_API_URL: str = "https://lexica.art/api/v1/search"
    
    # Unsplash API (Optional - requires key for higher rate limits)
    UNSPLASH_ACCESS_KEY: Optional[str] = None
    UNSPLASH_API_URL: str = "https://api.unsplash.com/search/photos"
    
    # Pexels API (Free key from pexels.com)
    PEXELS_API_KEY: Optional[str] = None
    PEXELS_API_URL: str = "https://api.pexels.com/v1/search"
    
    # COCO Dataset
    COCO_IMAGES_URL: str = "http://images.cocodataset.org/val2017"
    COCO_ANNOTATIONS_URL: str = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ENABLE_COCO: bool = True
    COCO_SUBSET_SIZE: Optional[int] = 5000  # Use subset for faster indexing (None = all)
    
    # Search Settings
    LOCAL_WEIGHT: float = 0.5  # Weight for local results
    COCO_WEIGHT: float = 0.3   # Weight for COCO results
    API_WEIGHT: float = 0.2    # Weight for API results
    ENABLE_RESULT_FUSION: bool = True  # Combine results from all sources
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()