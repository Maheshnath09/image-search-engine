"""
Enhanced FAISS Database Manager
Handles multiple indices (local, COCO, combined) and result fusion
"""
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import logging

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_index(embeddings: np.ndarray, metadata: list, 
               index_path: Path, metadata_path: Path):
    """
    Save FAISS index and metadata to disk
    
    Args:
        embeddings: numpy array of shape (n_images, embedding_dim)
        metadata: list of dicts with image information
        index_path: path to save FAISS index
        metadata_path: path to save metadata
    """
    # Create directory if needed
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"✅ Saved index with {index.ntotal} vectors to {index_path}")


def load_index(index_path: Path, metadata_path: Path) -> Tuple[faiss.Index, list]:
    """
    Load FAISS index and metadata from disk
    
    Returns:
        tuple: (faiss_index, metadata_list)
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"✅ Loaded index with {index.ntotal} vectors from {index_path}")
    return index, metadata


def search_embeddings(query_vector: np.ndarray, index: faiss.Index, 
                     metadata: list, k: int = 10) -> List[Dict]:
    """
    Search for similar images using query vector
    
    Args:
        query_vector: numpy array of shape (embedding_dim,)
        index: FAISS index
        metadata: list of image metadata
        k: number of results to return
    
    Returns:
        list: list of dicts with search results
    """
    # Ensure query vector is 2D and normalized
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    query_vector = query_vector.astype('float32')
    faiss.normalize_L2(query_vector)
    
    # Search
    k = min(k, index.ntotal)
    distances, indices = index.search(query_vector, k)
    
    # Format results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['score'] = float(distance)
            result['rank'] = i + 1
            results.append(result)
    
    return results


def create_combined_index(embeddings_list: List[np.ndarray], 
                         metadata_list: List[list],
                         combined_index_path: Path,
                         combined_metadata_path: Path):
    """
    Combine multiple indices into one
    
    Args:
        embeddings_list: list of embedding arrays
        metadata_list: list of metadata lists
        combined_index_path: path to save combined index
        combined_metadata_path: path to save combined metadata
    """
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings_list)
    
    # Combine all metadata (update IDs to be unique)
    all_metadata = []
    current_id = 0
    
    for metadata in metadata_list:
        for item in metadata:
            item_copy = item.copy()
            item_copy['original_id'] = item_copy.get('id', current_id)
            item_copy['id'] = current_id
            all_metadata.append(item_copy)
            current_id += 1
    
    # Save combined index
    save_index(all_embeddings, all_metadata, 
               combined_index_path, combined_metadata_path)
    
    logger.info(f"✅ Created combined index with {len(all_metadata)} total images")


class MultiIndexSearcher:
    """Handles searching across multiple indices with result fusion"""
    
    def __init__(self):
        self.indices = {}
        self.metadatas = {}
        self.load_available_indices()
    
    def load_available_indices(self):
        """Load all available indices"""
        # Try to load combined index first (preferred)
        if settings.COMBINED_INDEX.exists():
            try:
                idx, meta = load_index(settings.COMBINED_INDEX, 
                                      settings.COMBINED_METADATA)
                self.indices['combined'] = idx
                self.metadatas['combined'] = meta
                logger.info("✅ Loaded combined index")
                return  # Use combined index only
            except Exception as e:
                logger.warning(f"Could not load combined index: {e}")
        
        # Load individual indices
        if settings.LOCAL_INDEX.exists():
            try:
                idx, meta = load_index(settings.LOCAL_INDEX, 
                                      settings.LOCAL_METADATA)
                self.indices['local'] = idx
                self.metadatas['local'] = meta
                logger.info("✅ Loaded local index")
            except Exception as e:
                logger.warning(f"Could not load local index: {e}")
        
        if settings.COCO_INDEX.exists() and settings.ENABLE_COCO:
            try:
                idx, meta = load_index(settings.COCO_INDEX,
                                      settings.COCO_METADATA)
                self.indices['coco'] = idx
                self.metadatas['coco'] = meta
                logger.info("✅ Loaded COCO index")
            except Exception as e:
                logger.warning(f"Could not load COCO index: {e}")
        
        if not self.indices:
            raise FileNotFoundError("No indices found! Run indexer.py first.")
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               sources: Optional[List[str]] = None) -> List[Dict]:
        """
        Search across specified indices
        
        Args:
            query_vector: query embedding
            k: number of results to return
            sources: list of sources to search (None = all)
        
        Returns:
            list of results sorted by score
        """
        if 'combined' in self.indices:
            # Use combined index
            return search_embeddings(query_vector, 
                                   self.indices['combined'],
                                   self.metadatas['combined'], k)
        
        # Search multiple indices separately
        all_results = []
        sources_to_search = sources or list(self.indices.keys())
        
        for source in sources_to_search:
            if source in self.indices:
                results = search_embeddings(
                    query_vector,
                    self.indices[source],
                    self.metadatas[source],
                    k=k
                )
                all_results.extend(results)
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:k]
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded indices"""
        stats = {
            'indices_loaded': list(self.indices.keys()),
            'total_images': sum(idx.ntotal for idx in self.indices.values())
        }
        
        # Count by source
        if 'combined' not in self.indices:
            for source, meta in self.metadatas.items():
                stats[f'{source}_count'] = len(meta)
        
        return stats