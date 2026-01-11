"""
Enhanced Image Indexer - Builds FAISS indices for local, COCO, and combined datasets
Run this to create vector databases for all image sources
"""
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from inference_model import get_model
from database import save_index, create_combined_index
from config import settings
from coco_dataset import COCODatasetManager

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def load_images_from_directory(directory: Path):
    """Load all images from directory"""
    image_paths = []
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist!")
    
    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_FORMATS:
            image_paths.append(file_path)
    
    return sorted(image_paths)


def build_local_index():
    """Build index for local images"""
    print("\n" + "="*60)
    print("🔷 BUILDING LOCAL IMAGE INDEX")
    print("="*60)
    
    # Create vector store directory
    settings.VECTOR_STORE_DIR.mkdir(exist_ok=True)
    
    # Load model
    print("📦 Loading CLIP model...")
    model = get_model()
    
    # Load all image paths
    print(f"📂 Scanning directory: {settings.DATA_DIR}")
    image_paths = load_images_from_directory(settings.DATA_DIR)
    
    if not image_paths:
        print(f"⚠️  No images found in {settings.DATA_DIR}")
        return None, None
    
    print(f"✅ Found {len(image_paths)} images")
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embeddings = []
    image_metadata = []
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing local images")):
        try:
            img = Image.open(img_path).convert('RGB')
            embedding = model.encode(img, convert_to_tensor=False)
            
            embeddings.append(embedding)
            image_metadata.append({
                'id': idx,
                'filename': img_path.name,
                'path': str(img_path),
                'source': 'local'
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    if not embeddings:
        print("❌ No embeddings generated!")
        return None, None
    
    embeddings_array = np.array(embeddings).astype('float32')
    print(f"📊 Embeddings shape: {embeddings_array.shape}")
    
    # Save index
    print("💾 Saving local index...")
    save_index(embeddings_array, image_metadata, 
               settings.LOCAL_INDEX, settings.LOCAL_METADATA)
    
    print("✅ Local index built successfully")
    return embeddings_array, image_metadata


def build_coco_index():
    """Build index for COCO dataset"""
    print("\n" + "="*60)
    print("🔷 BUILDING COCO DATASET INDEX")
    print("="*60)
    
    if not settings.ENABLE_COCO:
        print("⏭️  COCO indexing disabled in config")
        return None, None
    
    # Load model
    model = get_model()
    
    # Get COCO metadata
    try:
        coco_manager = COCODatasetManager()
        
        # Check if COCO images exist
        if not coco_manager.images_dir.exists() or not list(coco_manager.images_dir.glob("*.jpg")):
            print("⚠️  COCO images not found. Downloading...")
            print("💡 This may take a while (downloading subset of COCO dataset)")
            coco_manager.setup_coco_dataset(settings.COCO_SUBSET_SIZE)
        
        print("📂 Loading COCO metadata...")
        coco_metadata = coco_manager.get_image_metadata()
        print(f"✅ Found {len(coco_metadata)} COCO images")
        
    except Exception as e:
        print(f"❌ Error loading COCO: {e}")
        return None, None
    
    # Generate embeddings
    print("🔄 Generating COCO embeddings...")
    embeddings = []
    valid_metadata = []
    
    for idx, meta in enumerate(tqdm(coco_metadata, desc="Processing COCO images")):
        try:
            img = Image.open(meta['path']).convert('RGB')
            embedding = model.encode(img, convert_to_tensor=False)
            
            embeddings.append(embedding)
            meta['id'] = idx  # Update ID for COCO index
            valid_metadata.append(meta)
            
        except Exception as e:
            print(f"⚠️  Error processing {meta['filename']}: {e}")
            continue
    
    if not embeddings:
        print("❌ No COCO embeddings generated!")
        return None, None
    
    embeddings_array = np.array(embeddings).astype('float32')
    print(f"📊 COCO Embeddings shape: {embeddings_array.shape}")
    
    # Save COCO index
    print("💾 Saving COCO index...")
    save_index(embeddings_array, valid_metadata,
               settings.COCO_INDEX, settings.COCO_METADATA)
    
    print("✅ COCO index built successfully")
    return embeddings_array, valid_metadata


def build_all_indices(include_coco: bool = True, combine: bool = True):
    """Build all indices and optionally combine them"""
    print("\n" + "🚀 "*30)
    print("STARTING COMPREHENSIVE IMAGE INDEXING")
    print("🚀 "*30 + "\n")
    
    # Build local index
    local_emb, local_meta = build_local_index()
    
    # Build COCO index
    coco_emb, coco_meta = None, None
    if include_coco:
        coco_emb, coco_meta = build_coco_index()
    
    # Create combined index
    if combine and local_emb is not None:
        print("\n" + "="*60)
        print("🔷 CREATING COMBINED INDEX")
        print("="*60)
        
        embeddings_list = [local_emb]
        metadata_list = [local_meta]
        
        if coco_emb is not None:
            embeddings_list.append(coco_emb)
            metadata_list.append(coco_meta)
        
        create_combined_index(
            embeddings_list,
            metadata_list,
            settings.COMBINED_INDEX,
            settings.COMBINED_METADATA
        )
        
        print("✅ Combined index created successfully")
    
    # Summary
    print("\n" + "="*60)
    print("📊 INDEXING SUMMARY")
    print("="*60)
    print(f"✅ Local images: {len(local_meta) if local_meta else 0}")
    print(f"✅ COCO images: {len(coco_meta) if coco_meta else 0}")
    print(f"✅ Total indexed: {(len(local_meta) if local_meta else 0) + (len(coco_meta) if coco_meta else 0)}")
    print("="*60)
    print("\n✅ All indices built successfully!")
    print("💡 You can now start the API server with: uvicorn main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build image search indices')
    parser.add_argument('--local-only', action='store_true', 
                        help='Build only local index, skip COCO')
    parser.add_argument('--coco-only', action='store_true',
                        help='Build only COCO index, skip local')
    parser.add_argument('--no-combine', action='store_true',
                        help='Do not create combined index')
    
    args = parser.parse_args()
    
    try:
        if args.coco_only:
            build_coco_index()
        elif args.local_only:
            build_local_index()
        else:
            build_all_indices(
                include_coco=not args.local_only,
                combine=not args.no_combine
            )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise