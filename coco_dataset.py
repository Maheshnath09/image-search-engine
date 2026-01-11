"""
COCO Dataset Integration
Downloads and manages COCO dataset for enhanced search results
"""
import os
import json
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COCODatasetManager:
    """Manages COCO dataset download and metadata"""
    
    def __init__(self):
        self.coco_dir = settings.COCO_DIR
        self.images_dir = self.coco_dir / "val2017"
        self.annotations_file = self.coco_dir / "annotations" / "instances_val2017.json"
        self.captions_file = self.coco_dir / "annotations" / "captions_val2017.json"
        self.coco_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, desc: str = "Downloading"):
        """Download file with progress bar"""
        if destination.exists():
            logger.info(f"File already exists: {destination}")
            return
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def download_annotations(self):
        """Download COCO annotations"""
        logger.info("📥 Downloading COCO annotations...")
        
        annotations_zip = self.coco_dir / "annotations_trainval2017.zip"
        
        if not self.annotations_file.exists():
            self.download_file(
                settings.COCO_ANNOTATIONS_URL,
                annotations_zip,
                "Annotations"
            )
            
            # Extract
            logger.info("📦 Extracting annotations...")
            with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                zip_ref.extractall(self.coco_dir)
            
            # Clean up zip
            annotations_zip.unlink()
        
        logger.info("✅ Annotations ready")
    
    def download_subset_images(self, num_images: Optional[int] = None):
        """Download subset of COCO validation images"""
        logger.info("📥 Downloading COCO images subset...")
        
        # Load annotations to get image list
        if not self.annotations_file.exists():
            self.download_annotations()
        
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data['images']
        
        # Use subset if specified
        if num_images and num_images < len(images):
            images = images[:num_images]
            logger.info(f"Using subset of {num_images} images")
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        skipped = 0
        
        for img_info in tqdm(images, desc="Downloading images"):
            img_filename = img_info['file_name']
            img_path = self.images_dir / img_filename
            
            if img_path.exists():
                skipped += 1
                continue
            
            try:
                img_url = f"{settings.COCO_IMAGES_URL}/{img_filename}"
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded += 1
                
            except Exception as e:
                logger.error(f"Error downloading {img_filename}: {e}")
        
        logger.info(f"✅ Downloaded: {downloaded}, Skipped: {skipped}")
    
    def get_image_metadata(self) -> List[Dict]:
        """Get metadata for all COCO images with captions"""
        if not self.annotations_file.exists() or not self.captions_file.exists():
            raise FileNotFoundError("COCO annotations not found. Run download first.")
        
        # Load captions
        with open(self.captions_file, 'r') as f:
            captions_data = json.load(f)
        
        # Create mapping: image_id -> captions
        image_captions = {}
        for caption in captions_data['annotations']:
            img_id = caption['image_id']
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(caption['caption'])
        
        # Load images info
        with open(self.annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        metadata = []
        for img_info in annotations_data['images']:
            img_path = self.images_dir / img_info['file_name']
            
            # Only include if image exists
            if img_path.exists():
                captions = image_captions.get(img_info['id'], [])
                metadata.append({
                    'id': img_info['id'],
                    'filename': img_info['file_name'],
                    'path': str(img_path),
                    'captions': captions,
                    'caption': captions[0] if captions else "",  # Primary caption
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'source': 'coco'
                })
        
        return metadata
    
    def setup_coco_dataset(self, num_images: Optional[int] = None):
        """Complete setup of COCO dataset"""
        logger.info("🚀 Setting up COCO dataset...")
        
        # Download annotations
        self.download_annotations()
        
        # Download images
        subset_size = num_images or settings.COCO_SUBSET_SIZE
        self.download_subset_images(subset_size)
        
        logger.info("✅ COCO dataset setup complete")


def download_coco_dataset():
    """Convenience function to download COCO dataset"""
    manager = COCODatasetManager()
    manager.setup_coco_dataset(settings.COCO_SUBSET_SIZE)


if __name__ == "__main__":
    # Run this to download COCO dataset
    download_coco_dataset()