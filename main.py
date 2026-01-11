"""
Enhanced FastAPI Backend with Multi-Source Search
Integrates local images, COCO dataset, and external APIs
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # <-- NEW IMPORT
from pydantic import BaseModel
from typing import List, Optional
import io
from PIL import Image
import uvicorn
import os # <-- NEW IMPORT for path checking

from inference_model import embed_text, embed_image, get_model
from database import MultiIndexSearcher
from external_apis import AggregatedSearchClient
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Image Search Engine API",
    description="Multi-source image search with CLIP, COCO, and external APIs",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global searchers
multi_index_searcher = None
api_client = None


# Pydantic models
class TextSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    sources: Optional[List[str]] = None  # ['local', 'coco', 'openverse', 'lexica', etc.]
    include_apis: Optional[bool] = True


class ImageSearchRequest(BaseModel):
    top_k: Optional[int] = 10
    sources: Optional[List[str]] = None
    include_apis: Optional[bool] = False  # APIs don't support image search


class SearchResponse(BaseModel):
    results: List[dict]
    total: int
    sources_used: List[str]
    stats: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Load indices and initialize API clients"""
    global multi_index_searcher, api_client
    
    try:
        print("🚀 Starting Enhanced Image Search Engine API...")
        
        # Load CLIP model
        print("📦 Loading CLIP model...")
        get_model()
        
        # Load FAISS indices
        print("📦 Loading FAISS indices...")
        multi_index_searcher = MultiIndexSearcher()
        
        # Print stats
        stats = multi_index_searcher.get_stats()
        print(f"✅ Loaded indices: {stats['indices_loaded']}")
        print(f"✅ Total indexed images: {stats['total_images']}")
        
        # Initialize API client
        if settings.ENABLE_EXTERNAL_APIS:
            print("🌐 Initializing external API clients...")
            api_client = AggregatedSearchClient()
            print("✅ API clients ready")
        
        print("✅ API is ready!")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("💡 Make sure to run 'python indexer.py' first!")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Image Search Engine API",
        "version": "2.0.0",
        "features": [
            "Local image search",
            "COCO dataset search",
            "External API integration (Openverse, Lexica, etc.)",
            "Multi-source result fusion"
        ],
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "search_text": "/search-text",
            "search_image": "/search-image",
            # FIX ADDITION
            "image_content": "/image-content?path=..."
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = multi_index_searcher.get_stats() if multi_index_searcher else {}
    
    return {
        "status": "healthy",
        "model_loaded": get_model() is not None,
        "indices_loaded": stats.get('indices_loaded', []),
        "total_images": stats.get('total_images', 0),
        "external_apis_enabled": settings.ENABLE_EXTERNAL_APIS
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    if not multi_index_searcher:
        raise HTTPException(status_code=503, detail="Searcher not initialized")
    
    stats = multi_index_searcher.get_stats()
    stats['config'] = {
        'coco_enabled': settings.ENABLE_COCO,
        'external_apis_enabled': settings.ENABLE_EXTERNAL_APIS,
        'result_fusion': settings.ENABLE_RESULT_FUSION
    }
    
    return stats


@app.post("/search-text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Enhanced text search across multiple sources
    
    Sources can include: 'local', 'coco', 'openverse', 'lexica', 'unsplash', 'pexels'
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not multi_index_searcher:
            raise HTTPException(status_code=503, detail="Searcher not initialized")
        
        all_results = []
        sources_used = []
        
        # Determine which sources to use
        requested_sources = request.sources or ['local', 'coco']
        index_sources = [s for s in requested_sources if s in ['local', 'coco', 'combined']]
        api_sources = [s for s in requested_sources if s in ['openverse', 'lexica', 'unsplash', 'pexels']]
        
        # Search local/COCO indices
        if index_sources or not request.sources:
            query_vector = embed_text(request.query)
            
            index_results = multi_index_searcher.search(
                query_vector,
                k=request.top_k,
                sources=index_sources if index_sources else None
            )
            
            all_results.extend(index_results)
            sources_used.extend(multi_index_searcher.get_stats()['indices_loaded'])
        
        # Search external APIs
        if settings.ENABLE_EXTERNAL_APIS and request.include_apis and api_client:
            try:
                if api_sources:
                    api_results = api_client.search_specific(
                        request.query,
                        api_sources,
                        limit_per_source=max(5, request.top_k // 2)
                    )
                else:
                    api_results = api_client.search_all(
                        request.query,
                        limit_per_source=max(5, request.top_k // 4)
                    )
                
                # Add score for API results (lower priority than indexed)
                for i, result in enumerate(api_results):
                    result['score'] = 0.5 - (i * 0.01)  # Decreasing score
                    result['rank'] = len(all_results) + i + 1
                
                all_results.extend(api_results)
                sources_used.extend([r['source'] for r in api_results])
                
            except Exception as e:
                print(f"API search error: {e}")
        
        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for result in all_results:
            key = result.get('url') or result.get('path')
            if key and key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit to top_k
        final_results = unique_results[:request.top_k]
        
        # Update ranks
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
        
        return SearchResponse(
            results=final_results,
            total=len(final_results),
            sources_used=list(set(sources_used))
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search-image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    top_k: Optional[int] = Query(10),
    sources: Optional[str] = Query(None),
    include_apis: Optional[bool] = Query(False)
):
    """
    Image similarity search
    Note: External APIs don't support image-to-image search
    """
    try:
        if not multi_index_searcher:
            raise HTTPException(status_code=503, detail="Searcher not initialized")
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate embedding
        query_vector = embed_image(image)
        
        # Parse sources
        source_list = sources.split(',') if sources else None
        
        # Search indices
        results = multi_index_searcher.search(
            query_vector,
            k=top_k,
            sources=source_list
        )
        
        sources_used = multi_index_searcher.get_stats()['indices_loaded']
        
        return SearchResponse(
            results=results,
            total=len(results),
            sources_used=sources_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/sources")
async def list_sources():
    """List all available search sources"""
    sources = {
        "indexed": multi_index_searcher.get_stats()['indices_loaded'] if multi_index_searcher else [],
        "apis": []
    }
    
    if settings.ENABLE_EXTERNAL_APIS:
        sources["apis"] = ["openverse", "lexica"]
        if settings.UNSPLASH_ACCESS_KEY:
            sources["apis"].append("unsplash")
        if settings.PEXELS_API_KEY:
            sources["apis"].append("pexels")
    
    return sources

# --- FIX ADDITION: Endpoint to serve local image content ---
@app.get("/image-content")
async def get_image_content(path: str = Query(..., description="Absolute path to the local image file")):
    """
    Serves a local image file as a FileResponse. This allows the Streamlit
    frontend to access images from indexed datasets like COCO.
    """
    # Security check: Ensure the path exists and is a file
    if not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Image file not found at path: {path}")
    
    # You might want additional security checks here to ensure the path 
    # is within an expected directory (e.g., 'data/coco')
    
    try:
        # Return the file as a response
        return FileResponse(path)
    except Exception:
        raise HTTPException(status_code=500, detail="Error serving file content.")
# --- END FIX ADDITION ---


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)