"""
Enhanced Streamlit Frontend with Multi-Source Search
Supports local, COCO, and external API search
"""
import streamlit as st
import requests
from PIL import Image
import io
import os
from pathlib import Path

# Configuration - Use environment variable for deployment flexibility
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Enhanced Image Search Engine",
    page_icon="🔍",
    layout="wide"
)


def check_api_health():
    """Check if API is running and get stats"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def get_available_sources():
    """Get list of available search sources"""
    try:
        response = requests.get(f"{API_URL}/sources", timeout=2)
        if response.status_code == 200:
            return response.json()
        return {"indexed": [], "apis": []}
    except:
        return {"indexed": [], "apis": []}


# --- NEW FUNCTION TO FETCH LOCAL IMAGE CONTENT VIA API ---
def fetch_image_from_path(path: str):
    """Fetches image content (bytes) from the backend API using the file path."""
    try:
        # This assumes your FastAPI backend has an endpoint like /image-content?path=...
        response = requests.get(
            f"{API_URL}/image-content", 
            params={'path': path}, 
            timeout=10
        )
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        # Optionally log or show error, but keeping it quiet prevents clutter
        # st.warning(f"Failed to fetch image content for {path}: {e}")
        return None
# --- END NEW FUNCTION ---


def search_by_text(query: str, top_k: int = 10, sources: list = None, include_apis: bool = True):
    """Search images by text query"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "include_apis": include_apis
        }
        
        if sources:
            payload["sources"] = sources
        
        response = requests.post(
            f"{API_URL}/search-text",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def search_by_image(image_file, top_k: int = 10, sources: list = None):
    """Search images by uploaded image"""
    try:
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        params = {
            "top_k": top_k,
            "sources": ",".join(sources) if sources else None
        }
        
        response = requests.post(
            f"{API_URL}/search-image",
            files=files,
            params={k: v for k, v in params.items() if v is not None},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def display_result_card(result: dict, col):
    """Display a single result card"""
    with col:
        try:
            image_to_display = None
            
            # Display image
            # 1. Handle External API result (URL)
            if 'url' in result and result['url']:
                # (Fix #1 from last response: Removed use_container_width=True)
                image_to_display = result.get('thumbnail') or result['url']
                
            # 2. Handle Local/COCO result (Path)
            elif 'path' in result: 
                # FIX #2: Fetch image content as bytes from the backend API
                image_bytes = fetch_image_from_path(result['path'])
                if image_bytes:
                    image_to_display = Image.open(io.BytesIO(image_bytes))
            
            # Display the fetched image or URL
            if image_to_display:
                st.image(image_to_display)
            else:
                st.warning("Image not available")
                return

            # Display metadata
            st.caption(f"**{result.get('title') or result.get('filename', 'Unknown')}**")
            
            # Source badge
            source = result.get('source', 'unknown')
            source_colors = {
                'local': '🟢',
                'coco': '🔵',
                'openverse': '🟠',
                'lexica': '🟣',
                'unsplash': '🔴',
                'pexels': '🟡'
            }
            st.caption(f"{source_colors.get(source, '⚪')} {source.upper()}")
            
            # Score
            if 'score' in result:
                st.caption(f"Score: {result['score']:.3f}")
            
            # Additional info for COCO
            if source == 'coco' and 'caption' in result:
                with st.expander("📝 Caption"):
                    st.write(result['caption'])
            
            # Additional info for APIs
            if source in ['openverse', 'unsplash', 'pexels']:
                if 'creator' in result or 'photographer' in result:
                    creator = result.get('creator') or result.get('photographer')
                    st.caption(f"👤 {creator}")
            
            # Download link for API results
            if 'url' in result:
                st.markdown(f"[🔗 View Source]({result['url']})", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error displaying result: {e}")


def display_results(results):
    """Display search results in a grid"""
    if not results or not results.get('results'):
        st.warning("No results found")
        return
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Results", results['total'])
    with col2:
        sources_str = ", ".join(results.get('sources_used', []))
        st.metric("Sources", len(results.get('sources_used', [])))
        st.caption(sources_str)
    
    st.markdown("---")
    
    # Display results in grid
    cols_per_row = 4
    results_list = results['results']
    
    for i in range(0, len(results_list), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results_list):
                display_result_card(results_list[idx], col)


def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🔍 Enhanced Image Search Engine")
    st.markdown("*Multi-source search powered by CLIP, COCO, and external APIs*")
    
    # Check API health
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.error("⚠️ API is not running! Start it with: `uvicorn main:app --reload`")
        st.stop()
    
    # Display API status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ API Connected")
    with col2:
        if health_data:
            st.info(f"📊 {health_data.get('total_images', 0)} indexed images")
    with col3:
        if health_data and health_data.get('external_apis_enabled'):
            st.info("🌐 APIs Enabled")
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Search Settings")
        
        # Get available sources
        available_sources = get_available_sources()
        
        # Number of results
        top_k = st.slider("Number of results", 5, 50, 20)
        
        st.markdown("---")
        st.subheader("📊 Data Sources")
        
        # Source selection
        indexed_sources = available_sources.get('indexed', [])
        api_sources = available_sources.get('apis', [])
        
        st.markdown("**Indexed Sources:**")
        selected_indexed = []
        for source in indexed_sources:
            if st.checkbox(source.upper(), value=True, key=f"idx_{source}"):
                selected_indexed.append(source)
        
        include_apis = False
        selected_apis = []
        
        if api_sources:
            st.markdown("**External APIs:**")
            include_apis = st.checkbox("Enable API Search", value=True)
            
            if include_apis:
                for source in api_sources:
                    if st.checkbox(source.capitalize(), value=True, key=f"api_{source}"):
                        selected_apis.append(source)
        
        st.markdown("---")
        st.markdown("### 💡 Features")
        st.markdown("""
        - 🏠 Local images
        - 🎯 COCO dataset (5000+ images)
        - 🌐 Openverse (Creative Commons)
        - 🎨 Lexica (AI art)
        - 📸 Unsplash (optional)
        - 🖼️ Pexels (optional)
        """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔤 Text Search", "🖼️ Image Search", "📊 Statistics"])
    
    # Text Search Tab
    with tab1:
        st.header("Search by Text Description")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            text_query = st.text_input(
                "Enter your search query",
                placeholder="e.g., cat sitting on a chair, sunset over mountains, red sports car...",
                label_visibility="collapsed"
            )
        
        with col2:
            # (Fix from last response: Removed use_container_width=True)
            search_button = st.button("🔍 Search", key="text_search")
        
        # Example queries
        st.markdown("**Try these examples:**")
        example_col1, example_col2, example_col3, example_col4 = st.columns(4)
        
        examples = [
            "cat playing with ball",
            "sunset over ocean",
            "person riding bicycle",
            "modern architecture"
        ]
        
        for i, (col, example) in enumerate(zip([example_col1, example_col2, example_col3, example_col4], examples)):
            with col:
                if st.button(f"💡 {example}", key=f"ex_{i}"):
                    text_query = example
                    search_button = True
        
        if search_button and text_query:
            with st.spinner("🔍 Searching across all sources..."):
                sources = selected_indexed + (selected_apis if include_apis else [])
                results = search_by_text(text_query, top_k, sources or None, include_apis)
                
                if results:
                    display_results(results)
        
        elif search_button and not text_query:
            st.warning("Please enter a search query")
    
    # Image Search Tab
    with tab2:
        st.header("Search by Image Similarity")
        
        # Check if we have indexed images
        if health_data and health_data.get('total_images', 0) == 0:
            st.warning("⚠️ **Image search is currently unavailable**")
            st.info("""
            Image-to-image search requires locally indexed images (COCO dataset or custom images).
            
            **What you can do:**
            - Use **Text Search** tab with external APIs (Openverse, Lexica)
            - The app maintainer can run `python indexer.py` to enable local image search
            """)
        else:
            st.info("Upload an image to find similar images in the indexed datasets")
            
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(uploaded_file, caption="Query Image")
                    
                    if st.button("🔍 Find Similar", key="image_search"):
                        with st.spinner("🔍 Searching for similar images..."):
                            uploaded_file.seek(0)
                            results = search_by_image(uploaded_file, top_k, selected_indexed or None)
                            
                            if results:
                                with col2:
                                    st.markdown("### Similar Images")
                                display_results(results)
    
    # Statistics Tab
    with tab3:
        st.header("📊 System Statistics")
        
        if health_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Indexed Data")
                st.json({
                    "indices_loaded": health_data.get('indices_loaded', []),
                    "total_images": health_data.get('total_images', 0)
                })
            
            with col2:
                st.subheader("Configuration")
                st.json({
                    "model": "CLIP ViT-B-32",
                    "vector_db": "FAISS",
                    "external_apis": health_data.get('external_apis_enabled', False)
                })
        
        st.markdown("---")
        st.subheader("Available Sources")
        st.json(available_sources)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, FastAPI, CLIP, FAISS & Multiple Data Sources | "
        "v2.0"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()