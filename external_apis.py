"""
External API integrations for enhanced image search
Supports: Openverse, Lexica, Unsplash, Pexels
"""
import requests
from typing import List, Dict, Optional
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalAPIClient:
    """Base class for external API clients"""
    
    def __init__(self):
        self.timeout = settings.API_TIMEOUT
        self.max_results = settings.MAX_API_RESULTS
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search images - to be implemented by subclasses"""
        raise NotImplementedError


class OpenverseAPI(ExternalAPIClient):
    """
    Openverse API - Free, no key required
    Searches Creative Commons licensed images
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = settings.OPENVERSE_API_URL
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Openverse for images"""
        try:
            endpoint = f"{self.base_url}/images/"
            params = {
                'q': query,
                'page_size': min(limit, 20),  # Max 20 per request
                'format': 'json'
            }
            
            headers = {}
            if settings.OPENVERSE_CLIENT_ID and settings.OPENVERSE_CLIENT_SECRET:
                # Optional: increases rate limits
                headers['Authorization'] = f'Bearer {settings.OPENVERSE_CLIENT_ID}'
            
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'source': 'openverse',
                    'url': item.get('url'),
                    'thumbnail': item.get('thumbnail'),
                    'title': item.get('title', 'Untitled'),
                    'creator': item.get('creator', 'Unknown'),
                    'license': item.get('license', 'Unknown'),
                    'tags': item.get('tags', [])
                })
            
            logger.info(f"Openverse: Found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Openverse API error: {e}")
            return []


class LexicaAPI(ExternalAPIClient):
    """
    Lexica API - Free, no key required
    AI-generated art search engine
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = settings.LEXICA_API_URL
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Lexica for AI-generated images"""
        try:
            params = {
                'q': query,
                'limit': min(limit, 100)  # Max 100 per request
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('images', []):
                results.append({
                    'source': 'lexica',
                    'url': item.get('src'),
                    'thumbnail': item.get('srcSmall', item.get('src')),
                    'title': item.get('prompt', 'AI Generated'),
                    'prompt': item.get('prompt'),
                    'model': item.get('model', 'Stable Diffusion'),
                    'width': item.get('width'),
                    'height': item.get('height')
                })
            
            logger.info(f"Lexica: Found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Lexica API error: {e}")
            return []


class UnsplashAPI(ExternalAPIClient):
    """
    Unsplash API - Requires free API key
    High-quality stock photography
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = settings.UNSPLASH_API_URL
        self.api_key = settings.UNSPLASH_ACCESS_KEY
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Unsplash for images"""
        if not self.api_key:
            logger.warning("Unsplash API key not configured")
            return []
        
        try:
            params = {
                'query': query,
                'per_page': min(limit, 30),
                'orientation': 'landscape'
            }
            
            headers = {
                'Authorization': f'Client-ID {self.api_key}'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'source': 'unsplash',
                    'url': item['urls']['regular'],
                    'thumbnail': item['urls']['small'],
                    'title': item.get('description') or item.get('alt_description', 'Photo'),
                    'photographer': item['user']['name'],
                    'photographer_url': item['user']['links']['html'],
                    'download_link': item['links']['download']
                })
            
            logger.info(f"Unsplash: Found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Unsplash API error: {e}")
            return []


class PexelsAPI(ExternalAPIClient):
    """
    Pexels API - Requires free API key
    Free stock photos and videos
    """
    
    def __init__(self):
        super().__init__()
        self.base_url = settings.PEXELS_API_URL
        self.api_key = settings.PEXELS_API_KEY
    
    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Pexels for images"""
        if not self.api_key:
            logger.warning("Pexels API key not configured")
            return []
        
        try:
            params = {
                'query': query,
                'per_page': min(limit, 80)
            }
            
            headers = {
                'Authorization': self.api_key
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('photos', []):
                results.append({
                    'source': 'pexels',
                    'url': item['src']['large'],
                    'thumbnail': item['src']['small'],
                    'title': f"Photo by {item['photographer']}",
                    'photographer': item['photographer'],
                    'photographer_url': item['photographer_url'],
                    'width': item['width'],
                    'height': item['height']
                })
            
            logger.info(f"Pexels: Found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Pexels API error: {e}")
            return []


class AggregatedSearchClient:
    """Aggregates results from all external APIs"""
    
    def __init__(self):
        self.clients = {
            'openverse': OpenverseAPI(),
            'lexica': LexicaAPI(),
            'unsplash': UnsplashAPI(),
            'pexels': PexelsAPI()
        }
    
    def search_all(self, query: str, limit_per_source: int = 10) -> List[Dict]:
        """Search all available APIs"""
        all_results = []
        
        for name, client in self.clients.items():
            try:
                results = client.search(query, limit=limit_per_source)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error with {name}: {e}")
        
        logger.info(f"Total API results: {len(all_results)}")
        return all_results
    
    def search_specific(self, query: str, sources: List[str], limit_per_source: int = 10) -> List[Dict]:
        """Search specific APIs only"""
        all_results = []
        
        for source in sources:
            if source in self.clients:
                try:
                    results = self.clients[source].search(query, limit=limit_per_source)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error with {source}: {e}")
        
        return all_results