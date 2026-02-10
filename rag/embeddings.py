import os
from typing import List, Optional
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenRouterembeddings:
    def __init__(self, api_key: Optional[str] = None, model: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize OpenRouter embeddings.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: OpenRouter model name for embeddings
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Only print if this is the first initialization of this model
        if not hasattr(self.__class__, '_initialized_models'):
            self.__class__._initialized_models = set()
        
        if model not in self.__class__._initialized_models:
            print(f"✅ OpenRouter Embeddings initialized with model: {model}")
            self.__class__._initialized_models.add(model)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string using OpenRouter.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding floats
        """
        return self.embed_documents([text])[0]
    
    # ADD THIS: LangChain compatible method
    def embed_text_langchain(self, text: str) -> List[float]:
        """LangChain compatible method for embedding single queries."""
        return self.embed_text(text)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents using OpenRouter batch API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Your site URL
            "X-Title": "Medical Chatbot"  # Your app name
        }
        
        # OpenRouter expects array of strings for embeddings
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenRouter API error: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        """
        # Test with a small text to get dimension
        dummy_embedding = self.embed_text("test")
        return len(dummy_embedding)

# LangChain compatible wrapper
class OpenRouterLangChainEmbeddings:
    def __init__(self, model: str = "openai/text-embedding-3-small"):
        """
        LangChain compatible wrapper for OpenRouter embeddings.
        """
        # Track initialized models at class level
        if not hasattr(self.__class__, '_initialized_models'):
            self.__class__._initialized_models = set()
        
        self.client = OpenRouterembeddings(model=model)
        
        # Only print if this is the first initialization of this model
        if model not in self.__class__._initialized_models:
            print(f"✅ OpenRouter LangChain Embeddings initialized with model: {model}")
            self.__class__._initialized_models.add(model)
    
    def embed_text(self, text: str) -> List[float]:
        """LangChain interface method for single query."""
        return self.client.embed_text(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain interface method for multiple documents."""
        return self.client.embed_documents(texts)

# Create global instances - but with tracking to avoid duplicate messages
_embeddings_initialized = False
_langchain_embeddings_initialized = False

def _init_embeddings_once():
    """Initialize embeddings only once to avoid duplicate messages."""
    global openrouter_embeddings, _embeddings_initialized
    
    if not _embeddings_initialized:
        openrouter_embeddings = OpenRouterembeddings()
        _embeddings_initialized = True
    return openrouter_embeddings

def _init_langchain_embeddings_once():
    """Initialize LangChain embeddings only once to avoid duplicate messages."""
    global langchain_embeddings, _langchain_embeddings_initialized
    
    if not _langchain_embeddings_initialized:
        langchain_embeddings = OpenRouterLangChainEmbeddings()
        _langchain_embeddings_initialized = True
    return langchain_embeddings

# Initialize only when accessed
openrouter_embeddings = None
langchain_embeddings = None

def get_embeddings():
    """
    Get LangChain compatible embeddings instance.
    
    Returns:
        OpenRouterLangChainEmbeddings instance
    """
    return _init_langchain_embeddings_once()

def get_embedding_model():
    """
    Get the full OpenRouterEmbeddings instance.
    
    Returns:
        OpenRouterEmbeddings instance
    """
    return _init_embeddings_once()

# Test function
def test_embeddings():
    """Test the embeddings functionality."""
    try:
        embeddings = get_embeddings()
        test_text = "Hello, this is a test."
        result = embeddings.embed_text(test_text)
        print(f"✅ Embeddings test successful! Dimension: {len(result)}")
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

# Compatibility alias (if needed by other parts of your code)
embeddingsgenerator = OpenRouterLangChainEmbeddings
EmbeddingsGenerator = OpenRouterLangChainEmbeddings