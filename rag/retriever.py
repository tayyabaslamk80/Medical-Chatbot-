# rag/retriever.py - UPDATED WITH OPENROUTER FOR OPENAI EMBEDDINGS AND REDIS CACHE
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import time
import hashlib

load_dotenv()

# Import cache manager
from langchain_cache import is_redis_connected, cache_get, cache_set, generate_cache_key

# Custom OpenAI-compatible embeddings using OpenRouter API
class OpenRouterOpenAIEmbeddings:
    """OpenAI-compatible embeddings using OpenRouter API"""
    
    def __init__(self, openai_api_key: str = None, model: str = "text-embedding-3-small"):
        # Use OpenRouter API key as if it were OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.openai_api_base = "https://openrouter.ai/api/v1"
        
        if not self.openai_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        print(f"✅ OpenRouter OpenAI Embeddings initialized with model: {self.model}")
        
        # Import OpenAI client and configure it to use OpenRouter
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base
            )
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents with Redis caching"""
        # Check cache first for each text
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if is_redis_connected():
                cache_key = generate_cache_key(f"embedding:doc:{self.model}", text=text[:500])
                cached = cache_get(cache_key)
                if cached is not None:
                    cached_results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all were cached, return in correct order
        if not uncached_texts and cached_results:
            print(f"💾 All embeddings from cache ({len(texts)} documents)")
            # Reconstruct in original order
            result = [None] * len(texts)
            for i, embedding in cached_results:
                result[i] = embedding
            return result
        
        if cached_results:
            print(f"💾 {len(cached_results)}/{len(texts)} embeddings from cache, computing {len(uncached_texts)}...")
        else:
            print(f"💾 Computing embeddings for {len(texts)} documents...")
        
        try:
            # Get embeddings for uncached texts
            response = self.client.embeddings.create(
                model=self.model,
                input=uncached_texts
            )
            
            # Process new embeddings
            new_embeddings = [item.embedding for item in response.data]
            
            # Cache the new embeddings
            if is_redis_connected():
                for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                    cache_key = generate_cache_key(f"embedding:doc:{self.model}", text=text[:500])
                    cache_set(cache_key, embedding, ttl=86400)  # 24 hours
            
            # Combine cached and new embeddings
            all_results = [None] * len(texts)
            
            # Add cached embeddings
            for i, embedding in cached_results:
                all_results[i] = embedding
            
            # Add new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                all_results[idx] = embedding
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error embedding documents: {e}")
            # Fallback to individual embedding
            embeddings = []
            for text in texts:
                embedding = self.embed_query(text)
                if embedding:
                    embeddings.append(embedding)
            return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with Redis caching"""
        # Check cache first
        if is_redis_connected():
            cache_key = generate_cache_key(f"embedding:query:{self.model}", text=text[:500])
            cached = cache_get(cache_key)
            
            if cached is not None:
                print(f"💾 Embedding from cache: {text[:50]}...")
                return cached
        
        print(f"💾 Computing embedding: {text[:50]}...")
        
        try:
            # Get embedding from API
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            if is_redis_connected():
                cache_key = generate_cache_key(f"embedding:query:{self.model}", text=text[:500])
                cache_set(cache_key, embedding, ttl=86400)  # 24 hours
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            return None

# Initialize embeddings globally
embeddings = None
rag_retriever = None
vector_store = None

def initialize_embeddings():
    """Initialize embeddings using OpenRouter for OpenAI model"""
    global embeddings
    
    if embeddings is not None:
        return embeddings
    
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            print("⚠️ OPENROUTER_API_KEY not found in environment variables")
            print("💡 Add to .env: OPENROUTER_API_KEY=your_openrouter_api_key")
            return None
        
        # Use the same OpenAI model but through OpenRouter
        embeddings = OpenRouterOpenAIEmbeddings(
            openai_api_key=openrouter_api_key,
            model="text-embedding-3-small"  # Same OpenAI model
        )
        
        print(f"✅ OpenAI embeddings via OpenRouter initialized: text-embedding-3-small (1536 dimensions)")
        return embeddings
        
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return None

def initialize_rag_system(data_dir: Optional[str] = None, force_reload: bool = False):
    """Initialize the complete RAG system"""
    global rag_retriever, vector_store, embeddings
    
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        if not embeddings:
            print("❌ Cannot initialize RAG without embeddings")
            return False
        
        # Initialize vector store
        from rag.vector_store import get_vector_store
        vector_store = get_vector_store()
        
        # Initialize vector store with embeddings
        vector_store.initialize(embeddings)
        
        # Get retriever from vector store
        rag_retriever = vector_store.get_retriever()
        
        print("✅ RAG system initialized with OpenAI embeddings via OpenRouter")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

# Auto-initialize if possible
try:
    # Check if we have the required API keys
    if os.getenv("PINECONE_API_KEY") and os.getenv("OPENROUTER_API_KEY"):
        print("🔑 API keys found, attempting to initialize RAG system...")
        success = initialize_rag_system()
        if success:
            print("✅ RAG system auto-initialized successfully")
        else:
            print("⚠️ RAG system auto-initialization failed")
    else:
        print("🔑 Missing API keys, RAG system will be initialized on first use")
except Exception as e:
    print(f"⚠️ Auto-initialization error: {e}")

# For backward compatibility
def get_embeddings():
    """Get embeddings instance"""
    if embeddings is None:
        initialize_embeddings()
    return embeddings

def get_retriever():
    """Get retriever instance"""
    global rag_retriever
    if rag_retriever is None:
        initialize_rag_system()
    return rag_retriever

def get_vectorstore():
    """Get vector store instance"""
    global vector_store
    if vector_store is None:
        from rag.vector_store import get_vector_store
        vector_store = get_vector_store()
    return vector_store