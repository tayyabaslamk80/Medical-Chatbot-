# rag/vector_store.py - KEEP SAME DIMENSION (1536) WITH REDIS CACHE
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List, Optional
import time
from dotenv import load_dotenv
import hashlib

load_dotenv()

# Import cache manager
from langchain_cache import is_redis_connected, cache_get, cache_set, generate_cache_key

class PineconeVectorStoreManager:
    """Manages Pinecone vector store operations with Redis caching"""
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        dimension: int = 1536
    ):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.dimension = dimension
        self.pc = None
        self.index = None
        self.vectorstore = None
        self.embeddings = None
        
    def initialize(self, embeddings=None):
        """Initialize Pinecone connection and vector store"""
        try:
            print(f"🔄 Initializing Pinecone: {self.index_name}")
            
            # Use provided embeddings or get from retriever
            if embeddings is None:
                try:
                    from rag.retriever import get_embeddings
                    embeddings = get_embeddings()
                    if not embeddings:
                        raise ValueError("Could not get embeddings from retriever")
                except ImportError:
                    raise ValueError("Could not import embeddings from retriever module")
            
            # Store embeddings reference
            self.embeddings = embeddings
            
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check/create index
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"📝 Creating index: {self.index_name} (dimension: {self.dimension})")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment.split('-')[1] if '-' in self.environment else "us-east-1"
                    )
                )
                
                # Wait for index
                print("⏳ Waiting for index to be ready...")
                time.sleep(30)
            else:
                print(f"✅ Using existing index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Create LangChain vector store
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=embeddings,
                text_key="text"
            )
            
            print(f"✅ Pinecone initialized: {self.index_name}")
            return True
            
        except Exception as e:
            print(f"❌ Pinecone initialization failed: {e}")
            raise
    
    def add_documents(self, documents: List[Document], embeddings=None):
        """Add documents to Pinecone with cache invalidation"""
        try:
            print(f"📤 Adding {len(documents)} documents to Pinecone...")
            
            # Initialize if needed
            if not self.vectorstore:
                if embeddings:
                    self.initialize(embeddings)
                else:
                    self.initialize()
            
            # Clear search cache for relevant queries when adding new documents
            if is_redis_connected():
                # Clear vector search cache since we're adding new documents
                cache_keys = self._get_cache_keys_for_invalidation(documents)
                if cache_keys:
                    print(f"🧹 Invalidating {len(cache_keys)} cached search results...")
            
            # Split into batches
            batch_size = 20  # Reasonable batch size for OpenRouter
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(documents) - 1)//batch_size + 1
                
                print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                try:
                    # Add batch
                    ids = self.vectorstore.add_documents(batch)
                    total_added += len(ids)
                    print(f"   ✅ Added batch {batch_num}")
                    
                    # Delay between batches
                    if batch_num < total_batches:
                        time.sleep(2)
                        
                except Exception as batch_error:
                    print(f"   ⚠️ Batch {batch_num} failed: {batch_error}")
                    
                    # Try alternative
                    try:
                        texts = [doc.page_content[:4000] for doc in batch]
                        metadatas = [doc.metadata for doc in batch]
                        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
                        total_added += len(batch)
                        print(f"   ✅ Added batch {batch_num} using add_texts")
                    except Exception as alt_error:
                        print(f"   ❌ Alternative method failed: {alt_error}")
            
            success_rate = (total_added / len(documents)) * 100
            print(f"📊 Added {total_added}/{len(documents)} documents ({success_rate:.1f}% success)")
            return total_added > 0
            
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search in Pinecone with Redis caching"""
        # Check cache first
        if is_redis_connected():
            cache_key = generate_cache_key(f"vector:search:{self.index_name}", query, k)
            cached_result = cache_get(cache_key)
            
            if cached_result is not None:
                print(f"💾 Vector search cache HIT: {query[:50]}...")
                
                # Convert cached data back to Document objects
                documents = []
                for doc_data in cached_result:
                    doc = Document(
                        page_content=doc_data["page_content"],
                        metadata=doc_data["metadata"]
                    )
                    documents.append(doc)
                return documents
        
        print(f"💾 Vector search cache MISS: {query[:50]}...")
        
        if not self.vectorstore:
            self.initialize()
        
        try:
            documents = self.vectorstore.similarity_search(query, k=k)
            
            # Cache the results
            if is_redis_connected() and documents:
                # Convert documents to serializable format
                serializable_docs = []
                for doc in documents:
                    serializable_docs.append({
                        "page_content": doc.page_content[:2000],  # Limit size for caching
                        "metadata": doc.metadata
                    })
                
                cache_key = generate_cache_key(f"vector:search:{self.index_name}", query, k)
                cache_set(cache_key, serializable_docs, ttl=1800)  # 30 minutes
                print(f"💾 Cached vector search results for: {query[:50]}...")
            
            return documents
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _get_cache_keys_for_invalidation(self, documents: List[Document]) -> List[str]:
        """Get cache keys to invalidate when adding new documents"""
        if not is_redis_connected():
            return []
        
        try:
            # Get all vector search cache keys
            keys = get_redis_client().keys(f"vector:search:{self.index_name}:*")
            return keys
        except:
            return []
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get LangChain retriever from vector store"""
        if not self.vectorstore:
            self.initialize()
        
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_document_count(self) -> int:
        """Get number of documents in index"""
        if not self.index:
            return 0
        
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 0
    
    def clear_index(self):
        """Clear all vectors from index and related cache"""
        try:
            if self.index:
                self.index.delete(delete_all=True)
                print("🧹 Cleared Pinecone index")
                
                # Clear related cache
                if is_redis_connected():
                    cache_clear_pattern(f"vector:search:{self.index_name}:*")
                    print("🧹 Cleared vector search cache")
                
                return True
        except Exception as e:
            print(f"❌ Failed to clear index: {e}")
            return False

# Global instance
_pinecone_store = None

def get_vector_store() -> PineconeVectorStoreManager:
    """Get singleton Pinecone store instance"""
    global _pinecone_store
    if _pinecone_store is None:
        _pinecone_store = PineconeVectorStoreManager()
    return _pinecone_store

def add_documents_to_vector_store(documents: List[Document]) -> bool:
    """Add documents to vector store"""
    try:
        print(f"📤 add_documents_to_vector_store called with {len(documents)} documents")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Get embeddings from retriever
        try:
            from rag.retriever import get_embeddings
            embeddings = get_embeddings()
        except ImportError:
            embeddings = None
        
        # Add documents
        result = vector_store.add_documents(documents, embeddings)
        
        if result:
            print(f"✅ Successfully added documents to Pinecone")
        else:
            print(f"❌ Failed to add documents to Pinecone")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in add_documents_to_vector_store: {e}")
        return False

# Import for cache clearing
from langchain_cache import cache_clear_pattern, get_redis_client