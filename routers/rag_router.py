# routers/rag_router.py - FIXED VERSION
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time

router = APIRouter()

# Import RAG components
try:
    from rag.retriever import rag_retriever, initialize_rag_system
    from rag.vector_store import get_vector_store
    from rag.document_loader import process_single_document
    
    RAG_AVAILABLE = rag_retriever is not None
except ImportError:
    RAG_AVAILABLE = False
    rag_retriever = None

# Pydantic models
class RAGQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RAGInitialize(BaseModel):
    data_dir: Optional[str] = None
    force_reload: Optional[bool] = False

# Health check
@router.get("/health")
async def rag_health():
    """Check RAG system health"""
    if not RAG_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "RAG system not available"
        }
    
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_document_count()
        
        return {
            "status": "healthy" if doc_count > 0 else "needs_initialization",
            "provider": "pinecone",
            "document_count": doc_count,
            "retriever_available": rag_retriever is not None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# Initialize RAG
@router.post("/initialize")
async def initialize_rag(request: RAGInitialize):
    """Initialize RAG system"""
    try:
        success = initialize_rag_system(
            data_dir=request.data_dir,
            force_reload=request.force_reload
        )
        
        if success:
            vector_store = get_vector_store()
            doc_count = vector_store.get_document_count()
            
            return {
                "success": True,
                "message": "RAG system initialized successfully",
                "document_count": doc_count,
                "force_reload": request.force_reload
            }
        else:
            return {
                "success": False,
                "message": "Failed to initialize RAG system"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")

# Search endpoint
@router.post("/search")
async def search_documents(request: RAGQuery):
    """Search for documents in RAG system"""
    if not RAG_AVAILABLE or not rag_retriever:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        documents = rag_retriever.get_relevant_documents(request.query)
        
        results = []
        for i, doc in enumerate(documents[:request.top_k]):
            results.append({
                "id": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown")
            })
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Get statistics
@router.get("/stats")
async def get_stats():
    """Get RAG system statistics"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_document_count()
        
        return {
            "success": True,
            "statistics": {
                "document_count": doc_count,
                "retriever_available": rag_retriever is not None,
                "provider": "pinecone"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Test connection
@router.get("/test")
async def test_connection():
    """Test Pinecone connection"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_document_count()
        
        return {
            "success": True,
            "message": "Pinecone connection successful",
            "provider": "pinecone",
            "document_count": doc_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")

# Clear vector store
@router.delete("/clear")
async def clear_vector_store():
    """Clear all documents from vector store"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    
    try:
        vector_store = get_vector_store()
        success = vector_store.clear_index()
        
        if success:
            return {
                "success": True,
                "message": "Vector store cleared successfully"
            }
        else:
            return {
                "success": False,
                "message": "Failed to clear vector store"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {str(e)}")