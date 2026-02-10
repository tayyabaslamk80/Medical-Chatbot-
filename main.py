import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from rag.document_loader import load_medical_documents
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import routers
from routers import auth_router, chatbot_router, firebase_router, rag_router

# Import database and firebase config
from database import client as mongo_client, db, users_collection
from firebase_config import firebase_app

# Import cache manager
from langchain_cache import get_cache_stats, cache_clear_pattern, is_redis_connected, cache_manager

# ============================================
# Check if index.html exists
# ============================================
INDEX_HTML_PATH = Path("index.html")
if not INDEX_HTML_PATH.exists():
    logger.warning("⚠️ index.html not found in root directory")
    logger.info("💡 Creating a minimal index.html for testing...")
    INDEX_HTML_PATH.write_text("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            h1 { color: #4f46e5; }
            .status { background: #f0f9ff; padding: 20px; border-radius: 10px; margin: 20px auto; max-width: 600px; }
        </style>
    </head>
    <body>
        <h1>🏥 Medical AI Assistant</h1>
        <div class="status">
            <h2>Backend is Running! ✅</h2>
            <p>Your Medical Chatbot API is successfully running.</p>
            <p><a href="/api/docs">View API Documentation</a></p>
            <p><a href="/api/health">Check Health Status</a></p>
        </div>
        <p>Replace this file with your actual index.html from your frontend.</p>
    </body>
    </html>
    """)

# ============================================
# Lifespan Manager for Startup/Shutdown Events
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup events
    logger.info("🚀 Starting Medical Chatbot Application...")
    
    # Check MongoDB connection
    try:
        if mongo_client:
            mongo_client.admin.command('ping')
            logger.info("✅ MongoDB connection verified")
        else:
            logger.warning("⚠️ MongoDB connection not available")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
    
    # Check Firebase initialization
    if firebase_app:
        logger.info("✅ Firebase Authentication initialized")
    else:
        logger.warning("⚠️ Firebase Authentication not available")
    
    # Initialize Redis cache
    if is_redis_connected():
        logger.info("✅ Redis cache initialized")
    else:
        logger.warning("⚠️ Redis cache not available")
    
    yield
    
    # Shutdown events
    logger.info("🛑 Shutting down Medical Chatbot Application...")
    
    # Close MongoDB connection
    if mongo_client:
        try:
            mongo_client.close()
            logger.info("✅ MongoDB connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing MongoDB connection: {e}")

# ============================================
# FastAPI Application Setup
# ============================================
app = FastAPI(
    title="Medical AI Assistant API",
    description="A comprehensive medical chatbot with RAG capabilities, voice processing, and multi-language support",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ============================================
# CORS Configuration
# ============================================
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://medicalchatbot-c1936.firebaseapp.com",
]

# Add CORS origins from environment variable
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    origins.extend([origin.strip() for origin in cors_origins_env.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# Include Routers
# ============================================
app.include_router(auth_router.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chatbot_router.router, prefix="/api/chatbot", tags=["Chatbot"])
app.include_router(firebase_router.router, prefix="/api/firebase", tags=["Firebase"])
app.include_router(rag_router.router, prefix="/api/rag", tags=["RAG System"])

# ============================================
# Serve Frontend HTML Files
# ============================================
@app.get("/")
async def serve_index():
    """Serve the main HTML interface"""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Medical Chatbot</title></head>
            <body>
                <h1>Medical Chatbot Backend is Running</h1>
                <p>But index.html was not found.</p>
                <p><a href="/api/docs">API Documentation</a></p>
            </body>
        </html>
        """)

@app.get("/chat")
async def serve_chat_interface():
    """Serve the chat interface"""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")

# ============================================
# Cache Management Endpoints
# ============================================
@app.get("/api/cache/stats")
async def cache_stats():
    """Get Redis cache statistics"""
    stats = get_cache_stats()
    
    # Add health status
    stats["cache_enabled"] = is_redis_connected()
    stats["timestamp"] = datetime.now().isoformat()
    
    return stats

@app.post("/api/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear cache entries by pattern"""
    if pattern == "*":
        # For safety, require confirmation for clearing all cache
        return {
            "error": "Use pattern 'rag:*' or 'llm:*' to clear specific cache. "
                    "To clear all cache, use pattern 'all' explicitly."
        }
    
    if pattern == "all":
        pattern = "*"
    
    cleared = cache_clear_pattern(pattern)
    
    return {
        "success": True,
        "message": f"Cleared {cleared} cache entries",
        "pattern": pattern,
        "cleared_count": cleared,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cache/health")
async def cache_health():
    """Check cache system health"""
    stats = get_cache_stats()
    
    return {
        "status": "healthy" if stats.get("connected") else "unhealthy",
        "redis_connected": stats.get("connected", False),
        "keys": stats.get("total_keys", 0),
        "memory": stats.get("memory_used", "N/A"),
        "hit_rate": f"{stats.get('hit_rate', 0):.1f}%",
        "uptime_seconds": stats.get("uptime", 0)
    }

# ============================================
# Health and Status Endpoints
# ============================================
@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "Medical AI Assistant",
        "version": "2.0.0"
    }
    
    # Check MongoDB
    try:
        if mongo_client:
            mongo_client.admin.command('ping')
            health_status["mongodb"] = "connected"
            health_status["database_name"] = db.name
        else:
            health_status["mongodb"] = "disconnected"
    except Exception as e:
        health_status["mongodb"] = f"error: {str(e)}"
    
    # Check Firebase
    health_status["firebase"] = "initialized" if firebase_app else "not_initialized"
    
    # Check Redis Cache
    health_status["redis_cache"] = "connected" if is_redis_connected() else "disconnected"
    
    # Get cache stats if connected
    if is_redis_connected():
        cache_stats = get_cache_stats()
        health_status["cache_stats"] = {
            "keys": cache_stats.get("total_keys", 0),
            "hit_rate": f"{cache_stats.get('hit_rate', 0):.1f}%"
        }
    
    # Check environment variables
    env_checks = {
        "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "pinecone_key": bool(os.getenv("PINECONE_API_KEY")),
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "mongodb_url": bool(os.getenv("MONGODB_URL")),
        "redis_host": bool(os.getenv("REDIS_HOST", "localhost")),
    }
    health_status["environment"] = env_checks
    
    return health_status

@app.get("/api/status")
async def system_status():
    """Detailed system status with configuration info"""
    # Safely get collection counts
    user_count = 0
    chat_count = 0
    message_count = 0
    
    if users_collection and hasattr(users_collection, 'count_documents'):
        try:
            user_count = users_collection.count_documents({})
        except:
            user_count = 0
    
    # Try to get chat and message counts
    if db:
        try:
            chat_count = db.chats.count_documents({}) if hasattr(db, 'chats') else 0
            message_count = db.messages.count_documents({}) if hasattr(db, 'messages') else 0
        except:
            chat_count = 0
            message_count = 0
    
    status_info = {
        "system": {
            "name": "Medical AI Assistant",
            "version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        },
        "database": {
            "mongodb_connected": mongo_client is not None,
            "firebase_initialized": firebase_app is not None,
            "redis_cache_connected": is_redis_connected(),
            "user_count": user_count,
            "chat_count": chat_count,
            "message_count": message_count
        },
        "ai_services": {
            "llm_provider": "OpenRouter",
            "model": os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-70b-instruct"),
            "rag_enabled": bool(os.getenv("PINECONE_API_KEY")),
            "voice_enabled": bool(os.getenv("OPENAI_API_KEY")),
            "whisper_model": os.getenv("WHISPER_MODEL", "whisper-1")
        },
        "features": {
            "rag_system": True,
            "voice_chat": True,
            "multi_language": True,
            "firebase_auth": True,
            "mongodb_storage": True,
            "redis_caching": is_redis_connected()
        },
        "endpoints": {
            "frontend": "/",
            "api_docs": "/api/docs",
            "chat": "/api/chatbot/chat",
            "voice_chat": "/api/chatbot/voice-chat",
            "auth": "/api/auth",
            "firebase": "/api/firebase",
            "rag": "/api/rag",
            "cache_stats": "/api/cache/stats",
            "cache_health": "/api/cache/health"
        }
    }
    
    return status_info

# ============================================
# API Information Endpoint
# ============================================
@app.get("/api/info")
async def api_information():
    """API information and documentation links"""
    return {
        "application": "Medical AI Assistant",
        "description": "A comprehensive medical chatbot with AI-powered assistance",
        "author": "Tayyab Aslam",
        "version": "2.0.0",
        "documentation": {
            "swagger_ui": "/api/docs",
            "redoc": "/api/redoc",
            "openapi_spec": "/api/openapi.json"
        },
        "repository": "https://github.com/tayyabaslam/medical-chatbot",
        "license": "MIT",
        "support": {
            "email": "tayyabaslam@example.com",
            "issues": "https://github.com/tayyabaslam/medical-chatbot/issues"
        }
    }

# ============================================
# Environment Configuration Endpoint
# ============================================
@app.get("/api/config")
async def get_configuration():
    """Get non-sensitive configuration (for debugging)"""
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": {
            "name": os.getenv("DATABASE_NAME", "medical_chatbot"),
            "url": "configured" if os.getenv("MONGODB_URL") else "not_configured"
        },
        "cache": {
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": os.getenv("REDIS_PORT", "6379"),
            "redis_db": os.getenv("REDIS_DB", "0"),
            "connected": is_redis_connected()
        },
        "ai_services": {
            "llm_model": os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-70b-instruct"),
            "rag_enabled": bool(os.getenv("PINECONE_API_KEY")),
            "voice_enabled": bool(os.getenv("OPENAI_API_KEY")),
            "max_audio_size": f"{os.getenv('MAX_AUDIO_SIZE_MB', '10')} MB"
        },
        "features": {
            "supported_languages": os.getenv("SUPPORTED_LANGUAGES", "hi,en,bn,ta,te,mr,gu,kn,ml,pa").split(","),
            "rag_chunk_size": os.getenv("RAG_CHUNK_SIZE", "1000"),
            "rag_chunk_overlap": os.getenv("RAG_CHUNK_OVERLAP", "200")
        }
    }

# ============================================
# Error Handlers
# ============================================
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    """Custom 404 error handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested resource {request.url.path} was not found",
            "suggestions": [
                "Check the URL path",
                "Visit /api/docs for API documentation",
                "Visit / for the web interface"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 error handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Something went wrong on our end"
        }
    )

# ============================================
# Application Entry Point
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    
    logger.info(f"🚀 Starting Medical Chatbot on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/api/docs")
    logger.info(f"🌐 Web Interface: http://{host}:{port}/")
    logger.info(f"💾 Redis Cache: {'Enabled' if is_redis_connected() else 'Disabled'}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )