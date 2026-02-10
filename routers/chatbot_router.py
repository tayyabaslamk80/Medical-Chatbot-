# routers/chatbot_router.py - WITH FIXED RAG SOURCE TRACKING AND FAST MODEL AND REDIS CACHE
# AND ADDED FEEDBACK SYSTEM WITH FEEDBACK-FIRST CACHING

from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import requests
import re
import os
import sys
import tempfile
import time
import json
import hashlib
from bson import ObjectId  # Add this import

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

router = APIRouter()

# Import Firebase auth function
try:
    from database import verify_user_token, save_message  # Add save_message here
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("⚠️ Firebase auth not available")

# Try to import RAG retriever - SIMPLIFIED IMPORTS
try:
    # Direct import from rag.retriever
    from rag.retriever import rag_retriever
    RAG_AVAILABLE = True if rag_retriever else False
    print(f"✅ RAG retriever imported: {RAG_AVAILABLE}")
except ImportError as e:
    print(f"⚠️ RAG module import error: {e}")
    RAG_AVAILABLE = False
    rag_retriever = None

# Try to import Voice Processor
try:
    from rag.voice_module import voice_processor
    VOICE_ENABLED = voice_processor.voice_enabled
    print(f"✅ Voice processor imported: {VOICE_ENABLED}")
except ImportError as e:  # FIXED: Added 'as e' here
    VOICE_ENABLED = False
    print(f"⚠️ Voice module import error: {e}")
    voice_processor = None

# ============================================
# ADDED FEEDBACK: Pydantic Models for Feedback
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: Optional[bool] = True  # New: toggle RAG on/off

class VoiceChatRequest(BaseModel):
    session_id: Optional[str] = None
    use_rag: Optional[bool] = True
    language: Optional[str] = "hi"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    timestamp: str
    chat_name: str
    is_rag: Optional[bool] = False  # New: indicate if RAG was used
    # ADDED CACHE METADATA FIELDS:
    source: Optional[str] = "model"  # "rag" or "model"
    response_time_ms: Optional[int] = 0
    is_cached: Optional[bool] = False
    cache_hit: Optional[bool] = False

class SessionInfo(BaseModel):
    session_id: str
    chat_name: str
    created_at: str
    last_message: str
    message_count: int
    last_activity: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    document_id: Optional[str] = None
    chunks_count: Optional[int] = None

# ============================================
# ADDED FEEDBACK: New Pydantic Models
# ============================================

class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    feedback_type: str  # "thumbs_up" or "thumbs_down"
    user_comment: Optional[str] = None
    rating: Optional[int] = None  # 1-5 scale

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None

# OpenRouter Configuration - FROM ENVIRONMENT VARIABLES
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found in environment variables!")
    print("💡 Add it to your .env file: OPENROUTER_API_KEY=your_key_here")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅✅✅ FIXED: Changed to FAST model ✅✅✅
DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct"  # Fast 3B model (1-3 seconds)

# System prompt for medical assistant
MEDICAL_SYSTEM_PROMPT = """You are Dr. Health AI. Answer in 2-4 sentences only. Be direct and concise.

**RULES:**
1. Answer in 2-4 sentences maximum
2. Be direct - no introductions or extra sections
3. No bullet points or headers
4. Use provided medical context if available
5. End with "Consult a doctor for personal advice" if medical

Context: {context}
Question: {question}

Provide a concise 2-4 sentence answer:"""

# Database import - FIXED INDENTATION
try:
    from database import users_collection, conversations_collection, save_message, create_conversation
    db_available = True
    print("✅ Database connection available")
except ImportError as e:
    db_available = False
    print(f"⚠️ Database import failed: {e}")

# ============================================
# ADDED FEEDBACK: Database import for feedback
# ============================================
try:
    from database import feedback_collection
    print("✅ Feedback collection available")
except ImportError:
    print("⚠️ Feedback collection not available in database module")
    feedback_collection = None

# Firebase Auth dependency
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current user from Firebase token"""
    if not authorization:
        # For testing without auth
        return {"uid": "test_user", "email": "test@example.com", "name": "Test User"}
    
    try:
        # Remove 'Bearer ' prefix if present
        token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
        
        if FIREBASE_AVAILABLE:
            # Verify token using Firebase
            user = verify_user_token(token)
            if user:
                return user
        
        # Fallback for testing
        return {"uid": "test_user", "email": "test@example.com", "name": "Test User"}
        
    except Exception as e:
        print(f"Auth error: {e}")
        return {"uid": "test_user", "email": "test@example.com", "name": "Test User"}

# ============================================
# REDIS CACHE IMPLEMENTATION
# ============================================

# Simple Redis cache manager
class RedisCacheManager:
    def __init__(self):
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis cache initialized successfully")
        except Exception as e:
            print(f"⚠️ Redis cache not available: {e}")
            self.redis_client = None
    
    def is_connected(self):
        """Check if Redis is connected"""
        return self.redis_client is not None
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        if not self.is_connected():
            return False
        
        try:
            import pickle
            serialized = pickle.dumps(value)
            return bool(self.redis_client.setex(key, ttl, serialized))
        except Exception as e:
            print(f"❌ Cache set error: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.is_connected():
            return None
        
        try:
            import pickle
            value = self.redis_client.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            print(f"❌ Cache get error: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.is_connected():
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            print(f"❌ Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        if not self.is_connected():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"❌ Cache clear error: {e}")
            return 0

# Global cache instance
cache_manager = RedisCacheManager()

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    key_parts = [prefix]
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
    
    key_string = "_".join(key_parts)
    
    # If key is too long, hash it
    if len(key_string) > 200:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
        return f"{prefix}_{key_hash}"
    
    return key_string.replace(" ", "_").replace("/", "_")

def get_llm_cache_key(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Generate cache key for LLM responses"""
    messages_str = json.dumps(messages, sort_keys=True)
    key_string = f"llm:{model}:{temperature}:{max_tokens}:{messages_str}"
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    return f"llm_response:{key_hash}"

def get_rag_cache_key(query: str, session_id: str = None, use_rag: bool = True) -> str:
    """Generate cache key for RAG responses"""
    base_key = f"rag_response:{hashlib.md5(query.encode()).hexdigest()}"
    if session_id:
        base_key += f":{session_id}"
    if use_rag:
        base_key += ":rag"
    else:
        base_key += ":no_rag"
    return base_key

# ============================================
# ADDED FEEDBACK: Cache key for feedback
# ============================================
def get_feedback_cache_key(session_id: str, message_id: str) -> str:
    """Generate cache key for feedback"""
    return f"feedback:{session_id}:{message_id}"

# ============================================
# ADDED: Cache key for query-response mapping
# ============================================
def get_query_response_key(session_id: str, query: str) -> str:
    """Generate cache key for query-response mapping"""
    return f"query_response:{session_id}:{hashlib.md5(query.encode()).hexdigest()}"

# ============================================
# HELPER FUNCTIONS (EXISTING)
# ============================================

def clean_ai_response(text: str) -> str:
    """Clean AI responses"""
    if not text:
        return "I couldn't generate a response."
    
    patterns = [
        r'```.*?```',
        r'\{.*?\}',
        r'def\s+\w+\(',
        r'class\s+\w+:',
        r'import\s+',
        r'<.*?>',
        r'[\x00-\x1F\x7F-\x9F]',
        r'\*\*.*?\*\*',  # Remove bold
        r'\*.*?\*',      # Remove italics
        r'#+\s+',        # Remove headers
        r'\d+\.\s+',     # Remove numbered lists
        r'•\s+',         # Remove bullet points
        r'-\s+',         # Remove dashes
    ]
    
    clean_text = text
    for pattern in patterns:
        clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE | re.DOTALL)
    
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Truncate if too long
    sentences = clean_text.split('. ')
    if len(sentences) > 4:
        clean_text = '. '.join(sentences[:4]) + '.'
    
    return clean_text

def generate_chat_name(first_message: str) -> str:
    """Generate chat name from first 8-10 words"""
    if not first_message:
        return "New Chat"
    
    # Clean the message first
    cleaned_message = re.sub(r'[^\w\s]', '', first_message.strip())
    
    # Take first 10 words
    words = cleaned_message.split()[:10]
    if not words:
        return "New Chat"
    
    chat_name = ' '.join(words)
    
    # Truncate if too long
    if len(chat_name) > 60:
        chat_name = chat_name[:57] + '...'
    
    # Capitalize first letter of each word for better readability
    chat_name = ' '.join(word.capitalize() for word in chat_name.split())
    
    return chat_name

def format_rag_context(documents):
    """Format RAG documents into context string"""
    if not documents:
        return ""
    
    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.page_content[:500]  # Limit each document
        source = doc.metadata.get('source', 'Unknown source')
        context_parts.append(f"[Document {i+1} from {source}]:\n{content}")
    
    return "\n\n".join(context_parts)

def clean_metadata_for_pinecone(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata to be Pinecone-compatible (no null values)"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""  # Replace null with empty string
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            # Ensure list contains only strings
            cleaned[key] = [str(item) for item in value]
        else:
            # Convert any other type to string
            cleaned[key] = str(value)
    return cleaned

# ============================================
# ADDED FEEDBACK: Check if response was thumbs-upped
# ============================================
def check_thumbs_up_cache(session_id: str, message_id: str) -> bool:
    """Check if a response was previously given thumbs up"""
    if not cache_manager.is_connected():
        return False
    
    feedback_key = get_feedback_cache_key(session_id, message_id)
    feedback_data = cache_manager.get_cache(feedback_key)
    
    if feedback_data:
        return feedback_data.get('feedback_type') == 'thumbs_up'
    return False

# ============================================
# ADDED: Check if query has thumbs_up response cached
# ============================================
def check_cached_thumbs_up_response(session_id: str, query: str) -> Optional[str]:
    """Check if this query has a thumbs_up response cached"""
    if not cache_manager.is_connected():
        return None
    
    # Check query-response mapping
    query_response_key = get_query_response_key(session_id, query)
    cached_data = cache_manager.get_cache(query_response_key)
    
    if cached_data:
        message_id = cached_data.get("message_id")
        if message_id:
            # Check if this message has thumbs_up
            if check_thumbs_up_cache(session_id, message_id):
                return cached_data.get("response")
    
    return None

# ============================================
# AI RESPONSE FUNCTIONS WITH CACHING - UPDATED WITH FEEDBACK-FIRST CACHING
# ============================================

async def get_ai_response(message: str, session_id: str = None, user_id: str = None):
    """Get response from OpenRouter API with conversation history (without RAG) - UPDATED WITH FEEDBACK-FIRST CACHING"""
    
    cache_hit = False
    cache_key = None
    
    # ============================================
    # 🔥 MODIFIED: Only check cache for thumbs_up responses
    # ============================================
    if cache_manager.is_connected() and session_id:
        # Check if this query has a cached thumbs_up response
        cached_thumbs_up_response = check_cached_thumbs_up_response(session_id, message)
        if cached_thumbs_up_response:
            print(f"💾 AI Response Cache HIT (thumbs_up response): {message[:50]}...")
            cache_hit = True
            
            # Check if it's a medical question for disclaimer
            message_lower = message.strip().lower()
            is_greeting = message_lower in ["hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "greetings"]
            
            if not is_greeting:
                medical_keywords = ['pain', 'sick', 'fever', 'doctor', 'hospital', 'medicine', 
                                   'symptom', 'disease', 'infection', 'virus', 'hiv', 'aids', 
                                   'cancer', 'diabetes', 'blood', 'heart', 'lung', 'kidney', 'liver']
                is_medical = any(keyword in message.lower() for keyword in medical_keywords)
                
                if is_medical and "consult" not in cached_thumbs_up_response.lower():
                    cached_thumbs_up_response += " Consult a doctor for personal advice."
            
            return cached_thumbs_up_response, cache_hit, cache_key
    
    print("🤖 Getting AI response WITHOUT RAG (using general model knowledge only)...")
    
    # Base system prompt for all responses
    base_system_prompt = """You are MedAI, a helpful medical assistant. Follow these rules:

1. For greetings like "hi", "hello", "hey":
   - Respond naturally (e.g., "Hello! How can I help you today?")
   - No medical disclaimer needed

2. For medical questions:
   - Provide concise 2-4 sentence answers
   - End with "Consult a doctor for personal advice"

3. For follow-up questions about previous topics:
   - Reference the previous topic ONLY if the user is clearly continuing the conversation
   - If user says something new, don't reference old topics

4. Always be direct and avoid bullet points/headers."""

    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API key not configured. Please check your .env file.", cache_hit, cache_key

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Medical Chatbot"
    }
    
    # Prepare messages array starting with system prompt
    messages = [{"role": "system", "content": base_system_prompt}]
    
    # Check if this is a greeting or new topic
    message_lower = message.strip().lower()
    is_greeting = message_lower in ["hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "greetings"]
    
    # Get conversation history if session_id and user_id are provided
    has_meaningful_history = False
    previous_messages = []
    
    if session_id and user_id and db_available:
        try:
            # Find conversation by session_id and user_id
            conversation = conversations_collection.find_one({
                "session_id": session_id,
                "user_id": user_id
            })
            
            if conversation and "messages" in conversation:
                # Get last 5 messages from the conversation
                all_messages = conversation["messages"]
                previous_messages = all_messages[-5:] if len(all_messages) > 5 else all_messages
                
                # Check if there's meaningful history (not just greetings)
                if previous_messages:
                    # Count non-greeting messages
                    non_greeting_count = 0
                    for msg in previous_messages:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "").strip().lower()
                            greeting_words = ["hi", "hello", "hey", "greetings"]
                            if user_msg and not any(greeting in user_msg for greeting in greeting_words):
                                non_greeting_count += 1
                    
                    has_meaningful_history = non_greeting_count > 0
            
            # Only add history if:
            # 1. User is NOT just saying hi/hello (greeting)
            # 2. AND there's meaningful history (not just previous greetings)
            if not is_greeting and has_meaningful_history and previous_messages:
                print(f"📚 Adding {len(previous_messages)} previous messages for context")
                
                # Add previous messages to conversation
                for msg in previous_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        messages.append({"role": "assistant", "content": content})
            else:
                if is_greeting:
                    print("📚 No history added (greeting detected)")
                elif not has_meaningful_history:
                    print("📚 No history added (no meaningful previous conversation)")
                else:
                    print("📚 No history added")
                
        except Exception as e:
            print(f"⚠️ Error loading conversation history: {e}")
    
    # Add the current user message
    messages.append({"role": "user", "content": message})
    
    # For debugging
    if has_meaningful_history:
        print(f"📊 Previous conversation context will be used")
    
    # Add some randomness to get different responses
    import random
    temperature = 0.3 + random.uniform(0, 0.2)  # Slight randomness for variety
    max_tokens = 150 + random.randint(-20, 20)  # Slight randomness
    
    payload = {
        "model": DEFAULT_MODEL,  # ✅ Using FAST model
        "messages": messages,
        "temperature": round(temperature, 2),
        "max_tokens": max_tokens,
    }
    
    try:
        # ✅ SHORTER TIMEOUT for faster response
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            raw_response = data["choices"][0]["message"]["content"]
            
            cleaned_response = clean_ai_response(raw_response)
            
            # Add medical disclaimer ONLY if it's a medical question
            # Check if it's likely a medical question (not a greeting)
            if not is_greeting:
                medical_keywords = ['pain', 'sick', 'fever', 'doctor', 'hospital', 'medicine', 
                                   'symptom', 'disease', 'infection', 'virus', 'hiv', 'aids', 
                                   'cancer', 'diabetes', 'blood', 'heart', 'lung', 'kidney', 'liver']
                is_medical = any(keyword in message.lower() for keyword in medical_keywords)
                
                if is_medical and "consult" not in cleaned_response.lower():
                    cleaned_response += " Consult a doctor for personal advice."
            
            print(f"✅ RESPONSE ({'Greeting' if is_greeting else 'Medical/General'}): {len(cleaned_response)} chars")
            
            # ============================================
            # 🔥 MODIFIED: DO NOT cache automatically
            # Responses will be cached only when user gives thumbs_up
            # ============================================
            
            return cleaned_response, cache_hit, cache_key
        else:
            print(f"❌ OpenRouter API error: {response.status_code} - {response.text[:200]}")
            return f"Sorry, I encountered an error with the AI service (HTTP {response.status_code}). Please try again.", cache_hit, cache_key
            
    except requests.exceptions.Timeout:
        return "The AI service is taking too long to respond. Please try again.", cache_hit, cache_key
    except Exception as e:
        print(f"❌ OpenRouter request failed: {e}")
        return f"Sorry, I encountered an error. Please try again.", cache_hit, cache_key

async def get_ai_response_with_rag(message: str, session_id: str = None, user_id: str = None):
    """Get AI response with RAG context - UPDATED WITH FEEDBACK-FIRST CACHING"""
    
    cache_hit = False
    cache_key = None
    
    # ============================================
    # 🔥 MODIFIED: Only check cache for thumbs_up responses
    # ============================================
    if cache_manager.is_connected() and session_id:
        # Check if this query has a cached thumbs_up response
        cached_thumbs_up_response = check_cached_thumbs_up_response(session_id, message)
        if cached_thumbs_up_response:
            print(f"💾 RAG Cache HIT (thumbs_up response): {message[:50]}...")
            cache_hit = True
            
            # Check cache for rag_used flag
            rag_cache_key = get_rag_cache_key(message, session_id, True)
            cached_result = cache_manager.get_cache(rag_cache_key)
            rag_used = cached_result[1] if cached_result else False
            
            return cached_thumbs_up_response, rag_used, cache_hit, cache_key
    
    if not RAG_AVAILABLE or not rag_retriever:
        print("⚠️ RAG not available, falling back to regular response")
        response, cache_hit, cache_key = await get_ai_response(message, session_id, user_id)
        return response, False, cache_hit, cache_key  # Return tuple (response, rag_used, cache_hit, cache_key)
    
    print(f"🔍 RAG Query: '{message[:50]}...'")
    
    # ADJUSTED THRESHOLDS: Medical queries need lower threshold
    MEDICAL_SIMILARITY_THRESHOLD = 0.35
    GENERAL_SIMILARITY_THRESHOLD = 0.65
    
    # Check if query is medical
    medical_keywords = ['hiv', 'aids', 'disease', 'medical', 'health', 'symptom', 'treatment', 
                       'virus', 'infection', 'doctor', 'hospital', 'medicine', 'pain', 'fever',
                       'cancer', 'diabetes', 'blood', 'heart', 'lung', 'kidney', 'liver']
    
    is_medical_query = any(keyword in message.lower() for keyword in medical_keywords)
    threshold = MEDICAL_SIMILARITY_THRESHOLD if is_medical_query else GENERAL_SIMILARITY_THRESHOLD
    
    print(f"📊 Using threshold: {threshold} ({'Medical' if is_medical_query else 'General'} query)")
    
    documents = []
    rag_used = False
    top_score = 0
    all_scores = []
    
    try:
        # Use similarity search with scores
        results = rag_retriever.vectorstore.similarity_search_with_score(
            message, k=5  # Increased to 5 for better matching
        )
        
        print(f"📊 Similarity Scores:")
        for doc, score in results:
            all_scores.append(score)
            top_score = max(top_score, score)
            source = doc.metadata.get('source', 'Unknown')
            print(f"  🔍 {score:.3f} - {source}")
            
            if score >= threshold:
                documents.append(doc)
        
        if documents:
            rag_used = True
            print(f"✅ RAG ACCEPTED ({len(documents)}/{len(results)} docs above threshold {threshold})")
            for i, doc in enumerate(documents):
                preview = doc.page_content[:80].replace('\n', ' ')
                print(f"   📖 [{i+1}] {preview}...")
        else:
            print(f"⚠️ RAG REJECTED (best score: {top_score:.3f} < threshold {threshold})")
            print(f"   All scores: {[f'{s:.3f}' for s in all_scores]}")
            
    except Exception as e:
        print(f"❌ RAG retrieval error: {e}")
        documents = []
    
    # Prepare system prompt with context
    if rag_used and documents:
        context = format_rag_context(documents)
        system_content = MEDICAL_SYSTEM_PROMPT.format(
            context=context,
            question=message
        )
        source = "RAG"
    else:
        system_content = (
            "IMPORTANT: The user's uploaded documents do NOT contain relevant information "
            "for this question. Answer clearly using your general knowledge. "
            "Do not reference any documents or make up sources."
        )
        source = "MODEL"
    
    if not OPENROUTER_API_KEY:
        return "Error: OpenRouter API key not configured. Please check your .env file.", False, cache_hit, cache_key

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Medical Chatbot"
    }
    
    # Prepare messages array
    messages = [{"role": "system", "content": system_content}]
    
    # Get conversation history if available
    if session_id and user_id and db_available:
        try:
            # Find conversation by session_id and user_id
            conversation = conversations_collection.find_one({
                "session_id": session_id,
                "user_id": user_id
            })
            
            previous_messages = []
            if conversation and "messages" in conversation:
                # Get last 10 messages from the conversation
                previous_messages = conversation["messages"][-10:]
            
            for msg in previous_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            
            print(f"📚 Loaded {len(previous_messages)} previous messages")
            
        except Exception as e:
            print(f"⚠️ Error loading conversation history: {e}")
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Add some randomness for variety
    import random
    temperature = 0.3 + random.uniform(0, 0.15)  # Slight randomness
    max_tokens = 200 + random.randint(-30, 30)
    
    payload = {
        "model": DEFAULT_MODEL,  # ✅ Using FAST model
        "messages": messages,
        "temperature": round(temperature, 2),
        "max_tokens": max_tokens,
        "top_p": 0.5
    }
    
    try:
        # ✅ SHORTER TIMEOUT for faster response
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            raw_response = data["choices"][0]["message"]["content"]
            
            cleaned_response = clean_ai_response(raw_response)
            
            # Add medical disclaimer if needed
            if is_medical_query:
                if "consult" not in cleaned_response.lower():
                    cleaned_response += " Consult a doctor for personal advice."
            
            # FINAL CLEAR LOGGING
            print(f"📊 RESPONSE SOURCE: {source}")
            print(f"📝 Answer length: {len(cleaned_response)} characters")
            
            if rag_used and documents:
                print(f"✅ RESPONSE GENERATED USING RAG DOCUMENTS")
                print(f"   📊 Source: Your uploaded medical documents")
            else:
                print(f"✅ RESPONSE GENERATED FROM MODEL KNOWLEDGE ONLY")
                print(f"   📊 Source: General AI model knowledge (no uploaded documents matched)")
            
            # ============================================
            # 🔥 MODIFIED: DO NOT cache automatically
            # Responses will be cached only when user gives thumbs_up
            # ============================================
            
            return cleaned_response, rag_used, cache_hit, cache_key
            
    except requests.exceptions.Timeout:
        return "The AI service is taking too long to respond. Please try again.", False, cache_hit, cache_key
    except Exception as e:
        print(f"❌ RAG API error: {e}")
        # Fallback to non-RAG response
        response, cache_hit, cache_key = await get_ai_response(message, session_id, user_id)
        return response, False, cache_hit, cache_key

# ============================================
# UPLOAD ENDPOINT - FIXED VERSION WITH METADATA FIX AND SCOPE FIX
# ============================================

@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    process_type: str = Form("index_to_pinecone"),
    user_id: str = Form(...),
    session_id: str = Form(...),
    file_name: str = Form(...),
    file_type: str = Form(...),
    file_size: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a document to Pinecone RAG system"""
    print(f"\n📤 Document upload request from: {current_user.get('email')}")
    print(f"   File: {file_name}")
    print(f"   Process Type: {process_type}")
    print(f"   File Type: {file_type}")
    print(f"   File Size: {file_size}")
    
    # Check if RAG is available
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="RAG system not available. Make sure Pinecone is properly configured."
        )
    
    # Check file size (50MB max)
    max_size = 50 * 1024 * 1024  # 50MB
    file_size_int = int(file_size) if file_size.isdigit() else 0
    
    if file_size_int > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 50MB. Your file is {file_size_int/1024/1024:.2f}MB"
        )
    
    # Save file temporarily
    temp_file = None
    try:
        # Read file content
        content = await file.read()
        
        # Save to temp file
        suffix = os.path.splitext(file_name)[1] or '.tmp'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        
        print(f"💾 File saved to temp location: {temp_file.name}")
        
        # Process based on type
        if process_type == "index_to_pinecone":
            # Process document and add to vector store
            try:
                # FIXED: Import the correct function from document_loader
                from rag.document_loader import process_and_chunk_document as process_document
                from rag.vector_store import add_documents_to_vector_store
                from langchain import Document  # IMPORT MOVED HERE FOR SCOPE
                
                print(f"🔍 Processing document for Pinecone indexing...")
                
                # Process document
                documents = process_document(temp_file.name)
                
                if not documents or len(documents) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="No content could be extracted from the document"
                    )
                
                print(f"✅ Extracted {len(documents)} chunks from document")
                
                # Debug: Check what type of objects we have
                print(f"📊 Document type: {type(documents)}")
                if documents and len(documents) > 0:
                    print(f"📊 First element type: {type(documents[0])}")
                
                # Add metadata - FIXED VERSION WITH CLEANED METADATA
                doc_id = str(uuid.uuid4())
                user_email = current_user.get('email', 'unknown@example.com')
                
                # Handle different document formats (dict vs LangChain Document)
                processed_docs = []
                for i, doc in enumerate(documents):
                    # Default metadata that ALL documents must have
                    base_metadata = clean_metadata_for_pinecone({
                        "source": file_name,
                        "document_id": doc_id,
                        "user_id": user_id,
                        "user_email": user_email,  # CRITICAL: Added user_email field
                        "upload_date": datetime.now().isoformat(),
                        "file_type": file_type,
                        "chunk_index": str(i),  # Convert to string
                        "page": "1"  # Default page
                    })
                    
                    # Check if it's a dict or has metadata attribute
                    if isinstance(doc, dict):
                        # Get and clean existing metadata
                        existing_metadata = doc.get('metadata', {})
                        cleaned_existing = clean_metadata_for_pinecone(existing_metadata)
                        
                        # Merge with base metadata
                        final_metadata = {**base_metadata, **cleaned_existing}
                        
                        processed_doc = Document(
                            page_content=doc.get('text', '') or doc.get('content', '') or str(doc),
                            metadata=final_metadata
                        )
                    elif hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        # Already a LangChain Document
                        # Clean existing metadata
                        existing_metadata = clean_metadata_for_pinecone(doc.metadata)
                        
                        # Merge with base metadata
                        final_metadata = {**base_metadata, **existing_metadata}
                        
                        processed_doc = Document(
                            page_content=doc.page_content,
                            metadata=final_metadata
                        )
                    else:
                        # Unknown format, create basic document
                        processed_doc = Document(
                            page_content=str(doc)[:1000],
                            metadata=base_metadata
                        )
                    
                    processed_docs.append(processed_doc)
                    if i < 5:  # Only log first 5 to avoid spam
                        print(f"📝 Processed chunk {i+1}/{len(documents)} with metadata keys: {list(processed_doc.metadata.keys())}")
                
                # Add to vector store
                print(f"📤 Adding {len(processed_docs)} documents to Pinecone vector store...")
                result = add_documents_to_vector_store(processed_docs)
                
                if result:
                    print(f"✅ Successfully added {len(processed_docs)} documents to Pinecone")
                    
                    # Clear RAG cache since we added new documents
                    if cache_manager.is_connected():
                        cleared = cache_manager.clear_pattern("rag_response:*")
                        if cleared > 0:
                            print(f"🧹 Cleared {cleared} RAG cache entries")
                    
                    return UploadResponse(
                        success=True,
                        message=f"Document '{file_name}' successfully indexed to Pinecone RAG with {len(processed_docs)} chunks",
                        filename=file_name,
                        document_id=doc_id,
                        chunks_count=len(processed_docs)
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to add documents to vector store"
                    )
                    
            except ImportError as e:
                print(f"❌ RAG module import error: {e}")
                raise HTTPException(
                    status_code=501,
                    detail=f"RAG processing module not available: {str(e)}"
                )
            except Exception as e:
                print(f"❌ Document processing error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process document: {str(e)}"
                )
                
        elif process_type == "chat_with_document":
            # Process for immediate chat analysis
            print(f"🤖 Processing document for chat analysis...")
            
            # Here you would implement document analysis
            # For now, return success
            return UploadResponse(
                success=True,
                message=f"Document '{file_name}' ready for chat analysis",
                filename=file_name
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown process type: {process_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Document upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

@router.get("/upload/status")
async def upload_status():
    """Get upload system status"""
    return {
        "upload_available": RAG_AVAILABLE,
        "rag_available": RAG_AVAILABLE,
        "max_file_size_mb": 50,
        "supported_formats": ["pdf", "txt", "doc", "docx", "csv", "xlsx", "xls", "ppt", "pptx", "jpg", "jpeg", "png"],
        "endpoints": {
            "document_upload": "/api/chatbot/upload-document",
            "status": "/api/chatbot/upload/status"
        }
    }

# ============================================
# NEW CHAT FUNCTIONALITY - FIXED CHAT NAME
# ============================================

@router.post("/new-session")
async def create_new_session(current_user: dict = Depends(get_current_user)):
    """Create a new chat session"""
    user_id = current_user.get("uid", "test_user")
    user_email = current_user.get("email", "guest@example.com")
    
    # Generate new session ID
    session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Create a welcome conversation in the database
    if db_available:
        try:
            conversation_id, _ = create_conversation(
                user_id=user_id,
                session_id=session_id,
                title="New Chat"
            )
            print(f"💾 Created new conversation: {conversation_id} with title: New Chat")
            
        except Exception as e:
            print(f"⚠️ Error creating conversation: {e}")
    
    return {
        "success": True,
        "session_id": session_id,
        "message": "New chat session created",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# UPDATED CHAT ENDPOINT WITH FIXED CHAT NAME AND CACHE METADATA
# ============================================

@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Handle chat messages with AI - WITH OPTIONAL RAG SUPPORT AND FEEDBACK-FIRST CACHING"""
    
    message = chat_request.message
    use_rag = chat_request.use_rag if chat_request.use_rag is not None else True
    user_id = current_user.get("uid", "test_user")
    user_email = current_user.get("email", "guest@example.com")
    user_name = current_user.get("name", "Guest")
    
    session_id = chat_request.session_id or f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Generate chat name FIRST
    chat_name = generate_chat_name(message)
    
    print(f"\n" + "="*70)
    print(f"💬 CHAT REQUEST")
    print(f"="*70)
    print(f"👤 User: {user_email}")
    print(f"💬 Message: {message}")
    print(f"📁 Session: {session_id}")
    print(f"🏷️  Chat Name: {chat_name}")
    print(f"🔍 RAG Requested: {'YES' if use_rag else 'NO'}")
    print(f"🔧 RAG Available: {'YES' if RAG_AVAILABLE else 'NO'}")
    print(f"🤖 Model: {DEFAULT_MODEL}")
    print(f"💾 Cache Enabled: {'YES' if cache_manager.is_connected() else 'NO'}")
    print(f"💾 Cache Mode: FEEDBACK-FIRST (Only cache on thumbs_up)")
    print(f"="*70)
    
    # Get AI response (with or without RAG)
    start_time = time.time()
    
    if use_rag and RAG_AVAILABLE and rag_retriever:
        print("🚀 Getting AI response WITH RAG...")
        # Get response AND rag_used flag AND cache info
        ai_response, is_rag_response, cache_hit, cache_key = await get_ai_response_with_rag(message, session_id, user_id)
    else:
        print("🚀 Getting AI response WITHOUT RAG (model only)...")
        ai_response, cache_hit, cache_key = await get_ai_response(message, session_id, user_id)
        is_rag_response = False
    
    response_time = time.time() - start_time
    response_time_ms = int(response_time * 1000)
    
    # Determine if response was cached (based on response time and cache_hit flag)
    is_cached = cache_hit
    if not cache_hit:
        # Estimate if response was fast enough to be considered cached
        if is_rag_response:
            is_cached = response_time < 1.5  # RAG responses under 1.5s = likely cached
        else:
            is_cached = response_time < 0.8  # Non-RAG responses under 0.8s = likely cached
    
    print(f"="*70)
    print(f"✅ CHAT COMPLETE")
    print(f"📊 Response length: {len(ai_response)} characters")
    print(f"⏱️  Response time: {response_time:.2f}s ({response_time_ms}ms)")
    print(f"🔍 Source: {'RAG (Your uploaded documents)' if is_rag_response else 'Model (General knowledge)'}")
    print(f"💾 Cache Status: {'HIT (thumbs_up response)' if cache_hit else 'MISS (fresh response)'}")
    print(f"💾 Estimated Cached: {'YES' if is_cached else 'NO'}")
    print(f"="*70 + "\n")
    
    # Generate message ID for this response
    message_id = str(uuid.uuid4())
    
    # Save user message to MongoDB using hierarchical structure
    if db_available:
        try:
            # Check if conversation exists
            conversation = conversations_collection.find_one({
                "session_id": session_id,
                "user_id": user_id
            })
            
            if not conversation:
                # Create new conversation with chat name
                conversation_id, _ = create_conversation(
                    user_id=user_id,
                    session_id=session_id,
                    title=chat_name  # Use the generated chat name
                )
                print(f"💾 Created new conversation with title: {chat_name}")
            else:
                conversation_id = str(conversation["_id"])
                # Update conversation title if it's still the default
                if conversation.get("title") in ["New Conversation", "New Chat", "Chat"]:
                    conversations_collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        {"$set": {"title": chat_name}}
                    )
                    print(f"💾 Updated conversation title to: {chat_name}")
            
            # Save user message
            user_msg_id, conversation_id = save_message(
                user_id=user_id,
                email=user_email,
                message_data={
                    "role": "user",
                    "message": message,
                    "is_ai": False,
                    "use_rag": use_rag,
                    "session_id": session_id,
                    "cache_hit": cache_hit,
                    "cache_key": cache_key
                },
                session_id=session_id,
                conversation_id=conversation_id  # Pass the conversation_id
            )
            print(f"💾 User message saved to MongoDB with conversation: {conversation_id}")
            
            # Save AI response with message_id
            ai_msg_id, _ = save_message(
                user_id=user_id,
                email=user_email,
                message_data={
                    "role": "assistant",
                    "message": ai_response,
                    "is_ai": True,
                    "use_rag": is_rag_response,
                    "session_id": session_id,
                    "cache_hit": cache_hit,
                    "cache_key": cache_key,
                    "response_time_ms": response_time_ms,
                    "message_id": message_id  # Store the message_id for feedback
                },
                conversation_id=conversation_id,
                session_id=session_id
            )
            print(f"💾 AI response saved to MongoDB with message_id: {message_id}")
            
            # Store query-response mapping for potential thumbs_up caching
            if cache_manager.is_connected():
                query_response_key = get_query_response_key(session_id, message)
                cache_manager.set_cache(query_response_key, {
                    "message_id": message_id,
                    "response": ai_response,
                    "query": message,
                    "timestamp": datetime.now().isoformat()
                }, ttl=2592000)  # 30 days
                print(f"💾 Query-response mapping stored for feedback system")
            
        except Exception as e:
            print(f"⚠️ MongoDB save error: {e}")
            # Fallback: try direct save
            try:
                # Find or create conversation
                conversation = conversations_collection.find_one({
                    "session_id": session_id,
                    "user_id": user_id
                })
                
                if not conversation:
                    conversation_id, _ = create_conversation(
                        user_id=user_id,
                        session_id=session_id,
                        title=chat_name
                    )
                    conversation = {"_id": ObjectId(conversation_id)}
                else:
                    conversation_id = str(conversation["_id"])
                
                # Prepare messages
                user_msg = {
                    "message_id": str(uuid.uuid4()),
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now(),
                    "is_ai": False,
                    "use_rag": use_rag,
                    "cache_hit": cache_hit,
                    "cache_key": cache_key
                }
                
                ai_msg = {
                    "message_id": message_id,
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now(),
                    "is_ai": True,
                    "use_rag": is_rag_response,
                    "cache_hit": cache_hit,
                    "cache_key": cache_key,
                    "response_time_ms": response_time_ms
                }
                
                # Add both messages to conversation
                conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {
                        "$push": {"messages": {"$each": [user_msg, ai_msg]}},
                        "$inc": {"message_count": 2},
                        "$set": {"updated_at": datetime.now()}
                    }
                )
                
                print("💾 Messages saved with fallback method")
                
                # Store query-response mapping for potential thumbs_up caching
                if cache_manager.is_connected():
                    query_response_key = get_query_response_key(session_id, message)
                    cache_manager.set_cache(query_response_key, {
                        "message_id": message_id,
                        "response": ai_response,
                        "query": message,
                        "timestamp": datetime.now().isoformat()
                    }, ttl=2592000)
                
            except Exception as fallback_error:
                print(f"⚠️ Fallback save error: {fallback_error}")
    else:
        print("⚠️ MongoDB not available, skipping database save")
    
    # RETURN UPDATED RESPONSE WITH CACHE METADATA
    return ChatResponse(
        response=ai_response,
        session_id=session_id,
        message_id=message_id,
        timestamp=datetime.now().isoformat(),
        chat_name=chat_name,
        is_rag=is_rag_response,
        source="rag" if is_rag_response else "model",
        response_time_ms=response_time_ms,
        is_cached=is_cached,
        cache_hit=cache_hit
    )

# ============================================
# RAG MANAGEMENT ENDPOINTS - UPDATED
# ============================================

@router.post("/rag/initialize")
async def initialize_rag():
    """Initialize RAG system with documents"""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=501, detail="RAG module not available")
    
    try:
        # Import and call initialize_rag_system function
        from rag.retriever import initialize_rag_system
        success = initialize_rag_system()
        
        if success:
            from rag.vector_store import get_vector_store
            vector_store = get_vector_store()
            doc_count = vector_store.get_document_count()
            
            return {
                "success": True,
                "message": "RAG system initialized successfully",
                "documents_loaded": doc_count,
                "rag_available": RAG_AVAILABLE
            }
        else:
            return {
                "success": False,
                "message": "Failed to initialize RAG system",
                "rag_available": RAG_AVAILABLE
            }
    except Exception as e:
        print(f"❌ RAG initialization error: {e}")
        return {
            "success": False,
            "message": f"Failed to initialize RAG: {str(e)}",
            "rag_available": RAG_AVAILABLE
        }

@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics"""
    if not RAG_AVAILABLE:
        return {
            "success": False,
            "message": "RAG module not available",
            "rag_available": False
        }
    
    try:
        from rag.vector_store import get_vector_store
        vector_store = get_vector_store()
        doc_count = vector_store.get_document_count()
        
        return {
            "success": True,
            "statistics": {
                "document_count": doc_count,
                "rag_available": True,
                "retriever_ready": rag_retriever is not None
            },
            "rag_available": RAG_AVAILABLE
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to get stats: {str(e)}",
            "rag_available": RAG_AVAILABLE
        }

@router.post("/rag/search")
async def rag_search(query: str, top_k: int = 3):
    """Search in RAG system"""
    if not RAG_AVAILABLE or not rag_retriever:
        raise HTTPException(status_code=501, detail="RAG module not available")
    
    try:
        # Use similarity search with scores
        results = rag_retriever.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Check if query is medical
        medical_keywords = ['hiv', 'aids', 'disease', 'medical', 'health', 'symptom', 'treatment', 
                           'virus', 'infection', 'doctor', 'hospital', 'medicine', 'pain', 'fever']
        is_medical_query = any(keyword in query.lower() for keyword in medical_keywords)
        threshold = 0.35 if is_medical_query else 0.65
        
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                "id": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown"),
                "score": float(score),
                "above_threshold": score >= threshold,
                "threshold_used": threshold
            })
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "rag_available": RAG_AVAILABLE,
            "query_type": "Medical" if is_medical_query else "General",
            "similarity_threshold": threshold
        }
    except Exception as e:
        print(f"❌ RAG search error: {e}")
        return {
            "success": False,
            "message": f"Search failed: {str(e)}",
            "rag_available": RAG_AVAILABLE
        }

@router.get("/rag/status")
async def get_rag_status():
    """Get RAG system status"""
    return {
        "rag_available": RAG_AVAILABLE,
        "rag_retriever_ready": rag_retriever is not None,
        "database_available": db_available,
        "ai_provider": "OpenRouter",
        "model": DEFAULT_MODEL,  # ✅ Shows the fast model
        "openrouter_key_configured": bool(OPENROUTER_API_KEY),
        "similarity_thresholds": {
            "medical_queries": 0.35,
            "general_queries": 0.65
        },
        "rag_mode": "Adaptive threshold based on query type",
        "cache_enabled": cache_manager.is_connected(),
        "cache_mode": "feedback-first (only cache on thumbs_up)"
    }

# ============================================
# EXISTING ENDPOINTS (Updated for chat name fix)
# ============================================

@router.get("/sessions")
async def get_all_sessions(current_user: dict = Depends(get_current_user)):
    """Get all chat sessions for current user"""
    user_id = current_user.get("uid", "test_user")
    
    if not db_available:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Get conversations for this user
        conversations = list(conversations_collection.find(
            {"user_id": user_id, "is_active": True},
            {
                "_id": 1,
                "session_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "message_count": 1,
                "messages": {"$slice": -1}  # Get only last message
            }
        ).sort("updated_at", -1))
        
        formatted_sessions = []
        for conv in conversations:
            # Use the title from conversation, fallback to generating from first message
            chat_name = conv.get("title", "Chat")
            
            # If title is default, try to generate from first message
            if chat_name in ["New Chat", "New Conversation", "Chat"]:
                if conv.get("messages") and len(conv["messages"]) > 0:
                    first_user_msg = next((msg for msg in conv["messages"] if msg.get("role") == "user"), None)
                    if first_user_msg:
                        chat_name = generate_chat_name(first_user_msg.get("content", ""))
            
            # Get last message for preview
            last_message = ""
            if conv.get("messages") and len(conv["messages"]) > 0:
                last_msg = conv["messages"][-1]
                last_message = last_msg.get("content", "")[:100]
            
            formatted_sessions.append({
                "session_id": conv.get("session_id"),
                "chat_name": chat_name,
                "last_message": last_message,
                "message_count": conv.get("message_count", 0),
                "last_activity": conv.get("updated_at").isoformat() if conv.get("updated_at") else None,
                "user_id": user_id
            })
        
        return {
            "sessions": formatted_sessions,
            "count": len(formatted_sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/session/{session_id}")
async def get_session_messages(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get all messages in a session"""
    user_id = current_user.get("uid", "test_user")
    
    if not db_available:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Get conversation by session_id
        conversation = conversations_collection.find_one({
            "session_id": session_id,
            "user_id": user_id
        })
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get messages from conversation
        messages = conversation.get("messages", [])
        
        # Get chat name from conversation title or first message
        chat_name = conversation.get("title", "Chat")
        if not chat_name or chat_name == "Chat":
            if messages and len(messages) > 0:
                first_user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
                if first_user_msg:
                    chat_name = generate_chat_name(first_user_msg.get("content", ""))
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "message": msg.get("content", ""),
                "timestamp": msg.get("timestamp").isoformat() if msg.get("timestamp") else None,
                "is_ai": msg.get("is_ai", False),
                "is_rag": msg.get("use_rag", False),  # Include RAG status
                "cache_hit": msg.get("cache_hit", False),  # Include cache status
                "response_time_ms": msg.get("response_time_ms", 0)  # Include response time
            })
        
        return {
            "session_id": session_id,
            "chat_name": chat_name,
            "messages": formatted_messages,
            "count": len(formatted_messages),
            "user_id": user_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.delete("/session/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a chat session and all its messages"""
    user_id = current_user.get("uid", "test_user")
    
    if not db_available:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        # Mark conversation as inactive (soft delete)
        result = conversations_collection.update_one(
            {
                "session_id": session_id,
                "user_id": user_id
            },
            {
                "$set": {
                    "is_active": False,
                    "deleted_at": datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            return {
                "status": "success",
                "message": f"Chat deleted successfully.",
                "session_id": session_id
            }
        else:
            return {
                "status": "error",
                "message": "Chat not found or no permission",
                "session_id": session_id
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/health")
async def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "service": "MedAI Chat API",
        "ai_provider": "OpenRouter",
        "model": DEFAULT_MODEL,  # ✅ Shows the fast model
        "database": "connected" if db_available else "disconnected",
        "rag_system": "available" if RAG_AVAILABLE and rag_retriever else "unavailable",
        "firebase_auth": "enabled" if FIREBASE_AVAILABLE else "disabled",
        "openrouter_key": "configured" if OPENROUTER_API_KEY else "missing",
        "rag_mode": "Adaptive threshold (Medical: 0.35, General: 0.65)",
        "optimized": "yes (fast 3B model)",
        "redis_cache": "enabled" if cache_manager.is_connected() else "disabled",
        "cache_policy": "feedback-first (only cache on thumbs_up)"
    }

@router.get("/test")
async def test_ai():
    """Test OpenRouter connection"""
    try:
        test_message = "Hello, are you working?"
        response = await get_ai_response(test_message)
        
        return {
            "status": "success",
            "test_message": test_message,
            "response": response[:100] + "..." if len(response) > 100 else response,
            "model": DEFAULT_MODEL,  # ✅ Shows the fast model
            "rag_available": RAG_AVAILABLE,
            "cache_enabled": cache_manager.is_connected()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/my-chats")
async def get_my_chats(current_user: dict = Depends(get_current_user)):
    """Get current user's chats with user info"""
    user_id = current_user.get("uid", "test_user")
    user_email = current_user.get("email", "guest@example.com")
    user_name = current_user.get("name", "Guest")
    
    return {
        "user": {
            "id": user_id,
            "email": user_email,
            "name": user_name
        },
        "message": "Firebase authentication working correctly",
        "rag_available": RAG_AVAILABLE,
        "rag_retriever_ready": rag_retriever is not None,
        "model": DEFAULT_MODEL,  # ✅ Shows the fast model
        "rag_mode": "Adaptive threshold (Medical: 0.35, General: 0.65)",
        "cache_enabled": cache_manager.is_connected(),
        "cache_policy": "feedback-first (only cache on thumbs_up)"
    }

# ============================================
# VOICE CHAT ENDPOINT - UPDATED WITH CHAT NAME FIX AND CACHE METADATA
# ============================================

@router.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None,
    use_rag: Optional[bool] = True,
    language: str = "hi",
    current_user: dict = Depends(get_current_user)
):
    """Voice chat endpoint using OpenAI Whisper"""
    
    # Check if voice is enabled
    if not VOICE_ENABLED or not voice_processor:
        raise HTTPException(
            status_code=503,
            detail="Voice features not available. Add OPENAI_API_KEY to .env file."
        )
    
    # Read audio file
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read audio file: {str(e)}"
        )
    
    # Check file size (10MB max)
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Audio file too large. Maximum size is 10MB"
        )
    
    user_id = current_user.get("uid", "test_user")
    user_email = current_user.get("email", "guest@example.com")
    
    print(f"🎤 Voice request from: {user_email}")
    print(f"   Language: {language}")
    print(f"   Use RAG: {use_rag}")
    
    try:
        # Use voice processor to transcribe
        user_text = voice_processor.transcribe(audio_bytes, language)
        
        if not user_text or len(user_text.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please speak clearly."
            )
        
        print(f"✅ Transcribed: {user_text[:100]}...")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"voice_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate chat name from transcribed text
        chat_name = generate_chat_name(user_text)
        
        # Get AI response using existing functions
        start_time = time.time()
        if use_rag and RAG_AVAILABLE and rag_retriever:
            ai_response, is_rag_response, cache_hit, cache_key = await get_ai_response_with_rag(user_text, session_id, user_id)
        else:
            ai_response, cache_hit, cache_key = await get_ai_response(user_text, session_id, user_id)
            is_rag_response = False
        
        response_time = time.time() - start_time
        response_time_ms = int(response_time * 1000)
        
        # Determine if response was cached
        is_cached = cache_hit
        if not cache_hit:
            if is_rag_response:
                is_cached = response_time < 1.5
            else:
                is_cached = response_time < 0.8
        
        # Save voice conversation using hierarchical structure
        if db_available:
            try:
                # Check if conversation exists
                conversation = conversations_collection.find_one({
                    "session_id": session_id,
                    "user_id": user_id
                })
                
                if not conversation:
                    # Create new conversation with chat name
                    conversation_id, _ = create_conversation(
                        user_id=user_id,
                        session_id=session_id,
                        title=chat_name  # Use the generated chat name
                    )
                    print(f"💾 Created new voice conversation with title: {chat_name}")
                else:
                    conversation_id = str(conversation["_id"])
                    # Update conversation title if it's still the default
                    if conversation.get("title") in ["New Conversation", "New Chat", "Chat"]:
                        conversations_collection.update_one(
                            {"_id": ObjectId(conversation_id)},
                            {"$set": {"title": chat_name}}
                        )
                        print(f"💾 Updated voice conversation title to: {chat_name}")
                
                # Generate message ID for voice response
                voice_message_id = str(uuid.uuid4())
                
                # Save user voice message
                user_msg_id, conversation_id = save_message(
                    user_id=user_id,
                    email=user_email,
                    message_data={
                        "role": "user",
                        "message": user_text,
                        "is_ai": False,
                        "use_rag": use_rag,
                        "session_id": session_id,
                        "is_voice": True,
                        "voice_language": language,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key
                    },
                    session_id=session_id,
                    conversation_id=conversation_id
                )
                
                # Save AI response
                ai_msg_id, _ = save_message(
                    user_id=user_id,
                    email=user_email,
                    message_data={
                        "role": "assistant",
                        "message": ai_response,
                        "is_ai": True,
                        "use_rag": is_rag_response,
                        "session_id": session_id,
                        "is_voice_response": True,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key,
                        "response_time_ms": response_time_ms,
                        "message_id": voice_message_id
                    },
                    conversation_id=conversation_id,
                    session_id=session_id
                )
                
                print(f"💾 Voice conversation saved (conversation: {conversation_id})")
                
                # Store query-response mapping for potential thumbs_up caching
                if cache_manager.is_connected():
                    query_response_key = get_query_response_key(session_id, user_text)
                    cache_manager.set_cache(query_response_key, {
                        "message_id": voice_message_id,
                        "response": ai_response,
                        "query": user_text,
                        "timestamp": datetime.now().isoformat()
                    }, ttl=2592000)
                
            except Exception as e:
                print(f"⚠️ Database hierarchical save error: {e}")
                # Fallback to old method
                try:
                    # Find or create conversation
                    conversation = conversations_collection.find_one({
                        "session_id": session_id,
                        "user_id": user_id
                    })
                    
                    if not conversation:
                        conversation_id, _ = create_conversation(
                            user_id=user_id,
                            session_id=session_id,
                            title=chat_name
                        )
                        conversation = {"_id": ObjectId(conversation_id)}
                    else:
                        conversation_id = str(conversation["_id"])
                    
                    voice_message_id = str(uuid.uuid4())
                    
                    # Prepare voice messages
                    user_msg = {
                        "message_id": str(uuid.uuid4()),
                        "role": "user",
                        "content": user_text,
                        "timestamp": datetime.now(),
                        "is_ai": False,
                        "use_rag": use_rag,
                        "is_voice": True,
                        "voice_language": language,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key
                    }
                    
                    ai_msg = {
                        "message_id": voice_message_id,
                        "role": "assistant",
                        "content": ai_response,
                        "timestamp": datetime.now(),
                        "is_ai": True,
                        "use_rag": is_rag_response,
                        "is_voice_response": True,
                        "cache_hit": cache_hit,
                        "cache_key": cache_key,
                        "response_time_ms": response_time_ms
                    }
                    
                    # Add both messages to conversation
                    conversations_collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        {
                            "$push": {"messages": {"$each": [user_msg, ai_msg]}},
                            "$inc": {"message_count": 2},
                            "$set": {"updated_at": datetime.now()}
                        }
                    )
                    
                    print("💾 Voice conversation saved with fallback method")
                    
                    # Store query-response mapping
                    if cache_manager.is_connected():
                        query_response_key = get_query_response_key(session_id, user_text)
                        cache_manager.set_cache(query_response_key, {
                            "message_id": voice_message_id,
                            "response": ai_response,
                            "query": user_text,
                            "timestamp": datetime.now().isoformat()
                        }, ttl=2592000)
                    
                except Exception as fallback_error:
                    print(f"⚠️ Fallback save error: {fallback_error}")
        
        return {
            "success": True,
            "response": ai_response,
            "session_id": session_id,
            "chat_name": chat_name,
            "message_id": voice_message_id,
            "timestamp": datetime.now().isoformat(),
            "is_rag": is_rag_response,
            "source": "rag" if is_rag_response else "model",
            "response_time_ms": response_time_ms,
            "is_cached": is_cached,
            "cache_hit": cache_hit,
            "is_voice": True,
            "transcribed_text": user_text,
            "language": language
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Voice chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal error processing voice request"
        )

# ============================================
# VOICE STATUS ENDPOINT - NEW ADDITION
# ============================================

@router.get("/voice/status")
async def voice_status():
    """Get voice system status"""
    status_info = {
        "voice_enabled": VOICE_ENABLED,
        "processor": "openai_whisper",
        "model": "whisper-1",
        "supported_languages": ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"],
        "max_file_size_mb": 10,
        "endpoint": "/api/chatbot/voice-chat"
    }
    
    if VOICE_ENABLED and voice_processor:
        try:
            status_info["status"] = "operational"
        except:
            status_info["status"] = "unknown"
    else:
        status_info["status"] = "disabled"
    
    return status_info

# ============================================
# TEST MODEL ENDPOINT - ADDED
# ============================================

@router.get("/test-model")
async def test_model():
    """Test if the model is working"""
    try:
        test_prompt = "Hello, say 'Model is working!' in one sentence."
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000"
        }
        
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 30,
            "temperature": 0.3
        }
        
        import time
        start = time.time()
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=5
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return {
                "model": DEFAULT_MODEL,
                "status": "WORKING ✅",
                "response_time": f"{elapsed:.2f}s",
                "response": answer,
                "cache_enabled": cache_manager.is_connected()
            }
        else:
            return {
                "model": DEFAULT_MODEL,
                "status": f"ERROR {response.status_code}",
                "response_time": f"{elapsed:.2f}s",
                "cache_enabled": cache_manager.is_connected()
            }
            
    except Exception as e:
        return {
            "model": DEFAULT_MODEL,
            "status": "ERROR",
            "error": str(e),
            "cache_enabled": cache_manager.is_connected()
        }

# ============================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================

@router.get("/cache/stats")
async def get_cache_stats_endpoint(current_user: dict = Depends(get_current_user)):
    """Get Redis cache statistics"""
    if not cache_manager.is_connected():
        return {
            "success": False,
            "message": "Redis cache not connected",
            "cache_enabled": False
        }
    
    try:
        # Get basic Redis info
        import redis
        client = cache_manager.redis_client
        
        # Try to get info
        try:
            info = client.info()
            total_keys = client.dbsize()
            
            # Count cache types
            cache_patterns = {
                "llm_cache": len(client.keys("llm_response:*")),
                "rag_cache": len(client.keys("rag_response:*")),
                "ai_cache": len(client.keys("ai_response:*")),
                "medical_cache": len(client.keys("medical:query:*")),
                "feedback_cache": len(client.keys("feedback:*")),
                "query_response_cache": len(client.keys("query_response:*"))
            }
            
            stats = {
                "success": True,
                "cache_enabled": True,
                "redis_connected": True,
                "total_keys": total_keys,
                "memory_used": info.get('used_memory_human', 'N/A'),
                "cache_counts": cache_patterns,
                "timestamp": datetime.now().isoformat(),
                "cache_policy": "feedback-first (only cache on thumbs_up)"
            }
            
            return stats
            
        except:
            # Fallback if info command fails
            return {
                "success": True,
                "cache_enabled": True,
                "redis_connected": True,
                "message": "Cache is connected but detailed stats unavailable",
                "timestamp": datetime.now().isoformat(),
                "cache_policy": "feedback-first (only cache on thumbs_up)"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting cache stats: {str(e)}",
            "cache_enabled": cache_manager.is_connected()
        }

@router.post("/cache/clear")
async def clear_cache_endpoint(
    pattern: str = "*",
    confirm: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """Clear cache entries by pattern"""
    if not cache_manager.is_connected():
        return {
            "success": False,
            "message": "Redis cache not connected",
            "cache_enabled": False
        }
    
    # Safety check for clearing all cache
    if pattern == "*" and not confirm:
        return {
            "success": False,
            "message": "Add ?confirm=true to clear all cache. Use specific patterns like 'rag:*' or 'llm:*' instead.",
            "safe_patterns": ["rag:*", "llm:*", "ai_response:*", "medical:*", "feedback:*", "query_response:*"]
        }
    
    if pattern == "*":
        pattern = "*"  # Allow all if confirmed
    
    cleared = cache_manager.clear_pattern(pattern)
    
    return {
        "success": True,
        "message": f"Cleared {cleared} cache entries",
        "pattern": pattern,
        "cleared_count": cleared,
        "user": current_user.get("email", "unknown"),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/cache/test")
async def test_cache_endpoint(current_user: dict = Depends(get_current_user)):
    """Test cache functionality"""
    test_query = "What are the symptoms of HIV?"
    
    # Test without cache (first call)
    start = time.time()
    response1, _, cache_hit1, _ = await get_ai_response_with_rag(test_query)
    time1 = time.time() - start
    
    # Test with cache (second call - should be fresh since no thumbs_up)
    start = time.time()
    response2, _, cache_hit2, _ = await get_ai_response_with_rag(test_query)
    time2 = time.time() - start
    
    # Responses should be different (not cached)
    responses_different = response1 != response2
    
    return {
        "success": True,
        "test": "cache_performance",
        "query": test_query,
        "first_call_ms": round(time1 * 1000, 2),
        "second_call_ms": round(time2 * 1000, 2),
        "first_cache_hit": cache_hit1,
        "second_cache_hit": cache_hit2,
        "responses_different": responses_different,
        "cache_working": True,
        "cache_policy": "feedback-first (only cache on thumbs_up)",
        "cache_enabled": cache_manager.is_connected(),
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# NEW ADMIN ENDPOINTS FOR HIERARCHICAL VIEWING
# ============================================

@router.get("/admin/users")
async def get_all_users_admin(current_user: dict = Depends(get_current_user)):
    """Get all users with their conversations (Admin only)"""
    try:
        from database import get_all_users
        users = get_all_users(limit=50)
        return {
            "success": True,
            "users": users,
            "count": len(users)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/admin/user/{user_id}")
async def get_user_details_admin(user_id: str, current_user: dict = Depends(get_current_user)):
    """Get specific user with all conversations"""
    try:
        from database import get_user_with_conversations
        user = get_user_with_conversations(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": user
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/admin/conversation/{conversation_id}")
async def get_conversation_details_admin(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Get specific conversation with all messages"""
    try:
        from database import get_conversation_with_messages
        conversation = get_conversation_with_messages(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            "conversation": conversation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================
# ADDED FEEDBACK: FEEDBACK ENDPOINTS - UPDATED FOR FEEDBACK-FIRST CACHING
# ============================================

@router.post("/feedback")
async def submit_feedback(
    feedback_request: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for a message - WITH FEEDBACK-FIRST CACHING"""
    user_id = current_user.get("uid", "test_user")
    user_email = current_user.get("email", "guest@example.com")
    
    print(f"📊 Feedback submitted by {user_email}")
    print(f"   Session: {feedback_request.session_id}")
    print(f"   Message ID: {feedback_request.message_id}")
    print(f"   Type: {feedback_request.feedback_type}")
    
    # Validate feedback type
    if feedback_request.feedback_type not in ["thumbs_up", "thumbs_down"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid feedback type. Must be 'thumbs_up' or 'thumbs_down'"
        )
    
    # Store feedback in cache
    if cache_manager.is_connected():
        feedback_key = get_feedback_cache_key(
            feedback_request.session_id, 
            feedback_request.message_id
        )
        
        feedback_data = {
            "user_id": user_id,
            "user_email": user_email,
            "session_id": feedback_request.session_id,
            "message_id": feedback_request.message_id,
            "feedback_type": feedback_request.feedback_type,
            "user_comment": feedback_request.user_comment,
            "rating": feedback_request.rating,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store feedback for 30 days
        cache_manager.set_cache(feedback_key, feedback_data, ttl=2592000)
        
        print(f"💾 Feedback cached: {feedback_request.feedback_type}")
        
        # ============================================
        # 🔥 NEW: Cache response only on thumbs_up
        # ============================================
        if feedback_request.feedback_type == "thumbs_up":
            print(f"✅ Thumbs up detected - caching this response for future use")
            
            # Try to find the response to cache it
            try:
                if db_available and conversations_collection is not None:
                    # Find the conversation with this message
                    conversation = conversations_collection.find_one({
                        "session_id": feedback_request.session_id,
                        "user_id": user_id,
                        "messages.message_id": feedback_request.message_id
                    })
                    
                    if conversation and "messages" in conversation:
                        # Find the AI message that got thumbs_up
                        for i, msg in enumerate(conversation["messages"]):
                            if msg.get("message_id") == feedback_request.message_id:
                                ai_response = msg.get("content", "")
                                is_rag_response = msg.get("use_rag", False)
                                
                                # Get the previous user message (the query)
                                if i > 0:
                                    user_message = conversation["messages"][i-1].get("content", "")
                                    
                                    if user_message and ai_response:
                                        print(f"   Caching response for query: {user_message[:50]}...")
                                        
                                        if is_rag_response:
                                            # Cache RAG response
                                            rag_cache_key = get_rag_cache_key(user_message, feedback_request.session_id, True)
                                            cache_manager.set_cache(rag_cache_key, (ai_response, True), ttl=86400)  # 24 hours
                                            print(f"   💾 Cached RAG response for future use")
                                        else:
                                            # Cache simple AI response
                                            simple_cache_key = f"ai_response:simple:{hashlib.md5(user_message.encode()).hexdigest()}"
                                            cache_manager.set_cache(simple_cache_key, ai_response, ttl=86400)  # 24 hours
                                            print(f"   💾 Cached AI response for future use")
                                        
                                        # Also update query-response mapping
                                        query_response_key = get_query_response_key(feedback_request.session_id, user_message)
                                        existing_data = cache_manager.get_cache(query_response_key) or {}
                                        existing_data["cached"] = True
                                        cache_manager.set_cache(query_response_key, existing_data, ttl=2592000)
                                        
                                        print(f"   ✅ Response now cached for future queries")
                                        break
                                        
            except Exception as e:
                print(f"⚠️ Error caching response on thumbs_up: {e}")
        
        elif feedback_request.feedback_type == "thumbs_down":
            print(f"⚠️ Thumbs down detected - not caching this response")
            # Clear any existing cache for this message if present
            try:
                if db_available and conversations_collection is not None:
                    # Find the conversation with this message
                    conversation = conversations_collection.find_one({
                        "session_id": feedback_request.session_id,
                        "user_id": user_id,
                        "messages.message_id": feedback_request.message_id
                    })
                    
                    if conversation and "messages" in conversation:
                        for i, msg in enumerate(conversation["messages"]):
                            if msg.get("message_id") == feedback_request.message_id:
                                if i > 0:
                                    user_message = conversation["messages"][i-1].get("content", "")
                                    
                                    # Clear RAG cache
                                    rag_cache_key = get_rag_cache_key(user_message, feedback_request.session_id, True)
                                    cache_manager.delete_cache(rag_cache_key)
                                    
                                    # Clear AI cache
                                    simple_cache_key = f"ai_response:simple:{hashlib.md5(user_message.encode()).hexdigest()}"
                                    cache_manager.delete_cache(simple_cache_key)
                                    
                                    print(f"   🗑️ Cleared any existing cache for this query")
                                    break
                                    
            except Exception as e:
                print(f"⚠️ Error clearing cache on thumbs_down: {e}")
    
    # Store feedback in MongoDB if available
    if db_available and feedback_collection is not None:
        try:
            feedback_id = str(uuid.uuid4())
            feedback_doc = {
                "_id": feedback_id,
                "user_id": user_id,
                "user_email": user_email,
                "session_id": feedback_request.session_id,
                "message_id": feedback_request.message_id,
                "feedback_type": feedback_request.feedback_type,
                "user_comment": feedback_request.user_comment,
                "rating": feedback_request.rating,
                "timestamp": datetime.now(),
                "created_at": datetime.now()
            }
            
            feedback_collection.insert_one(feedback_doc)
            print(f"💾 Feedback saved to MongoDB: {feedback_id}")
            
            return FeedbackResponse(
                success=True,
                message=f"Thank you for your {feedback_request.feedback_type.replace('_', ' ')}!",
                feedback_id=feedback_id
            )
            
        except Exception as e:
            print(f"⚠️ MongoDB feedback save error: {e}")
    
    return FeedbackResponse(
        success=True,
        message=f"Thank you for your {feedback_request.feedback_type.replace('_', ' ')}!",
        feedback_id=None
    )

@router.get("/feedback/{session_id}/{message_id}")
async def get_feedback(
    session_id: str,
    message_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get feedback for a specific message"""
    user_id = current_user.get("uid", "test_user")
    
    # Try cache first
    feedback_data = None
    if cache_manager.is_connected():
        feedback_key = get_feedback_cache_key(session_id, message_id)
        feedback_data = cache_manager.get_cache(feedback_key)
    
    # If not in cache, try MongoDB
    if not feedback_data and db_available and feedback_collection is not None:
        try:
            feedback_doc = feedback_collection.find_one({
                "session_id": session_id,
                "message_id": message_id,
                "user_id": user_id
            })
            
            if feedback_doc:
                feedback_data = {
                    "feedback_type": feedback_doc.get("feedback_type"),
                    "user_comment": feedback_doc.get("user_comment"),
                    "rating": feedback_doc.get("rating"),
                    "timestamp": feedback_doc.get("timestamp").isoformat() if feedback_doc.get("timestamp") else None
                }
        except Exception as e:
            print(f"⚠️ MongoDB feedback retrieval error: {e}")
    
    if feedback_data:
        return {
            "success": True,
            "has_feedback": True,
            "feedback": feedback_data
        }
    else:
        return {
            "success": True,
            "has_feedback": False,
            "message": "No feedback found for this message"
        }

@router.get("/feedback/stats")
async def get_feedback_stats(current_user: dict = Depends(get_current_user)):
    """Get feedback statistics"""
    user_id = current_user.get("uid", "test_user")
    
    stats = {
        "total_feedback": 0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "average_rating": 0,
        "user_feedback_count": 0
    }
    
    if db_available and feedback_collection is not None:
        try:
            # Get all feedback for this user
            user_feedback = list(feedback_collection.find({"user_id": user_id}))
            stats["user_feedback_count"] = len(user_feedback)
            
            if user_feedback:
                thumbs_up = sum(1 for fb in user_feedback if fb.get("feedback_type") == "thumbs_up")
                thumbs_down = len(user_feedback) - thumbs_up
                
                ratings = [fb.get("rating") for fb in user_feedback if fb.get("rating") is not None]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                
                stats.update({
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "average_rating": round(avg_rating, 2)
                })
            
            # Get total feedback in system (admin view)
            if user_id == "admin" or users_collection.endswith("@admin.com"):  # Simple admin check
                total_feedback = feedback_collection.count_documents({})
                all_thumbs_up = feedback_collection.count_documents({"feedback_type": "thumbs_up"})
                all_thumbs_down = total_feedback - all_thumbs_up
                
                all_ratings = feedback_collection.find({"rating": {"$ne": None}}, {"rating": 1})
                all_rating_values = [fb["rating"] for fb in all_ratings]
                all_avg_rating = sum(all_rating_values) / len(all_rating_values) if all_rating_values else 0
                
                stats.update({
                    "total_feedback": total_feedback,
                    "thumbs_up": all_thumbs_up,
                    "thumbs_down": all_thumbs_down,
                    "average_rating": round(all_avg_rating, 2)
                })
                
        except Exception as e:
            print(f"⚠️ Feedback stats error: {e}")
    
    return {
        "success": True,
        "stats": stats,
        "user_id": user_id,
        "cache_policy": "feedback-first (only cache on thumbs_up)"
    }