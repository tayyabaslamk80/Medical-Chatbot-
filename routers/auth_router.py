# routers/auth_router.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import hashlib
import os

# Import database - FIXED: No circular import
try:
    from database import users_collection
    print("✅ Auth router: MongoDB users_collection imported")
except ImportError as e:
    print(f"⚠️ Auth router import error: {e}")
    # Create dummy collection for fallback
    class DummyCollection:
        def find_one(self, *args, **kwargs): return None
        def insert_one(self, data):
            class Result: 
                inserted_id = "dummy_id"
            return Result()
        def update_one(self, *args, **kwargs):
            class Result:
                matched_count = 0
                modified_count = 0
            return Result()
    
    users_collection = DummyCollection()

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request models
class UserSignup(BaseModel):
    username: str
    email: str
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class TokenVerify(BaseModel):
    token: str

# Helper function to hash password
def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + pwdhash.hex()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    try:
        salt = bytes.fromhex(stored_password[:64])
        stored_hash = stored_password[64:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwdhash.hex() == stored_hash
    except:
        return False

def check_mongodb_connection():
    """Check if MongoDB collections are available"""
    try:
        # Check if it's a real MongoDB collection or dummy
        if hasattr(users_collection, 'database'):
            return True
        else:
            return False
    except:
        return False

@router.post("/signup")
async def signup(user: UserSignup):
    """Sign up a new user"""
    try:
        print(f"📝 Signup attempt for: {user.email}")
        
        # Check MongoDB connection
        if not check_mongodb_connection():
            raise HTTPException(status_code=503, detail="Database not available. Please try again later.")
        
        # Check if user already exists
        existing_user = users_collection.find_one({
            "$or": [
                {"email": user.email},
                {"username": user.username}
            ]
        })
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user data
        user_data = {
            "username": user.username,
            "email": user.email,
            "name": user.name or user.username,
            "password_hash": hash_password(user.password),
            "auth_provider": "custom",
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user into MongoDB
        result = users_collection.insert_one(user_data)
        user_id = str(result.inserted_id)
        
        # Create a simple token
        token = f"custom_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        return JSONResponse({
            "success": True,
            "message": "User created successfully",
            "user": {
                "id": user_id,
                "email": user.email,
                "name": user.name or user.username
            },
            "token": token,
            "database_connected": True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Signup error: {e}")
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@router.post("/login")
async def login(user: UserLogin):
    """Log in a user"""
    try:
        print(f"🔑 Login attempt for: {user.email}")
        
        # Check MongoDB connection
        if not check_mongodb_connection():
            raise HTTPException(status_code=503, detail="Database not available. Please try again later.")
        
        # Find user by email
        db_user = users_collection.find_one({"email": user.email})
        
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password
        if not verify_password(db_user["password_hash"], user.password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        users_collection.update_one(
            {"_id": db_user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Create token
        user_id = str(db_user["_id"])
        token = f"custom_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        return JSONResponse({
            "success": True,
            "message": "Login successful",
            "token": token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": db_user["email"],
                "name": db_user.get("name", ""),
                "username": db_user.get("username", "")
            },
            "database_connected": True
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/verify-token")
async def verify_token(request: TokenVerify):
    """Verify a user token"""
    try:
        # Check MongoDB connection
        if not check_mongodb_connection():
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Import here to avoid circular imports
        from database import verify_user_token
        
        user = verify_user_token(request.token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return JSONResponse({
            "success": True,
            "valid": True,
            "user": user,
            "database_connected": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

@router.get("/me")
async def get_current_user_info(token: str):
    """Get current user information"""
    try:
        # Check MongoDB connection
        if not check_mongodb_connection():
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Import here to avoid circular imports
        from database import verify_user_token
        
        user = verify_user_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get full user info from database
        user_id = user.get("id") or user.get("uid")
        if user_id.startswith("custom_"):
            user_id = user_id.replace("custom_", "").split("_")[0]
        
        db_user = users_collection.find_one(
            {"$or": [{"_id": user_id}, {"uid": user_id}]},
            {"password_hash": 0}
        )
        
        if db_user:
            if "_id" in db_user:
                db_user["id"] = str(db_user["_id"])
                del db_user["_id"]
            
            # Format timestamps
            for key in ["created_at", "last_login"]:
                if key in db_user and isinstance(db_user[key], datetime):
                    db_user[key] = db_user[key].isoformat()
        
        return JSONResponse({
            "success": True,
            "user": db_user or user,
            "database_connected": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user info: {str(e)}")

@router.get("/test")
async def test_auth():
    """Test authentication endpoint"""
    try:
        db_status = check_mongodb_connection()
        
        return JSONResponse({
            "success": True,
            "message": "Auth router is working",
            "mongodb_connected": db_status,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Auth router test failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })

@router.get("/status")
async def auth_status():
    """Get authentication system status"""
    try:
        db_status = check_mongodb_connection()
        
        # Check if database has users
        user_count = 0
        if db_status:
            try:
                user_count = users_collection.count_documents({})
            except:
                user_count = 0
        
        return JSONResponse({
            "success": True,
            "status": {
                "mongodb_connected": db_status,
                "user_count": user_count,
                "router": "operational"
            },
            "endpoints": {
                "signup": "POST /api/auth/signup",
                "login": "POST /api/auth/login",
                "verify_token": "POST /api/auth/verify-token",
                "me": "GET /api/auth/me?token=TOKEN"
            }
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "status": {
                "mongodb_connected": False,
                "error": str(e)
            }
        }, status_code=500)

# Health check endpoint
@router.get("/health")
async def auth_health():
    """Health check for authentication service"""
    try:
        db_status = check_mongodb_connection()
        
        return {
            "status": "healthy" if db_status else "degraded",
            "service": "authentication",
            "mongodb": "connected" if db_status else "disconnected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }