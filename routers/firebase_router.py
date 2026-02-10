from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import firebase_admin
from firebase_admin import auth, firestore

from database import users_collection, sync_user_to_firestore
from firebase_config import verify_firebase_token, create_custom_token

router = APIRouter(prefix="/api/firebase", tags=["Firebase"])

# Request/Response Models
class FirebaseLoginRequest(BaseModel):
    id_token: str

class FirebaseUserResponse(BaseModel):
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    token: str
    provider: str = "firebase"

class SyncUserRequest(BaseModel):
    uid: str
    email: str
    name: Optional[str] = None

class DeleteUserRequest(BaseModel):
    uid: str

@router.post("/verify-token")
async def verify_firebase_token_endpoint(request: FirebaseLoginRequest):
    """Verify Firebase ID token and create session"""
    try:
        # Verify the Firebase token
        decoded_token = verify_firebase_token(request.id_token)
        if not decoded_token:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Extract user info
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name") or decoded_token.get("display_name", "")
        
        if not uid:
            raise HTTPException(status_code=400, detail="No UID in token")
        
        # Check if user exists in MongoDB
        user = users_collection.find_one({"uid": uid})
        
        if not user:
            # Create new user in MongoDB
            user_data = {
                "uid": uid,
                "email": email,
                "name": name,
                "auth_provider": "firebase",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "verified": decoded_token.get("email_verified", False),
                "firebase_claims": decoded_token
            }
            
            # Add picture if available
            if decoded_token.get("picture"):
                user_data["picture"] = decoded_token["picture"]
            
            users_collection.insert_one(user_data)
            print(f"✅ Created new user from Firebase: {uid}")
        else:
            # Update last login
            users_collection.update_one(
                {"uid": uid},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        
        # Sync to Firestore
        sync_success = sync_user_to_firestore({
            "uid": uid,
            "email": email,
            "name": name,
            "provider": "firebase"
        })
        
        # Create a custom token for your app (optional)
        custom_token = create_custom_token(uid)
        
        return JSONResponse({
            "success": True,
            "user": {
                "uid": uid,
                "email": email,
                "name": name,
                "provider": "firebase"
            },
            "custom_token": custom_token.decode() if custom_token else None,
            "firestore_sync": sync_success,
            "message": "Authentication successful"
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@router.post("/sync-user")
async def sync_user(request: SyncUserRequest):
    """Sync user data between MongoDB and Firestore"""
    try:
        # Update MongoDB
        update_result = users_collection.update_one(
            {"uid": request.uid},
            {"$set": {
                "email": request.email,
                "name": request.name,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        
        # Sync to Firestore
        success = sync_user_to_firestore({
            "uid": request.uid,
            "email": request.email,
            "name": request.name
        })
        
        return JSONResponse({
            "success": success,
            "mongodb_updated": update_result.modified_count > 0,
            "mongodb_upserted": update_result.upserted_id is not None,
            "message": "User synced successfully" if success else "Firestore sync failed"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.get("/user/{uid}")
async def get_user(uid: str):
    """Get user data from both databases"""
    try:
        # Get from MongoDB
        mongo_user = users_collection.find_one({"uid": uid}, {"_id": 0, "firebase_claims": 0})
        
        # Get from Firebase Auth
        firebase_user = None
        try:
            firebase_user = auth.get_user(uid)
            firebase_user = {
                "uid": firebase_user.uid,
                "email": firebase_user.email,
                "display_name": firebase_user.display_name,
                "photo_url": firebase_user.photo_url,
                "email_verified": firebase_user.email_verified,
                "disabled": firebase_user.disabled
            }
        except Exception as auth_error:
            print(f"Firebase Auth error: {auth_error}")
        
        # Get from Firestore
        firestore_data = None
        try:
            from firebase_config import get_firestore_db
            firestore_db = get_firestore_db()
            firestore_doc = firestore_db.collection("users").document(uid).get()
            if firestore_doc.exists:
                firestore_data = firestore_doc.to_dict()
        except Exception as firestore_error:
            print(f"Firestore error: {firestore_error}")
        
        user_data = {
            "mongo": mongo_user,
            "firebase_auth": firebase_user,
            "firestore": firestore_data
        }
        
        return JSONResponse({
            "success": True,
            "user": user_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user: {str(e)}")

@router.post("/delete-user")
async def delete_user_endpoint(request: DeleteUserRequest):
    """Delete user from Firebase Auth"""
    try:
        # First, delete from Firebase Auth
        from firebase_config import delete_user
        delete_success = delete_user(request.uid)
        
        if not delete_success:
            raise HTTPException(status_code=500, detail="Failed to delete user from Firebase Auth")
        
        # Then, mark as deleted in MongoDB (soft delete)
        users_collection.update_one(
            {"uid": request.uid},
            {"$set": {
                "deleted": True,
                "deleted_at": datetime.utcnow()
            }}
        )
        
        return JSONResponse({
            "success": True,
            "message": f"User {request.uid} deleted successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/health")
async def firebase_health():
    """Check Firebase health status"""
    try:
        # Try to get Firebase app
        app = firebase_admin.get_app()
        
        # Try a simple Firestore operation
        from firebase_config import get_firestore_db
        db = get_firestore_db()
        
        # Create a test document
        test_ref = db.collection("health_checks").document()
        test_data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "healthy",
            "service": "medical_chatbot"
        }
        
        test_ref.set(test_data)
        
        # Read it back
        test_doc = test_ref.get()
        
        # Delete test document
        test_ref.delete()
        
        return JSONResponse({
            "success": True,
            "status": "healthy",
            "firebase": "connected",
            "firestore": "operational",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }, status_code=500)