import firebase_admin
from firebase_admin import credentials, auth
import os
import json
import base64

def initialize_firebase_auth():
    """Initialize Firebase Admin SDK for Authentication ONLY"""
    try:
        if firebase_admin._apps:
            # Already initialized
            print("✅ Firebase already initialized")
            return firebase_admin.get_app()
        
        print("🔄 Initializing Firebase Authentication...")
        
        # Try different methods to load credentials
        cred = None
        
        # 1. From service account file
        service_account_path = "firebase-service-account.json"
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            print(f"✅ Firebase config loaded from file: {service_account_path}")
        
        # 2. From environment variable (for cloud deployment)
        elif os.getenv("FIREBASE_CONFIG_JSON"):
            try:
                config_dict = json.loads(os.getenv("FIREBASE_CONFIG_JSON"))
                cred = credentials.Certificate(config_dict)
                print("✅ Firebase config loaded from environment variable")
            except Exception as e:
                print(f"❌ Error parsing FIREBASE_CONFIG_JSON: {e}")
        
        # 3. From base64 encoded config
        elif os.getenv("FIREBASE_CONFIG_BASE64"):
            try:
                config_json = json.loads(base64.b64decode(os.getenv("FIREBASE_CONFIG_BASE64")))
                cred = credentials.Certificate(config_json)
                print("✅ Firebase config loaded from base64")
            except Exception as e:
                print(f"❌ Error decoding FIREBASE_CONFIG_BASE64: {e}")
        
        if not cred:
            print("⚠️ No Firebase credentials found. Authentication will be disabled.")
            return None
        
        # Initialize Firebase app with auth only
        app = firebase_admin.initialize_app(cred)
        print("✅ Firebase Authentication initialized successfully")
        return app
        
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")
        return None

# Initialize Firebase Auth
firebase_app = initialize_firebase_auth()

def get_firebase_auth():
    """Get Firebase Auth instance"""
    if not firebase_admin._apps:
        return None
    return auth

def verify_firebase_token(id_token):
    """Verify Firebase ID token"""
    try:
        if not firebase_admin._apps:
            return None
        
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"❌ Token verification error: {e}")
        return None

def create_custom_token(uid):
    """Create custom token for Firebase Auth"""
    try:
        if not firebase_admin._apps:
            return None
        
        return auth.create_custom_token(uid)
    except Exception as e:
        print(f"❌ Custom token creation error: {e}")
        return None

# Export functions
__all__ = [
    'initialize_firebase_auth',
    'get_firebase_auth',
    'verify_firebase_token',
    'create_custom_token',
    'firebase_app'
]