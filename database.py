from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
from datetime import datetime
import certifi
from bson import ObjectId
from datetime import timedelta
import traceback
import uuid

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "medical_chatbot")

print("🔄 Connecting to MongoDB...")

# MongoDB Connection with error handling
try:
    # For MongoDB Atlas (cloud), use certifi for SSL
    if "mongodb+srv" in MONGODB_URL:
        client = MongoClient(MONGODB_URL, tlsCAFile=certifi.where())
        print("✅ Connected to MongoDB Atlas")
    else:
        client = MongoClient(MONGODB_URL)
        print("✅ Connected to local MongoDB")
    
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connection test successful")
    
    # Select database
    db = client[DATABASE_NAME]
    print(f"✅ Using database: {DATABASE_NAME}")
    
    # ============================================
    # 3 COLLECTIONS AS REQUESTED:
    # ============================================
    
    # 1. Users folder
    users_collection = db["users"]
    
    # 2. Conversations folder (with embedded messages)
    conversations_collection = db["conversations"]
    
    # 3. ADDED FEEDBACK: Feedback collection
    feedback_collection = db["feedback"]
    
    # Create indexes with duplicate index handling
    users_collection.create_index("email", unique=True, sparse=True)
    users_collection.create_index("uid", unique=True, sparse=True)
    
    conversations_collection.create_index("user_id")
    conversations_collection.create_index([("user_id", 1), ("updated_at", -1)])
    
    # FIX: Handle duplicate session_id index issue
    try:
        # Try to drop any existing session_id index first
        conversations_collection.drop_index("session_id_1")
        print("✅ Dropped old session_id index")
    except Exception as e:
        # Index might not exist - that's okay
        print(f"⚠️ Note: Could not drop session_id index (might not exist): {e}")
    
    # Create session_id index (non-unique)
    conversations_collection.create_index("session_id")
    
    # ADDED FEEDBACK: Create indexes for feedback collection
    feedback_collection.create_index([("user_id", 1), ("session_id", 1), ("message_id", 1)])
    feedback_collection.create_index("timestamp")
    feedback_collection.create_index("feedback_type")
    
    print("✅ MongoDB collections and indexes ready (including feedback)")
    
except ConnectionFailure as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("⚠️ Running in offline mode with dummy collections")
    
    # Create dummy collections to prevent crashes
    class DummyCollection:
        def find_one(self, *args, **kwargs): return None
        def insert_one(self, data):
            print(f"📝 Dummy insert: {data.get('email', 'No email')}")
            return type('obj', (object,), {'inserted_id': 'dummy_id'})()
        def find(self, *args, **kwargs): return []
        def update_one(self, *args, **kwargs):
            return type('obj', (object,), {'matched_count': 0, 'modified_count': 0})()
        def create_index(self, *args, **kwargs): pass
        def count_documents(self, *args, **kwargs): return 0
        def delete_many(self, *args, **kwargs):
            class Result:
                deleted_count = 0
            return Result()
        def aggregate(self, *args, **kwargs): return []
        def distinct(self, *args, **kwargs): return []
        def update_many(self, *args, **kwargs):
            return type('obj', (object,), {'modified_count': 0})()
    
    users_collection = DummyCollection()
    conversations_collection = DummyCollection()
    feedback_collection = DummyCollection()
    client = None
    db = None

# ============================================
# ADDED FEEDBACK: Feedback Functions
# ============================================

def save_feedback(user_id, email, feedback_data):
    """Save user feedback to MongoDB"""
    try:
        feedback_doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "user_email": email,
            "session_id": feedback_data.get("session_id"),
            "message_id": feedback_data.get("message_id"),
            "feedback_type": feedback_data.get("feedback_type"),  # "thumbs_up" or "thumbs_down"
            "user_comment": feedback_data.get("user_comment"),
            "rating": feedback_data.get("rating"),  # Optional 1-5 rating
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        
        result = feedback_collection.insert_one(feedback_doc)
        feedback_id = str(result.inserted_id)
        
        print(f"💾 Feedback saved: {feedback_data.get('feedback_type')} for message {feedback_data.get('message_id')}")
        return feedback_id
        
    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        traceback.print_exc()
        return None

def get_user_feedback(user_id, limit=50):
    """Get all feedback for a specific user"""
    try:
        feedbacks = list(feedback_collection.find(
            {"user_id": user_id},
            {
                "_id": 1,
                "session_id": 1,
                "message_id": 1,
                "feedback_type": 1,
                "user_comment": 1,
                "rating": 1,
                "timestamp": 1
            }
        ).sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string
        for fb in feedbacks:
            fb["_id"] = str(fb["_id"])
            if fb.get("timestamp"):
                fb["timestamp"] = fb["timestamp"].isoformat()
        
        return feedbacks
    except Exception as e:
        print(f"❌ Error getting user feedback: {e}")
        return []

def get_session_feedback(user_id, session_id):
    """Get all feedback for a specific session"""
    try:
        feedbacks = list(feedback_collection.find(
            {"user_id": user_id, "session_id": session_id},
            {
                "_id": 1,
                "message_id": 1,
                "feedback_type": 1,
                "user_comment": 1,
                "rating": 1,
                "timestamp": 1
            }
        ).sort("timestamp", -1))
        
        # Convert ObjectId to string
        for fb in feedbacks:
            fb["_id"] = str(fb["_id"])
            if fb.get("timestamp"):
                fb["timestamp"] = fb["timestamp"].isoformat()
        
        return feedbacks
    except Exception as e:
        print(f"❌ Error getting session feedback: {e}")
        return []

def get_feedback_stats(user_id=None):
    """Get feedback statistics"""
    try:
        query = {} if user_id is None else {"user_id": user_id}
        
        total_feedback = feedback_collection.count_documents(query)
        thumbs_up = feedback_collection.count_documents({**query, "feedback_type": "thumbs_up"})
        thumbs_down = total_feedback - thumbs_up
        
        # Calculate average rating
        pipeline = [
            {"$match": {**query, "rating": {"$ne": None}}},
            {"$group": {
                "_id": None,
                "avg_rating": {"$avg": "$rating"},
                "count": {"$sum": 1}
            }}
        ]
        
        rating_result = list(feedback_collection.aggregate(pipeline))
        avg_rating = rating_result[0]["avg_rating"] if rating_result else 0
        
        return {
            "total_feedback": total_feedback,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "average_rating": round(avg_rating, 2) if avg_rating else 0,
            "thumbs_up_percentage": round((thumbs_up / total_feedback * 100), 2) if total_feedback > 0 else 0
        }
    except Exception as e:
        print(f"❌ Error getting feedback stats: {e}")
        return {
            "total_feedback": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "average_rating": 0,
            "thumbs_up_percentage": 0
        }

def get_message_feedback(user_id, session_id, message_id):
    """Get feedback for a specific message"""
    try:
        feedback = feedback_collection.find_one({
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id
        })
        
        if feedback:
            feedback["_id"] = str(feedback["_id"])
            if feedback.get("timestamp"):
                feedback["timestamp"] = feedback["timestamp"].isoformat()
            return feedback
        return None
    except Exception as e:
        print(f"❌ Error getting message feedback: {e}")
        return None

# ============================================
# USER MANAGEMENT FUNCTIONS
# ============================================

def verify_user_token(token):
    """
    Verify user token - supports both Firebase and custom tokens
    Returns user data if valid
    """
    # If token is a Firebase ID token
    if token and len(token) > 100 and not token.startswith("custom_"):
        try:
            # Remove any prefixes
            id_token = token.replace("firebase_", "").replace("Bearer ", "")
            
            # Import here to avoid circular imports
            from firebase_config import verify_firebase_token
            decoded = verify_firebase_token(id_token)
            
            if decoded:
                # Check if user exists in MongoDB
                user = users_collection.find_one({"uid": decoded["uid"]})
                
                if not user:
                    # Create user in MongoDB from Firebase data
                    user_data = {
                        "uid": decoded["uid"],
                        "email": decoded.get("email"),
                        "name": decoded.get("name") or decoded.get("display_name", ""),
                        "auth_provider": "firebase",
                        "created_at": datetime.utcnow(),
                        "last_login": datetime.utcnow(),
                        "verified": decoded.get("email_verified", False)
                    }
                    
                    # Also add picture if available
                    if decoded.get("picture"):
                        user_data["picture"] = decoded["picture"]
                    
                    users_collection.insert_one(user_data)
                    print(f"✅ Created new user in MongoDB: {decoded['uid']}")
                else:
                    # Update last login
                    users_collection.update_one(
                        {"uid": decoded["uid"]},
                        {"$set": {"last_login": datetime.utcnow()}}
                    )
                
                return {
                    "uid": decoded["uid"],
                    "email": decoded.get("email"),
                    "name": decoded.get("name"),
                    "provider": "firebase",
                    "verified": decoded.get("email_verified", False)
                }
        except Exception as e:
            print(f"Firebase token verification error: {e}")
            return None
    
    # Custom token verification
    elif token and token.startswith("custom_"):
        try:
            user_id = token.replace("custom_", "")
            user = users_collection.find_one({"$or": [{"_id": user_id}, {"uid": user_id}]})
            
            if user:
                return {
                    "id": str(user.get("_id")),
                    "email": user.get("email"),
                    "name": user.get("name"),
                    "provider": "custom"
                }
        except Exception as e:
            print(f"Custom token verification error: {e}")
    
    return None

def get_or_create_user(user_data):
    """Get existing user or create new one"""
    try:
        # Look for user by uid or email
        user = users_collection.find_one({
            "$or": [
                {"uid": user_data.get("uid")},
                {"email": user_data.get("email")}
            ]
        })
        
        if user:
            # Update last login
            users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            print(f"✅ User found: {user.get('email')}")
            return str(user["_id"]), user
        else:
            # Create new user
            new_user = {
                "uid": user_data.get("uid"),
                "email": user_data.get("email"),
                "name": user_data.get("name", ""),
                "auth_provider": user_data.get("provider", "firebase"),
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "verified": user_data.get("verified", False),
                "active": True
            }
            
            # Add profile picture if available
            if user_data.get("picture"):
                new_user["picture"] = user_data.get("picture")
            
            result = users_collection.insert_one(new_user)
            user_id = str(result.inserted_id)
            print(f"✅ New user created: {user_data.get('email')}")
            
            # Return the created user data
            new_user["_id"] = result.inserted_id
            return user_id, new_user
            
    except Exception as e:
        print(f"❌ Error in get_or_create_user: {e}")
        return None, None

def get_all_users(limit=100):
    """Get all users with basic info"""
    try:
        users = list(users_collection.find(
            {"active": {"$ne": False}},
            {
                "_id": 1,
                "email": 1,
                "name": 1,
                "created_at": 1,
                "last_login": 1,
                "last_activity": 1
            }
        ).sort("last_login", -1).limit(limit))
        
        # Convert ObjectId to string and add conversation count
        for user in users:
            user["_id"] = str(user["_id"])
            user["conversation_count"] = conversations_collection.count_documents({
                "user_id": user.get("uid", user["_id"])
            })
        
        return users
    except Exception as e:
        print(f"❌ Error getting all users: {e}")
        return []

def create_user_in_mongodb(user_data):
    """Create a new user in MongoDB"""
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({
            "$or": [
                {"email": user_data.get("email")},
                {"uid": user_data.get("uid")}
            ]
        })
        
        if existing_user:
            print(f"⚠️ User already exists: {user_data.get('email')}")
            return str(existing_user["_id"])
        
        # Add timestamps
        user_data["created_at"] = datetime.utcnow()
        user_data["last_login"] = datetime.utcnow()
        
        # Insert user
        result = users_collection.insert_one(user_data)
        print(f"✅ User created in MongoDB: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Error creating user in MongoDB: {e}")
        return None

def get_user_by_email(email):
    """Get user by email from MongoDB"""
    try:
        user = users_collection.find_one({"email": email}, {"_id": 0, "password": 0})
        return user
    except Exception as e:
        print(f"❌ Error getting user by email: {e}")
        return None

def get_user_by_uid_or_email(identifier):
    """Get user by UID, email, or MongoDB _id"""
    try:
        # Try as MongoDB ObjectId first
        if ObjectId.is_valid(identifier):
            user = users_collection.find_one({"_id": ObjectId(identifier)})
            if user:
                user["_id"] = str(user["_id"])
                return user
        
        # Try as Firebase UID
        user = users_collection.find_one({"uid": identifier})
        if user:
            user["_id"] = str(user["_id"])
            return user
        
        # Try as email
        user = users_collection.find_one({"email": identifier})
        if user:
            user["_id"] = str(user["_id"])
            return user
        
        return None
    except Exception as e:
        print(f"❌ Error getting user by identifier: {e}")
        return None

def ensure_user_exists(firebase_user_data):
    """Ensure user exists in MongoDB, create if not"""
    try:
        uid = firebase_user_data.get("uid")
        email = firebase_user_data.get("email")
        
        if not uid:
            return None
        
        # Check if user exists
        user = users_collection.find_one({"uid": uid})
        
        if user:
            # Update last login
            users_collection.update_one(
                {"uid": uid},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            return str(user["_id"])
        else:
            # Create new user
            new_user = {
                "uid": uid,
                "email": email,
                "name": firebase_user_data.get("name", ""),
                "auth_provider": "firebase",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "verified": firebase_user_data.get("email_verified", False),
                "active": True
            }
            
            # Add profile picture if available
            if firebase_user_data.get("picture"):
                new_user["picture"] = firebase_user_data.get("picture")
            
            result = users_collection.insert_one(new_user)
            user_id = str(result.inserted_id)
            print(f"✅ Created new Firebase user: {email} (UID: {uid})")
            return user_id
            
    except Exception as e:
        print(f"❌ Error ensuring user exists: {e}")
        return None

def is_valid_object_id(obj_id):
    """Check if a string is a valid MongoDB ObjectId"""
    try:
        return ObjectId.is_valid(obj_id)
    except:
        return False

def update_user_profile(uid, update_data):
    """Update user profile in MongoDB"""
    try:
        update_data["updated_at"] = datetime.utcnow()
        
        result = users_collection.update_one(
            {"uid": uid},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    except Exception as e:
        print(f"❌ Error updating user profile: {e}")
        return False

def delete_user_from_mongodb(uid):
    """Delete user from MongoDB (soft delete)"""
    try:
        result = users_collection.update_one(
            {"uid": uid},
            {"$set": {
                "deleted": True,
                "deleted_at": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"❌ Error deleting user from MongoDB: {e}")
        return False

def sync_user_to_firestore(user_data):
    """Sync user data to Firestore (dummy implementation)"""
    try:
        # This is a dummy implementation since we don't have Firestore configured
        # In a real app, you would use firebase_admin.firestore
        
        print(f"📤 Would sync user to Firestore: {user_data.get('email', 'Unknown')}")
        
        # Return True to indicate success (for compatibility)
        return True
        
    except Exception as e:
        print(f"❌ Error syncing to Firestore (dummy): {e}")
        return False

# ============================================
# CONVERSATION FUNCTIONS (WITH EMBEDDED MESSAGES)
# ============================================

def create_conversation(user_id, session_id=None, title="New Conversation"):
    """Create a new conversation for a user"""
    try:
        if not session_id:
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        conversation_data = {
            "user_id": user_id,
            "session_id": session_id,
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0,
            "is_active": True,
            "messages": []  # Embedded messages array
        }
        
        result = conversations_collection.insert_one(conversation_data)
        conversation_id = str(result.inserted_id)
        print(f"✅ New conversation created: {conversation_id}")
        
        return conversation_id, session_id
        
    except Exception as e:
        print(f"❌ Error creating conversation: {e}")
        return None, None

def save_message(user_id, email, message_data, conversation_id=None, session_id=None):
    """Save a message - embedded directly in conversation"""
    try:
        # If no conversation_id provided, create or get existing conversation
        if not conversation_id:
            if session_id:
                # Find existing conversation by session_id
                conversation = conversations_collection.find_one({
                    "user_id": user_id,
                    "session_id": session_id
                })
                if conversation:
                    conversation_id = str(conversation["_id"])
                else:
                    # Create new conversation
                    conversation_id, _ = create_conversation(user_id, session_id)
            else:
                # Create new conversation with generated session_id
                conversation_id, session_id = create_conversation(user_id)
        
        # Prepare message document
        message_doc = {
            "message_id": str(uuid.uuid4()),
            "role": message_data.get("role", "user"),
            "content": message_data.get("message", ""),
            "timestamp": datetime.utcnow(),
            "is_ai": message_data.get("is_ai", False),
            "use_rag": message_data.get("use_rag", False)
        }
        
        # Add voice metadata if applicable
        if message_data.get("is_voice"):
            message_doc["is_voice"] = True
            message_doc["voice_language"] = message_data.get("voice_language")
        
        # Add the message to the conversation's messages array
        result = conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$push": {"messages": message_doc},
                "$inc": {"message_count": 1},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Update user's last activity
        users_collection.update_one(
            {"uid": user_id},
            {"$set": {"last_activity": datetime.utcnow()}}
        )
        
        print(f"✅ Message saved to conversation: {conversation_id}")
        return message_doc["message_id"], conversation_id
        
    except Exception as e:
        print(f"❌ Error saving message: {e}")
        print(f"   user_id: {user_id}")
        print(f"   email: {email}")
        traceback.print_exc()
        return None, None

def save_message_to_mongodb(message_data):
    """Save chat message to MongoDB - COMPATIBILITY FUNCTION"""
    try:
        # For backward compatibility - convert old format to new format
        user_id = message_data.get("user_id", "unknown")
        email = message_data.get("email", "unknown@example.com")
        
        # Extract message data in new format
        new_message_data = {
            "role": message_data.get("role", "user"),
            "message": message_data.get("message", ""),
            "is_ai": message_data.get("is_ai", False),
            "use_rag": message_data.get("use_rag", False),
            "session_id": message_data.get("session_id")
        }
        
        # Use the new save_message function
        message_id, conversation_id = save_message(
            user_id=user_id,
            email=email,
            message_data=new_message_data,
            session_id=message_data.get("session_id")
        )
        
        return message_id if message_id else None
        
    except Exception as e:
        print(f"❌ Error in save_message_to_mongodb: {e}")
        return None

def get_user_conversations(user_id, limit=20):
    """Get all conversations for a specific user"""
    try:
        conversations = list(conversations_collection.find(
            {"user_id": user_id, "is_active": True},
            {
                "_id": 1,
                "title": 1,
                "session_id": 1,
                "created_at": 1,
                "updated_at": 1,
                "message_count": 1,
                "messages": {"$slice": -1}  # Get only last message for preview
            }
        ).sort("updated_at", -1).limit(limit))
        
        # Convert ObjectId to string and format
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
            
            # Get last message preview
            if conv.get("messages") and len(conv["messages"]) > 0:
                last_msg = conv["messages"][-1]
                conv["last_message_preview"] = (last_msg["content"][:50] + "...") if len(last_msg["content"]) > 50 else last_msg["content"]
                conv["last_message_role"] = last_msg["role"]
            else:
                conv["last_message_preview"] = ""
                conv["last_message_role"] = None
            
            # Remove messages array to save bandwidth (we'll load full messages separately)
            conv.pop("messages", None)
        
        return conversations
    except Exception as e:
        print(f"❌ Error getting user conversations: {e}")
        return []

def get_conversation_with_messages(conversation_id):
    """Get specific conversation with all its messages"""
    try:
        conversation = conversations_collection.find_one(
            {"_id": ObjectId(conversation_id), "is_active": True},
            {
                "_id": 1,
                "user_id": 1,
                "title": 1,
                "session_id": 1,
                "created_at": 1,
                "updated_at": 1,
                "message_count": 1,
                "messages": 1  # Include all messages
            }
        )
        
        if not conversation:
            return None
        
        conversation["_id"] = str(conversation["_id"])
        
        # Get user info
        user = users_collection.find_one(
            {"uid": conversation["user_id"]},
            {"email": 1, "name": 1}
        )
        
        if user:
            conversation["user_email"] = user.get("email")
            conversation["user_name"] = user.get("name")
        
        # Format messages
        if "messages" in conversation:
            # Sort messages by timestamp
            conversation["messages"].sort(key=lambda x: x["timestamp"])
            
            # Convert ObjectId to string for any nested ObjectIds
            for msg in conversation["messages"]:
                if "timestamp" in msg and isinstance(msg["timestamp"], datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
        
        return conversation
    except Exception as e:
        print(f"❌ Error getting conversation with messages: {e}")
        return None

def get_user_with_conversations(user_id):
    """Get user details with their conversations"""
    try:
        # Get user
        if ObjectId.is_valid(user_id):
            user = users_collection.find_one(
                {"_id": ObjectId(user_id)},
                {
                    "_id": 1,
                    "email": 1,
                    "name": 1,
                    "created_at": 1,
                    "last_login": 1,
                    "last_activity": 1
                }
            )
        else:
            user = users_collection.find_one(
                {"uid": user_id},
                {
                    "_id": 1,
                    "email": 1,
                    "name": 1,
                    "created_at": 1,
                    "last_login": 1,
                    "last_activity": 1
                }
            )
        
        if not user:
            return None
        
        user["_id"] = str(user["_id"])
        
        # Get user's conversations
        conversations = get_user_conversations(user["_id"])
        user["conversations"] = conversations
        
        # Get total message count
        total_messages = sum([conv.get("message_count", 0) for conv in conversations])
        user["total_messages"] = total_messages
        
        return user
    except Exception as e:
        print(f"❌ Error getting user with conversations: {e}")
        return None

def get_user_chats(user_id, limit=50):
    """Get user's chat history from MongoDB - COMPATIBILITY FUNCTION"""
    try:
        # Get user's conversations
        conversations = get_user_conversations(user_id, limit)
        
        # Convert to old format for compatibility
        chats = []
        for conv in conversations:
            # Get full conversation with messages
            full_conv = get_conversation_with_messages(conv["_id"])
            if full_conv and "messages" in full_conv:
                for msg in full_conv["messages"]:
                    chats.append({
                        "role": msg.get("role"),
                        "message": msg.get("content"),
                        "timestamp": msg.get("timestamp"),
                        "is_ai": msg.get("is_ai", False),
                        "is_rag": msg.get("use_rag", False)
                    })
        
        # Sort by timestamp (newest first) and limit
        chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return chats[:limit]
        
    except Exception as e:
        print(f"❌ Error getting user chats: {e}")
        return []

def get_conversation_messages(conversation_id, limit=100):
    """Get all messages for a specific conversation - COMPATIBILITY FUNCTION"""
    try:
        conversation = get_conversation_with_messages(conversation_id)
        if not conversation or "messages" not in conversation:
            return []
        
        # Convert to old format
        messages = []
        for msg in conversation["messages"]:
            messages.append({
                "_id": msg.get("message_id"),
                "role": msg.get("role"),
                "message": msg.get("content"),
                "timestamp": msg.get("timestamp"),
                "is_ai": msg.get("is_ai", False),
                "use_rag": msg.get("use_rag", False),
                "session_id": conversation.get("session_id")
            })
        
        return messages[:limit]
        
    except Exception as e:
        print(f"❌ Error getting conversation messages: {e}")
        return []

# ============================================
# HELPER FUNCTIONS
# ============================================

def delete_conversation(conversation_id):
    """Soft delete a conversation"""
    try:
        if ObjectId.is_valid(conversation_id):
            # Mark conversation as inactive
            conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"is_active": False, "deleted_at": datetime.utcnow()}}
            )
            
            print(f"✅ Conversation {conversation_id} marked as deleted")
            return True
        else:
            print(f"⚠️ Invalid conversation_id format: {conversation_id}")
            return False
    except Exception as e:
        print(f"❌ Error deleting conversation: {e}")
        return False

def get_conversation_stats(conversation_id):
    """Get statistics for a conversation"""
    try:
        conversation = conversations_collection.find_one(
            {"_id": ObjectId(conversation_id)},
            {"messages": 1, "message_count": 1}
        )
        
        if not conversation:
            return None
        
        messages = conversation.get("messages", [])
        
        # Count AI vs user messages
        ai_messages = sum(1 for msg in messages if msg.get("is_ai", False))
        user_messages = len(messages) - ai_messages
        
        # Count RAG usage
        rag_messages = sum(1 for msg in messages if msg.get("use_rag", False))
        
        # Get first and last message times
        if messages:
            first_time = min(msg["timestamp"] for msg in messages)
            last_time = max(msg["timestamp"] for msg in messages)
        else:
            first_time = last_time = None
        
        return {
            "total_messages": conversation.get("message_count", 0),
            "ai_messages": ai_messages,
            "user_messages": user_messages,
            "rag_messages": rag_messages,
            "first_message_time": first_time,
            "last_message_time": last_time
        }
    except Exception as e:
        print(f"❌ Error getting conversation stats: {e}")
        return None

def get_user_stats(user_id):
    """Get user statistics"""
    try:
        # Get user's conversations
        conversations = list(conversations_collection.find(
            {"user_id": user_id, "is_active": True},
            {"message_count": 1}
        ))
        
        # Calculate total messages
        total_messages = sum([conv.get("message_count", 0) for conv in conversations])
        
        # Count unique sessions
        session_count = conversations_collection.distinct("session_id", {"user_id": user_id})
        
        # Get last activity
        last_conversation = conversations_collection.find_one(
            {"user_id": user_id, "is_active": True},
            sort=[("updated_at", -1)]
        )
        
        last_activity = last_conversation["updated_at"] if last_conversation else None
        
        return {
            "message_count": total_messages,
            "session_count": len(session_count),
            "last_activity": last_activity
        }
    except Exception as e:
        print(f"❌ Error getting user stats: {e}")
        return {
            "message_count": 0,
            "session_count": 0,
            "last_activity": None
        }

def backup_messages(user_id, session_id=None):
    """Backup user messages to a file"""
    try:
        # Get user's conversations
        query = {"user_id": user_id, "is_active": True}
        if session_id:
            query["session_id"] = session_id
        
        conversations = list(conversations_collection.find(query))
        
        if not conversations:
            return None
        
        backup_data = []
        for conv in conversations:
            if "messages" in conv:
                for msg in conv["messages"]:
                    backup_data.append({
                        "role": msg.get("role", "user"),
                        "message": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", datetime.utcnow()).isoformat(),
                        "session_id": conv.get("session_id", ""),
                        "is_ai": msg.get("is_ai", False),
                        "is_rag": msg.get("use_rag", False)
                    })
        
        return backup_data
    except Exception as e:
        print(f"❌ Error backing up messages: {e}")
        return None

def cleanup_inactive_users(days_inactive=30):
    """Mark users as inactive after specified days"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_inactive)
        
        result = users_collection.update_many(
            {
                "last_activity": {"$lt": cutoff_date},
                "active": True
            },
            {"$set": {"active": False, "inactive_since": datetime.utcnow()}}
        )
        
        print(f"✅ Marked {result.modified_count} users as inactive")
        return result.modified_count
    except Exception as e:
        print(f"❌ Error cleaning up inactive users: {e}")
        return 0

def get_users_with_message_counts(limit=100):
    """Get all users with their message counts (compatible with old view)"""
    try:
        # Get all active users
        users = list(users_collection.find(
            {"active": {"$ne": False}},
            {"_id": 1, "email": 1, "name": 1}
        ).limit(limit))
        
        result = []
        for user in users:
            user_id = user.get("uid") or str(user["_id"])
            
            # Get total messages for this user
            conversations = list(conversations_collection.find(
                {"user_id": user_id},
                {"message_count": 1}
            ))
            
            total_messages = sum([conv.get("message_count", 0) for conv in conversations])
            
            # Get last activity
            last_conv = conversations_collection.find_one(
                {"user_id": user_id},
                sort=[("updated_at", -1)]
            )
            
            result.append({
                "user_id": user_id,
                "email": user.get("email"),
                "message_count": total_messages,
                "last_activity": last_conv["updated_at"] if last_conv else None,
                "first_activity": None  # Not available in new structure
            })
        
        # Sort by last activity
        result.sort(key=lambda x: x.get("last_activity") or "", reverse=True)
        return result[:limit]
        
    except Exception as e:
        print(f"❌ Error getting users with counts: {e}")
        return []

print("✅ Database module loaded with FEEDBACK system: Users + Conversations + Feedback")