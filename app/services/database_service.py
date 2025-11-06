import os
import psycopg2
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient

from app.core.config import settings
from app.models.rag_models import UserProfile, ChatSession

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.mongo_client = None
        self.mongo_db = None
        self.sessions_collection = None
        self.users_collection = None
        self._initialize_mongodb()
        
    def _initialize_mongodb(self):
        """Initialize MongoDB connection with proper error handling"""
        try:
            # Use existing MongoDB connection from settings
            self.mongo_client = MongoClient(settings.MONGODB_URI)
            self.mongo_db = self.mongo_client[settings.MONGODB_DATABASE_NAME]
            self.sessions_collection = self.mongo_db[settings.MONGODB_CHAT_SESSIONS_COLLECTION]
            self.users_collection = self.mongo_db[settings.MONGODB_USERS_COLLECTION]
            
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB for RAG chatbot")
            
        except Exception as e:
            logger.warning(f"MongoDB initialization failed: {e}. Session storage will be disabled.")
            self.mongo_client = None
    
    def is_mongodb_available(self) -> bool:
        """Check if MongoDB is available"""
        return self.mongo_client is not None and self.sessions_collection is not None
        
    def get_postgres_connection(self):
        """Get PostgreSQL connection for RAG data queries"""
        if not settings.POSTGRES_URL:
            raise ValueError("PostgreSQL URL is not configured.")
        return psycopg2.connect(settings.POSTGRES_URL, cursor_factory=RealDictCursor)
    
    def save_user(self, user_profile: UserProfile) -> str:
        """Save or update user profile"""
        if not self.is_mongodb_available():
            raise Exception("Database not available")
        
        try:
            user_data = user_profile.dict()
            # Ensure google_id is present if available
            if getattr(user_profile, 'google_id', None):
                user_data['google_id'] = user_profile.google_id
            
            # Check if user exists
            existing_user = self.users_collection.find_one({"email": user_profile.email})
            
            if existing_user:
                # Update existing user
                user_data["last_login"] = datetime.utcnow()
                self.users_collection.update_one(
                    {"email": user_profile.email},
                    {"$set": user_data}
                )
                return existing_user.get("user_id") or existing_user.get("_id")
            else:
                # Create new user; ensure user_id is stored
                if "user_id" not in user_data:
                    user_data["user_id"] = user_profile.user_id
                result = self.users_collection.insert_one(user_data)
                return user_data.get("user_id")
                
        except Exception as e:
            logger.error(f"Failed to save user: {e}")
            raise Exception("Failed to save user")
    
    def get_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID"""
        if not self.is_mongodb_available():
            return None
        
        try:
            user_data = self.users_collection.find_one({"user_id": user_id})
            if user_data:
                # Remove MongoDB _id field
                user_data.pop('_id', None)
                return UserProfile(**user_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email"""
        if not self.is_mongodb_available():
            return None
        
        try:
            user_data = self.users_collection.find_one({"email": email})
            if user_data:
                # Remove MongoDB _id field
                user_data.pop('_id', None)
                return UserProfile(**user_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None
    
    def save_session(self, session_id: str, message: str, response: str, query_type: str, 
                    user_id: Optional[str] = None, sql_query: Optional[str] = None, 
                    query_results: Optional[Dict[str, Any]] = None):
        """Save chat session"""
        if not self.is_mongodb_available():
            logger.warning("MongoDB not available, session not saved")
            return
            
        try:
            session_data = ChatSession(
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                user_message=message,
                bot_response=response,
                query_type=query_type,
                sql_query=sql_query,
                query_results=query_results
            ).dict()
            
            self.sessions_collection.insert_one(session_data)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def get_session_context(self, session_id: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Get session context with user filtering"""
        if not self.is_mongodb_available():
            logger.warning("MongoDB not available, returning empty context")
            return []
            
        try:
            query = {"session_id": session_id}
            if user_id:
                query["user_id"] = user_id
                
            sessions = list(self.sessions_collection.find(query).sort("timestamp", -1).limit(limit))
            # Remove MongoDB _id field from results
            for session in sessions:
                session.pop('_id', None)
            return sessions
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return []
    
    def get_user_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's chat history"""
        if not self.is_mongodb_available():
            return []
            
        try:
            sessions = list(
                self.sessions_collection.find({"user_id": user_id})
                .sort("timestamp", -1)
                .limit(limit)
            )
            # Remove MongoDB _id field from results
            for session in sessions:
                session.pop('_id', None)
            return sessions
        except Exception as e:
            logger.error(f"Failed to get user chat history: {e}")
            return []
