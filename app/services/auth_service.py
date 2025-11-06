import httpx
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.rag_models import GoogleUserInfo, UserProfile, AuthToken


class JWTManager:
    @staticmethod
    def create_access_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> dict:
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


class GoogleOAuthManager:
    def __init__(self):
        self.client_id = settings.GOOGLE_CLIENT_ID
        self.client_secret = settings.GOOGLE_CLIENT_SECRET
        self.redirect_uri = settings.GOOGLE_REDIRECT_URI
        
    async def exchange_code_for_token(self, code: str, redirect_uri: Optional[str] = None) -> dict:
        """Exchange authorization code for access token"""
        token_url = "https://oauth2.googleapis.com/token"
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri or self.redirect_uri,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for token"
                )
            return response.json()
    
    async def get_user_info(self, access_token: str) -> GoogleUserInfo:
        """Get user information from Google"""
        user_info_url = f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(user_info_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from Google"
                )
            
            user_data = response.json()
            return GoogleUserInfo(**user_data)
    
    def get_auth_url(self, redirect_uri: Optional[str] = None) -> str:
        """Get Google OAuth authorization URL"""
        if not self.client_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured"
            )
        
        auth_url = (
            f"https://accounts.google.com/o/oauth2/auth?"
            f"response_type=code&"
            f"client_id={self.client_id}&"
            f"redirect_uri={(redirect_uri or self.redirect_uri)}&"
            f"scope=openid%20email%20profile&"
            f"access_type=offline"
        )
        
        return auth_url
