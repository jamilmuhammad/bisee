import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings, logger
from app.models.rag_models import AuthToken, GoogleTokenRequest, UserProfile, GoogleUserInfo
from app.models.token_models import TokenData, TokenResponse
from app.models.user_models import UserModelInDB
from app.services.auth_service import GoogleOAuthManager, JWTManager
from app.services.database_service import DatabaseManager
from app.api.v1.deps import get_current_active_user
from app.core.security import create_access_token, verify_google_token
from app.services.user_service import create_or_update_user_from_google
from datetime import timedelta

router = APIRouter()

# Initialize services for RAG chatbot
google_oauth = GoogleOAuthManager()
db_manager = DatabaseManager()
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
    """Get current authenticated user from JWT token for RAG chatbot"""
    try:
        payload = JWTManager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        user = db_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[UserProfile]:
    """Get current authenticated user (optional) for RAG chatbot"""
    try:
        return await get_current_user(credentials)
    except:
        return None


# Original Google auth endpoint (maintained for compatibility)
@router.post("/google", response_model=TokenResponse, tags=["Authentication"])
async def login_with_google(token_data: TokenData):
    logger.info("Received request for Google login.")
    google_user_info = await verify_google_token(token_data.google_id_token)
    if not google_user_info:
        logger.warning("Google ID token verification failed.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google ID Token"
        )

    user_google_id = google_user_info.get("sub")
    user_email = google_user_info.get("email")
    user_name = google_user_info.get("name")
    user_picture = google_user_info.get("picture")

    if not user_google_id or not user_email:  # Basic validation
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token missing required fields",
        )

    # create_or_update_user_from_google is synchronous (uses pymongo sync client)
    pydantic_user = create_or_update_user_from_google(
        google_id=user_google_id,
        email=user_email,
        name=user_name,
        picture=user_picture,
    )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            # Use google_id as the subject to uniquely identify the Google user
            "sub": pydantic_user.google_id,
            "email": pydantic_user.email,
        },
        expires_delta=access_token_expires,
    )
    logger.info(f"User {pydantic_user.email} authenticated successfully via Google.")
    return TokenResponse(
        access_token=access_token, token_type="bearer", user_info=pydantic_user
    )


@router.get("/users/me", response_model=UserModelInDB, tags=["Authentication"])
async def read_users_me(current_user: UserModelInDB = Depends(get_current_active_user)):
    return current_user


# New RAG chatbot auth endpoints
@router.get("/google/url", tags=["RAG Authentication"])
async def get_google_auth_url(redirect_uri: Optional[str] = None):
    """Get Google OAuth authorization URL for RAG chatbot"""
    try:
        auth_url = google_oauth.get_auth_url(redirect_uri)
        return {"auth_url": auth_url}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/google/callback", response_model=AuthToken, tags=["RAG Authentication"])
async def google_callback(request: GoogleTokenRequest):
    """Handle Google OAuth callback for RAG chatbot"""
    try:
        # Exchange code for token
        token_data = await google_oauth.exchange_code_for_token(request.code, request.redirect_uri)
        access_token = token_data.get("access_token")

        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get access token"
            )

        # Get user info from Google
        user_info = await google_oauth.get_user_info(access_token)

        # Check if user exists or create new user
        existing_user = db_manager.get_user_by_email(user_info.email)

        if existing_user:
            # Update last login
            existing_user.last_login = datetime.utcnow()
            db_manager.save_user(existing_user)
            user_profile = existing_user
        else:
            # Create new user
            user_profile = UserProfile(
                user_id=str(uuid.uuid4()),
                google_id=user_info.id,
                email=user_info.email,
                name=user_info.name,
                picture=user_info.picture,
                provider="google",
                preferences={},
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow(),
            )
            db_manager.save_user(user_profile)

        # Create JWT token
        # Use user_id as canonical subject; keep google_id in claims if available
        jwt_payload = {
            "sub": user_profile.user_id,
            "email": user_profile.email,
            "name": user_profile.name,
            "google_id": user_profile.google_id,
        }
        jwt_token = JWTManager.create_access_token(jwt_payload)

        return AuthToken(
            access_token=jwt_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
            user_profile=user_profile,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}",
        )


@router.get("/google/callback", response_model=AuthToken, tags=["RAG Authentication"])
async def google_callback_get(code: str, redirect_uri: Optional[str] = None):
    """Handle Google OAuth callback (GET) for browser redirects."""
    try:
        # Exchange code for token
        token_data = await google_oauth.exchange_code_for_token(code, redirect_uri)
        access_token = token_data.get("access_token")

        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get access token"
            )

        # Get user info from Google
        user_info = await google_oauth.get_user_info(access_token)

        # Check if user exists or create new user
        existing_user = db_manager.get_user_by_email(user_info.email)

        if existing_user:
            # Update last login
            existing_user.last_login = datetime.utcnow()
            db_manager.save_user(existing_user)
            user_profile = existing_user
        else:
            # Create new user
            user_profile = UserProfile(
                user_id=str(uuid.uuid4()),
                google_id=user_info.id,
                email=user_info.email,
                name=user_info.name,
                picture=user_info.picture,
                provider="google",
                preferences={},
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow(),
            )
            db_manager.save_user(user_profile)

        # Create JWT token (subject = internal user_id)
        jwt_payload = {
            "sub": user_profile.user_id,
            "email": user_profile.email,
            "name": user_profile.name,
            "google_id": user_profile.google_id,
        }
        jwt_token = JWTManager.create_access_token(jwt_payload)

        return AuthToken(
            access_token=jwt_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_profile=user_profile,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}",
        )


@router.get("/me", response_model=UserProfile, tags=["RAG Authentication"])
async def get_current_user_profile(current_user: UserProfile = Depends(get_current_user)):
    """Get current user profile for RAG chatbot"""
    return current_user


@router.post("/logout", tags=["RAG Authentication"])
async def logout(current_user: UserProfile = Depends(get_current_user)):
    """Logout user (client should delete the token)"""
    return {"message": "Logged out successfully"}


@router.get("/verify", tags=["RAG Authentication"])
async def verify_token(current_user: UserProfile = Depends(get_current_user)):
    """Verify if the current token is valid for RAG chatbot"""
    return {"valid": True, "user_id": current_user.user_id, "email": current_user.email}
