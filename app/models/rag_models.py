from pydantic import BaseModel, EmailStr
from typing import Dict, List, Optional, Any
from datetime import datetime


class UserProfile(BaseModel):
    user_id: str
    google_id: Optional[str] = None
    email: EmailStr
    name: str
    picture: Optional[str] = None
    provider: str = "google"
    preferences: Optional[Dict[str, Any]] = {}
    created_at: datetime
    last_login: datetime


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_profile: UserProfile


class GoogleTokenRequest(BaseModel):
    code: str
    redirect_uri: Optional[str] = None


class GoogleUserInfo(BaseModel):
    id: str
    email: str
    name: str
    picture: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class AnalysisState(BaseModel):
    analysis_type: Optional[str] = None
    forecast_results: Optional[Dict[str, Any]] = None
    simulation_results: Optional[Dict[str, Any]] = None
    predictive_model: Optional[str] = None
    simulation_parameters: Optional[Dict[str, Any]] = None
    data_validation: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    query_types: List[str]
    context: str
    sql_query: Optional[str] = None
    query_results: Optional[Dict[str, Any]] = None
    query_data: Optional[List[Dict[str, Any]]] = None
    final_response: str
    reflection_feedback: Optional[str] = None
    refined_response: Optional[str] = None
    analysis_state: Optional[AnalysisState] = None


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 10
    concept_map_id: Optional[str] = None


class RAGQueryResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    question: str
    top_k: int
    user_id: Optional[str] = None


class ChatSession(BaseModel):
    session_id: str
    user_id: Optional[str]
    timestamp: datetime
    user_message: str
    bot_response: str
    query_type: str
    sql_query: Optional[str] = None
    query_results: Optional[Dict[str, Any]] = None
