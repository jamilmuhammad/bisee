from fastapi import APIRouter
from app.api.v1.endpoints import auth_ep, rag_chat_ep

api_router_v1 = APIRouter()
api_router_v1.include_router(auth_ep.router, prefix="/auth", tags=["Authentication"])
# api_router_v1.include_router(chat_ep.router, prefix="/chat", tags=["Chat"])
api_router_v1.include_router(rag_chat_ep.router, prefix="/rag-chat", tags=["RAG Chat"])
