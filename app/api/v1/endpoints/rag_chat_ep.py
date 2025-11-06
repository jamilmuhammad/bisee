from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, List

from app.models.rag_models import ChatRequest, ChatResponse, UserProfile, RAGQueryRequest, RAGQueryResponse
from app.services.rag_chatbot_service import RAGChatbotService
from app.services.rag_service import RAGService
# Use centralized dependency helpers
from app.api.v1.deps import get_current_user as get_general_current_user
from app.api.v1.endpoints.auth_ep import get_current_user_optional, get_current_user as get_rag_current_user
router = APIRouter()

# Initialize RAG chatbot service
rag_service = RAGChatbotService()


@router.post("/rag", response_model=ChatResponse, tags=["RAG Chat"])
async def rag_chat(
    request: ChatRequest, 
    current_user: Optional[UserProfile] = Depends(get_current_user_optional)
):
    """
    RAG Chatbot endpoint for database analysis and insights
    Supports both authenticated and anonymous users
    """
    try:
        user_id = current_user.user_id if current_user else None
        response = await rag_service.process_chat(request, user_id)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat: {str(e)}"
        )


@router.get("/history", tags=["RAG Chat"])
async def get_chat_history(
    limit: int = 50,
    current_user: UserProfile = Depends(get_rag_current_user)
):
    """
    Get user's chat history
    Requires authentication
    """
    try:
        history = rag_service.get_user_chat_history(current_user.user_id, limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )


@router.get("/health", tags=["RAG Chat"])
async def chat_health_check():
    """Health check for RAG chatbot service"""
    try:
        # Test database connections
        postgres_status = "unknown"
        mongodb_status = "unknown"
        
        try:
            # Test PostgreSQL connection
            with rag_service.db_manager.get_postgres_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                postgres_status = "healthy"
        except Exception as e:
            postgres_status = f"error: {str(e)}"
        
        # Test MongoDB connection
        if rag_service.db_manager.is_mongodb_available():
            mongodb_status = "healthy"
        else:
            mongodb_status = "unavailable"
        
        return {
            "status": "healthy",
            "services": {
                "postgresql": postgres_status,
                "mongodb": mongodb_status,
                "llm": "healthy" if rag_service.llm else "unavailable"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/ask", response_model=RAGQueryResponse, tags=["RAG Chat"])
async def rag_retrieve(
    request: RAGQueryRequest,
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """RAG retrieval endpoint using MongoDB Atlas Vector Search and Groq LLM."""
    try:
        user_id = current_user.user_id if current_user else "anonymous"
        service = RAGService(user_id=user_id, concept_map_id=request.concept_map_id or "default")
        result = await service.run_rag(question=request.question, top_k=request.top_k)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["error"])

        # Normalize response
        return RAGQueryResponse(
            answer=result.get("answer") or result.get("output") or "",
            context=result.get("context"),
            question=request.question,
            top_k=request.top_k,
            user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"RAG retrieval failed: {str(e)}")
