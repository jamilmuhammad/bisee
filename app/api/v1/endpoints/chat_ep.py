# from fastapi import APIRouter, Body, HTTPException, Depends

# from app.core.config import logger, settings
# from app.models.user_models import UserModelInDB
# from app.models.chat_models import (
#     ChatHistoryResponse,
# )

# from app.models.cmvs_models import NodeDetailResponse
# from app.api.v1.deps import get_current_active_user
# from app.services.chat_service import ChatService
# from app.db.mongodb_utils import get_db
# from bson import ObjectId

# router = APIRouter()


# @router.post("/", response_model=NodeDetailResponse, tags=["BISEE AI RAG"])
# async def ask_question_endpoint(
#     current_user: UserModelInDB = Depends(get_current_active_user),
#     question: str = Body(..., description="The user's question"),
#     node_label: str = Body(
#         None, description="Label of the node (for chat history context)"
#     ),
#     top_k: int = Body(10, description="Number of relevant chunks to retrieve"),
# ):
#     """
#     Unified endpoint for asking questions about mind maps.

#     Flow:
#     1. Check if the exact question already exists in chat history for this node
#     2. If found, return the cached answer from history
#     3. If not found, run RAG workflow and save both question and answer to history
#     """
#     if not question:
#         raise HTTPException(status_code=400, detail="A question is required.")

#     logger.info(f"User '{current_user.email}' asking: '{question[:100]}...'")

#     try:
#         # Verify the mind map exists and belongs to the user
#         db = get_db()

#         # Initialize chat service for history management
#         from app.services.chat_service import ChatService
#         from app.models.chat_models import ChatMessage
#         import uuid
#         from datetime import datetime, timezone

#         chat_service = ChatService()

#         # Question not found in history, run RAG workflow
#         logger.info(f"Running RAG workflow for new question: '{question[:50]}...'")
#         bisee_service = BISEEAIService()
#         response = await bisee_service.query_mind_map(
#             user_id=current_user.id,
#             session_id=session_id,
#             query=enhanced_query,
#             top_k=top_k,
#         )

#         # Save both question and answer to chat history if node info provided
#         if node_id and node_label:
#             # Save the question
#             question_message = ChatMessage(
#                 id=str(uuid.uuid4()),
#                 type="question",
#                 content=question,
#                 cited_sources=[],
#                 timestamp=datetime.now(timezone.utc),
#                 node_id=node_id,
#                 user_id=current_user.id,
#                 map_id=map_id,
#             )

#             await chat_service.save_message(
#                 user_id=current_user.id,
#                 map_id=map_id,
#                 node_id=node_id,
#                 node_label=node_label,
#                 message=question_message,
#             )

#             # Save the answer
#             answer_message = ChatMessage(
#                 id=str(uuid.uuid4()),
#                 type="answer",
#                 content=response.answer,
#                 cited_sources=[
#                     source.model_dump() for source in response.cited_sources
#                 ],
#                 timestamp=datetime.now(timezone.utc),
#                 node_id=node_id,
#                 user_id=current_user.id,
#                 map_id=map_id,
#             )

#             await chat_service.save_message(
#                 user_id=current_user.id,
#                 map_id=map_id,
#                 node_id=node_id,
#                 node_label=node_label,
#                 message=answer_message,
#             )

#             logger.info(
#                 f"New question and answer saved to chat history for node {node_id}"
#             )

#         return response

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"API Error in ask_question_endpoint: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500, detail=f"An internal server error occurred: {e}"
#         )


# @router.delete(
#     "/delete/{map_id}/{node_id}",
#     response_model=ChatHistoryResponse,
#     tags=["Chat History"],
# )
# async def delete_chat_history_endpoint(
#     map_id: str,
#     node_id: str,
#     current_user: UserModelInDB = Depends(get_current_active_user),
# ):
#     """
#     Soft delete chat history for a specific node.
#     """
#     try:
#         # Validate map_id format
#         if not ObjectId.is_valid(map_id):
#             raise HTTPException(status_code=400, detail="Invalid map ID format")

#         # Verify the mind map exists and belongs to the user
#         db = get_db()
#         cm_collection = db[settings.MONGODB_MAPS_COLLECTION]
#         map_doc = cm_collection.find_one(
#             {"_id": ObjectId(map_id), "user_id": current_user.id}
#         )

#         if not map_doc:
#             raise HTTPException(status_code=404, detail="Mind map not found")

#         chat_service = ChatService()
#         result = await chat_service.soft_delete_conversation(
#             user_id=current_user.id, map_id=map_id, node_id=node_id
#         )

#         return result

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting chat history: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")
