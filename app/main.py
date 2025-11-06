# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import logger
from app.api.v1.routers import api_router_v1
from app.db.mongodb_utils import init_mongodb, get_mongo_client
from fastapi.openapi.utils import get_openapi
from app.api.v1.endpoints.auth_ep import google_callback_get as v1_google_callback_get

@asynccontextmanager
async def lifespan(
    app_instance: FastAPI,
):  # Renamed app to app_instance to avoid conflict
    logger.info("BISEE AI application startup...")
    init_mongodb()

    logger.info("BISEE AI LangGraph workflows initialized.")
    yield
    logger.info("BISEE AI application shutdown...")
    mongo_cli = get_mongo_client()
    if mongo_cli:
        mongo_cli.close()
        logger.info("MongoDB connection closed.")


# FastAPI App Instance
app = FastAPI(
    title="BISEE AI API",
    description=(
        "BISEE AI - Analyze database and extract insights. "
        "Authenticate by clicking the 'Authorize' button and pasting your JWT Bearer token."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include the v1 API router
app.include_router(api_router_v1, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Welcome to BISEE AI API!",
        "description": "Transform database into visual insights and ask intelligent questions",
        "version": "2.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status"""
    from datetime import datetime
    from app.services.database_service import DatabaseManager
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "BISEE AI API", 
        "version": "2.0.0",
        "services": {}
    }
    
    # Check MongoDB
    try:
        db_manager = DatabaseManager()
        if db_manager.is_mongodb_available():
            health_status["services"]["mongodb"] = "healthy"
        else:
            health_status["services"]["mongodb"] = "unavailable"
    except Exception as e:
        health_status["services"]["mongodb"] = f"error: {str(e)}"
    
    # Check PostgreSQL (if configured)
    try:
        from app.core.config import settings
        if settings.POSTGRES_URL:
            db_manager = DatabaseManager()
            with db_manager.get_postgres_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                health_status["services"]["postgresql"] = "healthy"
        else:
            health_status["services"]["postgresql"] = "not_configured"
    except Exception as e:
        health_status["services"]["postgresql"] = f"error: {str(e)}"
    
    # Check Groq API
    try:
        from app.core.config import settings
        if settings.GROQ_API_KEY:
            health_status["services"]["groq_api"] = "configured"
        else:
            health_status["services"]["groq_api"] = "not_configured"
    except Exception as e:
        health_status["services"]["groq_api"] = f"error: {str(e)}"
    
    return health_status


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Inject security scheme
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})[
        "BearerAuth"
    ] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter your JWT in the format: Bearer <token>",
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Compatibility alias for Google OAuth callback without version prefix
app.add_api_route(
    "/auth/google/callback",
    v1_google_callback_get,
    methods=["GET"],
    include_in_schema=False,
)
