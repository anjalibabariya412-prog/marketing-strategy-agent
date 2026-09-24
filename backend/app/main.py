from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api import conversation_router

app = FastAPI(title="Marketing Strategy Agent")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes (/start, /reply, /strategy)
app.include_router(conversation_router)


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Confirms the server is running and configuration loaded correctly.
    """
    return {
        "status": "ok",
        "model_in_use": settings.groq_model  # proves config is wired in
    }