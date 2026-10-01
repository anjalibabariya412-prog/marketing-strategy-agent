import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import conversation_router, pdf_extraction_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Marketing Strategy Agent")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes (/start, /reply, /strategy, /extract-pdf-text)
app.include_router(conversation_router)
app.include_router(pdf_extraction_router)

