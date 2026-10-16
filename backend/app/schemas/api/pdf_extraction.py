import uuid
from typing import Optional
from pydantic import BaseModel, Field

from backend.app.schemas.agent.pdf_insights import PDFInsights


class PDFExtractionResponse(BaseModel):
    """
    Response model for PDF text and insight extraction.
    """
    is_marketing_relevant: bool = Field(
        ...,
        description="Whether the uploaded document contains marketing, business growth, or campaign information."
    )
    relevance_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of why the document was classified as marketing-relevant or not."
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Identified document category (e.g., Marketing Plan, Campaign Report, Invoice, Resume)."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Executive summary of the marketing insights (or explanation if non-marketing). Backward-compatible with frontend data.summary."
    )
    pdf_insights: Optional[PDFInsights] = Field(
        default=None,
        description="Structured marketing insights extracted from the PDF if relevant."
    )
    character_count: int = Field(
        ...,
        description="The total character count of the cleaned extracted text before LLM analysis."
    )
    page_count: int = Field(
        default=1,
        description="Total number of pages in the uploaded PDF document."
    )
    document_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Unique ID of the persisted document in uploaded_documents table."
    )
    was_summarized: bool = Field(
        default=True,
        description="Indicates whether the document underwent LLM insight extraction."
    )
