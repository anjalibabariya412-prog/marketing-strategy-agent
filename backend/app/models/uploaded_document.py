"""
SQLAlchemy ORM model for the uploaded_documents table.

Stores metadata and extracted insights from uploaded PDF marketing plans,
campaign reviews, or business documents.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .strategy_request import StrategyRequest


class UploadedDocument(Base):
    """
    ORM model for the uploaded_documents table.

    Tracks uploaded documents, marketing relevance classification,
    and structured historical marketing insights.
    """

    __tablename__ = "uploaded_documents"

    # ---------------------------------------------------------------------------
    # Primary key
    # ---------------------------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ---------------------------------------------------------------------------
    # Foreign key — links to parent StrategyRequest (nullable on initial upload)
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional foreign key to strategy_requests.id. Associated during initial form submission.",
    )

    # ---------------------------------------------------------------------------
    # Document metadata
    # ---------------------------------------------------------------------------
    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Original filename of the uploaded PDF.",
    )

    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="File size in bytes.",
    )

    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Total page count of the uploaded PDF.",
    )

    # ---------------------------------------------------------------------------
    # Marketing relevance and classification
    # ---------------------------------------------------------------------------
    is_marketing_relevant: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether the document is classified as marketing-relevant.",
    )

    document_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Identified category of the document (e.g. Campaign Performance Report, Marketing Plan, Resume).",
    )

    relevance_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Reasoning for the marketing relevance classification.",
    )

    # ---------------------------------------------------------------------------
    # Summaries and structured insights
    # ---------------------------------------------------------------------------
    executive_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Executive summary of the marketing insights or explanation if non-marketing.",
    )

    pdf_insights: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Structured historical marketing insights extracted from the PDF as JSONB.",
    )

    raw_extracted_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Cleaned plain text and table content extracted from the PDF with page markers.",
    )

    # ---------------------------------------------------------------------------
    # Timestamps (timezone-aware)
    # ---------------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ---------------------------------------------------------------------------
    # Relationship — back-reference to parent StrategyRequest
    # ---------------------------------------------------------------------------
    strategy_request: Mapped[Optional["StrategyRequest"]] = relationship(
        "StrategyRequest",
        back_populates="uploaded_documents",
    )

    def __repr__(self) -> str:
        return (
            f"<UploadedDocument id={self.id!s} filename={self.filename!r} "
            f"relevant={self.is_marketing_relevant} request_id={self.strategy_request_id!s}>"
        )
