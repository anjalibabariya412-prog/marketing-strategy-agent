"""
SQLAlchemy ORM model for the llm_usage table.

Stores LLM call usage information for monitoring, debugging, and cost tracking.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .conversation import Conversation
    from .strategy_request import StrategyRequest


class LLMUsage(Base):
    """
    ORM model for the llm_usage table.

    Records individual LLM invocation details including provider, model name,
    operation, token counts (input, cached input, cache write, output), costs,
    execution duration, attempt sequence, execution outcome status, and sanitized error messages.
    """

    __tablename__ = "llm_usage"

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
    # Foreign keys (nullable until associations are available)
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional foreign key to strategy_requests.id.",
    )

    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional foreign key to conversations.id (thread_id).",
    )

    # ---------------------------------------------------------------------------
    # Invocation & Model Metadata
    # ---------------------------------------------------------------------------
    operation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Operation name (e.g. relevance_analysis, question_generation, strategy_generation).",
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="openai",
        comment="LLM provider name (e.g. openai, groq, anthropic).",
    )

    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Name/identifier of LLM model used (e.g. gpt-4o-mini).",
    )

    # ---------------------------------------------------------------------------
    # Attempt & Status Tracking
    # ---------------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="success",
        server_default="success",
        index=True,
        comment="Execution outcome status (e.g. success, failed).",
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Attempt sequence number for this LLM invocation.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Safely sanitized error description or category if the call or attempt failed.",
    )

    # ---------------------------------------------------------------------------
    # Token Counts
    # ---------------------------------------------------------------------------
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of prompt/input tokens.",
    )

    cached_input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of cached input tokens reported by the API.",
    )

    cache_write_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of cache-write prompt tokens reported by the API.",
    )

    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of completion/output tokens.",
    )

    # ---------------------------------------------------------------------------
    # Duration
    # ---------------------------------------------------------------------------
    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Call execution duration in milliseconds.",
    )

    # ---------------------------------------------------------------------------
    # Calculated Costs (Precise numeric monetary values)
    # ---------------------------------------------------------------------------
    input_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated input cost in USD.",
    )

    cached_input_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated cached-input cost in USD.",
    )

    cache_write_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated cache-write cost in USD.",
    )

    output_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated output cost in USD.",
    )

    total_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Total calculated call cost in USD.",
    )

    # ---------------------------------------------------------------------------
    # Timestamps (timezone-aware)
    # ---------------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # ---------------------------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------------------------
    strategy_request: Mapped[Optional["StrategyRequest"]] = relationship(
        "StrategyRequest",
        back_populates="llm_usage_records",
    )

    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="llm_usage_records",
    )

    def __repr__(self) -> str:
        return (
            f"<LLMUsage id={self.id!s} provider={self.provider!r} model={self.model!r} "
            f"op={self.operation!r} status={self.status!r} attempt={self.attempt_number}>"
        )

