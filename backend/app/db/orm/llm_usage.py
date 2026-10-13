"""
SQLAlchemy ORM model for the llm_usage table.

Stores LLM call usage information for monitoring, debugging, and cost tracking.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .conversation import Conversation
    from .strategy_request import StrategyRequest


class LLMUsage(Base):
    """
    ORM model for the llm_usage table.

    Records individual LLM invocation details including provider, model name,
    operation, token counts, pricing metadata, estimated costs, and execution duration.
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
    # Token Counts
    # ---------------------------------------------------------------------------
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of prompt/input tokens.",
    )

    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of completion/output tokens.",
    )

    total_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens consumed.",
    )

    # ---------------------------------------------------------------------------
    # Execution Status & Duration
    # ---------------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="success",
        comment="Execution status. Expected: success, failed.",
    )

    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Call execution duration in milliseconds.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if the LLM call failed.",
    )

    # ---------------------------------------------------------------------------
    # Pricing Metadata & Calculated Costs (Precise numeric monetary values)
    # ---------------------------------------------------------------------------
    pricing_input_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Price per 1M input tokens in USD.",
    )

    pricing_output_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Price per 1M output tokens in USD.",
    )

    pricing_cached_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Price per 1M cached tokens in USD.",
    )

    pricing_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Raw token pricing rate metadata snapshot.",
    )

    input_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated input cost in USD.",
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
            f"op={self.operation!r} status={self.status!r}>"
        )
