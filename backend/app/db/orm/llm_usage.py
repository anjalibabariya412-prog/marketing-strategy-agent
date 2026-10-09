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

    Records individual LLM invocation details including model name, operation,
    token counts, estimated costs, and execution latency.
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
    # Foreign keys
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to strategy_requests.id.",
    )

    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional foreign key to conversations.id.",
    )

    # ---------------------------------------------------------------------------
    # LLM Call Metadata
    # ---------------------------------------------------------------------------
    model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Name/identifier of LLM model used (e.g. gpt-4o-mini).",
    )

    operation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Operation name (e.g. relevance_analysis, question_generation, strategy_generation).",
    )

    # ---------------------------------------------------------------------------
    # Token Counts & Costs
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

    input_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated input cost.",
    )

    output_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Calculated output cost.",
    )

    total_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=12, scale=6),
        nullable=True,
        comment="Total calculated call cost.",
    )

    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Call execution duration in milliseconds.",
    )

    # ---------------------------------------------------------------------------
    # Execution Status
    # ---------------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="success",
        comment="Execution status. Expected: success, failed.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if the LLM call failed.",
    )

    # ---------------------------------------------------------------------------
    # Timestamps (timezone-aware)
    # ---------------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ---------------------------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------------------------
    strategy_request: Mapped["StrategyRequest"] = relationship(
        "StrategyRequest",
        back_populates="llm_usage_records",
    )

    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="llm_usage_records",
    )

    def __repr__(self) -> str:
        return (
            f"<LLMUsage id={self.id!s} model={self.model!r} "
            f"op={self.operation!r} status={self.status!r}>"
        )
