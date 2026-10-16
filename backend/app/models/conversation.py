"""
SQLAlchemy ORM model for the conversations table.

Represents the dynamic-question conversation associated with a strategy request.
The primary key `id` is the UUID used as the LangGraph thread_id.
"""

import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .chat_message import ChatMessage
    from .information_requirement import InformationRequirement
    from .llm_usage import LLMUsage
    from .strategy import Strategy
    from .strategy_request import StrategyRequest


class Conversation(Base):
    """
    ORM model for the conversations table.

    Each Conversation is linked 1-to-1 with a StrategyRequest.
    The `id` field serves as both the PK and the LangGraph thread_id.
    """

    __tablename__ = "conversations"

    __table_args__ = (
        UniqueConstraint(
            "strategy_request_id",
            name="uq_conversations_strategy_request_id",
        ),
    )

    # ---------------------------------------------------------------------------
    # Primary key — matches LangGraph thread_id
    # ---------------------------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Primary key, also serves as the LangGraph thread_id.",
    )

    # ---------------------------------------------------------------------------
    # Foreign key — links to parent StrategyRequest (1-to-1)
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to strategy_requests. Unique — enforces 1-to-1 relationship.",
    )

    # ---------------------------------------------------------------------------
    # Conversation state & tracking
    # ---------------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="waiting_for_reply",
        comment="Status of the conversation. Expected: waiting_for_reply, completed, failed.",
    )

    current_requirement_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="ID of the requirement currently being asked (string, non-FK).",
    )

    is_sufficient: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if information gathered is sufficient to generate strategy.",
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
    # Relationships
    # ---------------------------------------------------------------------------
    strategy_request: Mapped["StrategyRequest"] = relationship(
        "StrategyRequest",
        back_populates="conversation",
    )

    information_requirements: Mapped[List["InformationRequirement"]] = relationship(
        "InformationRequirement",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    chat_messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    strategies: Mapped[List["Strategy"]] = relationship(
        "Strategy",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    llm_usage_records: Mapped[List["LLMUsage"]] = relationship(
        "LLMUsage",
        back_populates="conversation",
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation id={self.id!s} status={self.status!r} "
            f"request_id={self.strategy_request_id!s}>"
        )
