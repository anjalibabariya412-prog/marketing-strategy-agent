"""
SQLAlchemy ORM model for the strategy_requests table.

Represents one complete request/session initiated by a user to generate
a marketing strategy.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .business import Business
    from .conversation import Conversation
    from .llm_usage import LLMUsage
    from .online_presence_source import OnlinePresenceSource


class StrategyRequest(Base):
    """
    ORM model for the strategy_requests table.

    Tracks the lifecycle of a single marketing strategy generation session.
    A StrategyRequest has:
      - 1-to-1 Business record
      - 1-to-N OnlinePresenceSource records
      - 1-to-1 Conversation record
      - 1-to-N LLMUsage records
    """

    __tablename__ = "strategy_requests"

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
    # Status / stage tracking
    # ---------------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="in_progress",
        comment=(
            "Lifecycle status of the request. "
            "Allowed values: in_progress, completed, failed, cancelled."
        ),
    )

    current_stage: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Current processing stage. Examples: initial_form, online_presence, "
            "requirement_analysis, dynamic_questions, strategy_generation, completed."
        ),
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

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ---------------------------------------------------------------------------
    # Relationships
    # ---------------------------------------------------------------------------
    business: Mapped[Optional["Business"]] = relationship(
        "Business",
        back_populates="strategy_request",
        uselist=False,
        cascade="all, delete-orphan",
    )

    online_presence_sources: Mapped[List["OnlinePresenceSource"]] = relationship(
        "OnlinePresenceSource",
        back_populates="strategy_request",
        cascade="all, delete-orphan",
    )

    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        back_populates="strategy_request",
        uselist=False,
        cascade="all, delete-orphan",
    )

    llm_usage_records: Mapped[List["LLMUsage"]] = relationship(
        "LLMUsage",
        back_populates="strategy_request",
        cascade="all, delete-orphan",
    )


    def __repr__(self) -> str:
        return (
            f"<StrategyRequest id={self.id!s} status={self.status!r} "
            f"stage={self.current_stage!r}>"
        )
