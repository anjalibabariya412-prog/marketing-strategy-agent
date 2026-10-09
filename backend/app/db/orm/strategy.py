"""
SQLAlchemy ORM model for the strategies table.

Stores complete generated marketing strategies as JSONB, supporting versioning per conversation.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .conversation import Conversation


class Strategy(Base):
    """
    ORM model for the strategies table.

    Stores versioned generated marketing strategies for a conversation.
    `strategy_json` contains the full JSONB payload matching MarketingStrategy.
    """

    __tablename__ = "strategies"

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "version",
            name="uq_strategies_conversation_id_version",
        ),
    )

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
    # Foreign key — links to parent Conversation (1-to-N)
    # ---------------------------------------------------------------------------
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to conversations.id.",
    )

    # ---------------------------------------------------------------------------
    # Versioning & Strategy Data
    # ---------------------------------------------------------------------------
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Version number of the generated strategy for this conversation.",
    )

    strategy_json: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Complete generated MarketingStrategy structure as JSONB.",
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
    # Relationship — back-reference to parent Conversation
    # ---------------------------------------------------------------------------
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="strategies",
    )

    def __repr__(self) -> str:
        return (
            f"<Strategy id={self.id!s} version={self.version} "
            f"conv_id={self.conversation_id!s}>"
        )
