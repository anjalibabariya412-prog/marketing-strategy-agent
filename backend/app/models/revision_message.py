"""
SQLAlchemy ORM model for the revision_messages table.

Stores individual user and assistant messages for strategy revisions
associated directly with a StrategyRequest.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .strategy_request import StrategyRequest


class RevisionMessage(Base):
    """
    ORM model for the revision_messages table.

    Stores conversation messages sequentially per strategy request for strategy revisions.
    Uses an auto-incrementing BigInteger PK and indexes (strategy_request_id, id).
    """

    __tablename__ = "revision_messages"

    __table_args__ = (
        Index("ix_revision_messages_strategy_request_id_id", "strategy_request_id", "id"),
    )

    # ---------------------------------------------------------------------------
    # Primary key — auto-incrementing BigInteger
    # ---------------------------------------------------------------------------
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )

    # ---------------------------------------------------------------------------
    # Foreign key — links to parent StrategyRequest (1-to-N)
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to strategy_requests.id.",
    )

    # ---------------------------------------------------------------------------
    # Message fields
    # ---------------------------------------------------------------------------
    sender: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Sender of the message. Expected: user, assistant.",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full message text content.",
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
    # Relationship — back-reference to parent StrategyRequest
    # ---------------------------------------------------------------------------
    strategy_request: Mapped["StrategyRequest"] = relationship(
        "StrategyRequest",
        back_populates="revision_messages",
    )

    def __repr__(self) -> str:
        return (
            f"<RevisionMessage id={self.id} sender={self.sender!r} "
            f"req_id={self.strategy_request_id!s}>"
        )
