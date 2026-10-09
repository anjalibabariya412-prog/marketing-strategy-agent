"""
SQLAlchemy ORM model for the chat_messages table.

Stores individual user and agent conversation messages associated with a conversation.
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .conversation import Conversation


class ChatMessage(Base):
    """
    ORM model for the chat_messages table.

    Stores conversation messages sequentially per conversation.
    Uses an auto-incrementing BigInteger PK and indexes (conversation_id, id).
    """

    __tablename__ = "chat_messages"

    __table_args__ = (
        Index("ix_chat_messages_conversation_id_id", "conversation_id", "id"),
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
    # Message fields
    # ---------------------------------------------------------------------------
    sender: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Sender of the message. Expected: user, agent.",
    )

    message_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Type/category of message (e.g. greeting, question, answer, system, completion).",
    )

    requirement_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Non-FK reference to requirement_id associated with this message.",
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
    # Relationship — back-reference to parent Conversation
    # ---------------------------------------------------------------------------
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="chat_messages",
    )

    def __repr__(self) -> str:
        return (
            f"<ChatMessage id={self.id} sender={self.sender!r} "
            f"conv_id={self.conversation_id!s}>"
        )
