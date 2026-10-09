"""
SQLAlchemy ORM model for the online_presence_sources table.

Stores the website/social media sources submitted for a strategy request
and the result of processing/scraping that source.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .strategy_request import StrategyRequest


class OnlinePresenceSource(Base):
    """
    ORM model for the online_presence_sources table.

    Tracks online sources (website, instagram, facebook, linkedin) attached to
    a strategy request, including scraping status, summary, and details.
    """

    __tablename__ = "online_presence_sources"

    __table_args__ = (
        UniqueConstraint(
            "strategy_request_id",
            "platform",
            name="uq_online_presence_sources_request_platform",
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
    # Foreign key — links to parent StrategyRequest (1-to-N)
    # ---------------------------------------------------------------------------
    strategy_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("strategy_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to strategy_requests.",
    )

    # ---------------------------------------------------------------------------
    # Source & scraping fields
    # ---------------------------------------------------------------------------
    platform: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Platform type. Expected values: website, instagram, facebook, linkedin.",
    )

    original_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original URL submitted by user.",
    )

    normalized_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Normalized URL after validation.",
    )

    scrape_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        comment="Scrape status. Expected values: success, failed, skipped, pending.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if scraping failed.",
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Extracted strategy summary from the source.",
    )

    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Structured source-specific scraped details.",
    )

    scraped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when scraping completed.",
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
    strategy_request: Mapped["StrategyRequest"] = relationship(
        "StrategyRequest",
        back_populates="online_presence_sources",
    )

    def __repr__(self) -> str:
        return (
            f"<OnlinePresenceSource id={self.id!s} platform={self.platform!r} "
            f"status={self.scrape_status!r} request_id={self.strategy_request_id!s}>"
        )
