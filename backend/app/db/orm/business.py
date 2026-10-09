"""
SQLAlchemy ORM model for the businesses table.

Stores the business information submitted for a strategy request.
Has a 1-to-1 relationship with StrategyRequest.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base

if TYPE_CHECKING:
    from .strategy_request import StrategyRequest


class Business(Base):
    """
    ORM model for the businesses table.

    Stores the initial business form data submitted by a user.
    Each Business record is linked 1-to-1 with exactly one StrategyRequest.

    Note: competitor data and online_presence_summary are NOT stored here.
    Competitor information is collected via the dynamic conversation flow.
    Online presence is stored in the future online_presence_sources table.
    """

    __tablename__ = "businesses"

    __table_args__ = (
        UniqueConstraint(
            "strategy_request_id",
            name="uq_businesses_strategy_request_id",
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
    # Core business fields (match API validation limits)
    # ---------------------------------------------------------------------------
    company_name: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        comment="Name or description of the business or company. Max 1000 chars.",
    )

    product_or_service: Mapped[str] = mapped_column(
        String(750),
        nullable=False,
        comment="Primary product or service offered by the business. Max 750 chars.",
    )

    marketing_goal: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Primary marketing goal or objective. Max 500 chars.",
    )

    target_audience: Mapped[str] = mapped_column(
        String(750),
        nullable=False,
        comment="Target audience or ideal customer profile. Max 750 chars.",
    )

    # ---------------------------------------------------------------------------
    # Budget fields (nullable — user may not always provide a budget)
    # ---------------------------------------------------------------------------
    budget_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=15, scale=2),
        nullable=True,
        comment="Marketing budget amount as a decimal. Must be > 0 when provided.",
    )

    budget_currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        comment="ISO currency code for budget_amount. Expected values: INR or USD.",
    )

    # ---------------------------------------------------------------------------
    # Optional marketing channel and web presence fields
    # ---------------------------------------------------------------------------
    current_marketing_channels: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Channels currently in use (e.g. Instagram, Google Ads). Max 500 chars.",
    )

    website_social_links: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Website or social media links provided by the user. Max 2000 chars.",
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
        back_populates="business",
    )

    def __repr__(self) -> str:
        return (
            f"<Business id={self.id!s} company={self.company_name!r} "
            f"request_id={self.strategy_request_id!s}>"
        )
