from typing import List, Optional
from pydantic import BaseModel, Field


class OnlineSourceSummary(BaseModel):
    """
    Summary of strategy-relevant information extracted from a specific online platform source.
    """
    summary: str = Field(
        ...,
        description="Concise summary of the source's content and online presence."
    )
    relevant_products_or_services: List[str] = Field(
        default_factory=list,
        description="List of specific products or services identified in the content."
    )
    marketing_content: List[str] = Field(
        default_factory=list,
        description="Key marketing themes, content topics, or posted materials."
    )
    positioning_or_messaging: List[str] = Field(
        default_factory=list,
        description="Value proposition, brand messaging, or slogan statements."
    )
    other_strategy_relevant_information: List[str] = Field(
        default_factory=list,
        description="Other strategy-relevant insights (e.g., audience interactions, active promos)."
    )


class OnlinePresenceContext(BaseModel):
    """
    Structured summary of a client's online presence across web and social media channels.
    """
    overall_summary: str = Field(
        ...,
        description="High-level synthesis across all processed online sources."
    )
    website: Optional[OnlineSourceSummary] = Field(
        default=None,
        description="Strategy summary extracted from website content."
    )
    instagram: Optional[OnlineSourceSummary] = Field(
        default=None,
        description="Strategy summary extracted from Instagram profile/posts."
    )
    facebook: Optional[OnlineSourceSummary] = Field(
        default=None,
        description="Strategy summary extracted from Facebook page/posts."
    )
    linkedin: Optional[OnlineSourceSummary] = Field(
        default=None,
        description="Strategy summary extracted from LinkedIn company page/posts."
    )
