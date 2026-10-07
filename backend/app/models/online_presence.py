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


def format_online_presence_context(op: Optional[OnlinePresenceContext]) -> Optional[str]:
    """
    Formats OnlinePresenceContext into a clean multi-line string for LLM prompts.
    Returns None if op is None or contains no meaningful summary information.
    """
    if not op:
        return None

    lines = []
    if op.overall_summary and op.overall_summary.strip():
        lines.append(f"Overall Online Summary: {op.overall_summary.strip()}")

    sources = [
        ("Website", op.website),
        ("Instagram", op.instagram),
        ("Facebook", op.facebook),
        ("LinkedIn", op.linkedin),
    ]

    for src_name, src_obj in sources:
        if src_obj and src_obj.summary and src_obj.summary.strip():
            lines.append(f"{src_name} Summary: {src_obj.summary.strip()}")
            if src_obj.relevant_products_or_services:
                lines.append(f"  {src_name} Products/Services: {', '.join(src_obj.relevant_products_or_services)}")
            if src_obj.positioning_or_messaging:
                lines.append(f"  {src_name} Messaging/USP: {', '.join(src_obj.positioning_or_messaging)}")
            if src_obj.marketing_content:
                lines.append(f"  {src_name} Marketing Content: {', '.join(src_obj.marketing_content)}")
            if src_obj.other_strategy_relevant_information:
                lines.append(f"  {src_name} Other Insights: {', '.join(src_obj.other_strategy_relevant_information)}")

    if not lines:
        return None

    return "\n".join(lines)
