from typing import Any, Dict, List, Optional
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
    profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized source profile object containing finalized profile fields."
    )
    posts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of normalized source post objects containing finalized post fields."
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

    @property
    def instagram_profile(self) -> Optional[Dict[str, Any]]:
        """Direct access to normalized Instagram profile if available."""
        return self.instagram.profile if self.instagram else None

    @property
    def instagram_posts(self) -> Optional[List[Dict[str, Any]]]:
        """Direct access to normalized Instagram posts if available."""
        return self.instagram.posts if self.instagram else None

    @property
    def facebook_profile(self) -> Optional[Dict[str, Any]]:
        """Direct access to normalized Facebook page/profile if available."""
        return self.facebook.profile if self.facebook else None

    @property
    def facebook_posts(self) -> Optional[List[Dict[str, Any]]]:
        """Direct access to normalized Facebook posts if available."""
        return self.facebook.posts if self.facebook else None


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
            if src_name == "Instagram":
                if src_obj.profile:
                    prof = src_obj.profile
                    prof_parts = []
                    uname = prof.get("username")
                    fname = prof.get("fullName")
                    if uname or fname:
                        prof_parts.append(f"@{uname}" if uname else fname)
                    if prof.get("followersCount") is not None:
                        prof_parts.append(f"{prof.get('followersCount')} followers")
                    if prof.get("postsCount") is not None:
                        prof_parts.append(f"{prof.get('postsCount')} posts")
                    if prof.get("biography"):
                        prof_parts.append(f"Bio: {prof.get('biography')}")
                    if prof_parts:
                        lines.append(f"  Instagram Profile Details: {' | '.join(prof_parts)}")
                if src_obj.posts:
                    posts = src_obj.posts
                    lines.append(f"  Instagram Posts: {len(posts)} recent post(s) tracked")
            elif src_name == "Facebook":
                if src_obj.profile:
                    prof = src_obj.profile
                    prof_parts = []
                    pname = prof.get("pageName") or prof.get("username")
                    if pname:
                        prof_parts.append(f"Page: {pname}")
                    if prof.get("followersCount") is not None:
                        prof_parts.append(f"{prof.get('followersCount')} followers")
                    if prof.get("likesCount") is not None:
                        prof_parts.append(f"{prof.get('likesCount')} likes")
                    if prof.get("about"):
                        prof_parts.append(f"About: {prof.get('about')}")
                    if prof_parts:
                        lines.append(f"  Facebook Page Details: {' | '.join(prof_parts)}")
                if src_obj.posts:
                    posts = src_obj.posts
                    lines.append(f"  Facebook Posts: {len(posts)} recent post(s) tracked")

    if not lines:
        return None

    return "\n".join(lines)
