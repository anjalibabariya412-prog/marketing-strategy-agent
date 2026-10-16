from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class CompetitorSearchResult(BaseModel):
    """
    Individual search result retrieved from Tavily for a competitor.
    """
    title: str = Field(..., description="Title of the search result page.")
    url: str = Field(..., description="URL of the search result page.")
    content: str = Field(..., description="Snippet or extracted text content from the page.")
    score: Optional[float] = Field(default=None, description="Relevance score assigned by the search provider.")
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when the result was retrieved."
    )


class CompetitorAnalysisItem(BaseModel):
    """
    Structured research findings and strategic analysis for a single competitor.
    """
    competitor_name: str = Field(..., description="Identified name of the competitor.")
    user_stated_context: Optional[str] = Field(
        default=None,
        description="User-provided observations, strengths, or context regarding this competitor."
    )
    official_website: Optional[str] = Field(
        default=None,
        description="Confidently identified official website URL, or None if uncertain."
    )
    products_and_services: List[str] = Field(
        default_factory=list,
        description="Key products or services offered by the competitor."
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="Target customer segment or audience evident from web presence."
    )
    positioning: Optional[str] = Field(
        default=None,
        description="Brand positioning, messaging, or core value proposition."
    )
    pricing_information: Optional[str] = Field(
        default=None,
        description="Pricing model, price points, or positioning if evidence is available, else None."
    )
    differentiators: List[str] = Field(
        default_factory=list,
        description="Notable differentiators, unique features, or market claims."
    )
    marketing_channels_and_campaigns: List[str] = Field(
        default_factory=list,
        description="Active marketing channels, social presence, or visible campaigns."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Evidence-supported strengths and advantages."
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Evidence-supported weaknesses, limitations, or underserved customer gaps."
    )
    source_urls: List[str] = Field(
        default_factory=list,
        description="Supporting source URLs from verified search results."
    )
    research_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when research was completed."
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score (0.0 to 1.0) in the accuracy and completeness of the research."
    )
    evidence_quality: Optional[str] = Field(
        default="medium",
        description="Overall quality of retrieved evidence: 'high', 'medium', 'low', or 'incomplete'."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Concise strategic takeaway for positioning against this competitor."
    )


class CompetitorResearchReport(BaseModel):
    """
    Aggregated competitor web research report for a marketing strategy request session.
    Survives checkpointing in MarketingAgentState and persists across graph executions.
    """
    competitors: List[CompetitorAnalysisItem] = Field(
        default_factory=list,
        description="List of structured competitor analysis items."
    )
    overall_summary: Optional[str] = Field(
        default=None,
        description="High-level synthesis of competitive landscape across researched competitors."
    )
    researched_names: List[str] = Field(
        default_factory=list,
        description="List of normalized lowercase competitor names already researched to prevent duplicates."
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when the research report was created."
    )

    def is_already_researched(self, competitor_name: str) -> bool:
        """Checks if a competitor name has already been researched (case-insensitive)."""
        if not competitor_name:
            return False
        clean = competitor_name.strip().lower()
        return clean in [name.lower() for name in self.researched_names]
