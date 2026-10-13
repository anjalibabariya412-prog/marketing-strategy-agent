import uuid
from typing import List, Optional
from pydantic import BaseModel, Field


class ChannelInsight(BaseModel):
    """
    Historical marketing channel and its associated activities, spend, and results.
    """
    channel: str = Field(..., description="Name of the marketing channel (e.g., Meta Ads, SEO, Email, Influencer, Google Ads).")
    activities: List[str] = Field(default_factory=list, description="Specific tactics, content types, or activities conducted on this channel.")
    spend: Optional[str] = Field(default=None, description="Historical budget or spend allocated to this channel, if mentioned.")
    results: Optional[str] = Field(default=None, description="Reported metrics, leads, or engagement for this channel, if mentioned.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this channel insight is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class CampaignInsight(BaseModel):
    """
    Past marketing campaign, its timeframe, objectives, budget, and results.
    """
    campaign_name: str = Field(..., description="Name or title of the campaign.")
    period_or_dates: Optional[str] = Field(default=None, description="Timeframe or dates when the campaign ran (e.g., Q3 2023, Summer 2022).")
    objective_or_goal: Optional[str] = Field(default=None, description="Stated objective or goal of the campaign.")
    budget_or_spend: Optional[str] = Field(default=None, description="Budget or spend for this campaign.")
    key_metrics_and_results: Optional[str] = Field(default=None, description="Key performance metrics and results achieved.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this campaign is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class PerformanceResult(BaseModel):
    """
    Specific quantitative or qualitative marketing performance result.
    """
    metric_name: str = Field(..., description="Name of the metric (e.g., Reach, Impressions, Leads, Conversions, CAC, ROAS).")
    value: str = Field(..., description="Value or number achieved (e.g., '15,000 leads', '$14.50 CAC', '3.5x ROAS').")
    context_or_channel: Optional[str] = Field(default=None, description="Channel, campaign, or timeframe context for this result.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this metric is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class ActivityOutcome(BaseModel):
    """
    An activity or experiment explicitly documented as successful or unsuccessful.
    """
    activity: str = Field(..., description="The marketing activity, tactic, or experiment conducted.")
    outcome_type: str = Field(..., description="Whether the activity was 'successful', 'unsuccessful', or 'mixed/inconclusive'.")
    details: str = Field(..., description="Detailed explanation of why it succeeded or failed and what occurred.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this activity outcome is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class AudienceLearning(BaseModel):
    """
    Key learning about target customer segments, demographics, or behaviors.
    """
    segment_or_insight: str = Field(..., description="Audience segment or customer behavior observed.")
    takeaway: str = Field(..., description="Key learning, preference, or demographic finding.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this learning is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class OfferOrMessagingInsight(BaseModel):
    """
    Promotions, discounts, creative angles, or messaging tested with conditions preserved.
    """
    offer_or_message: str = Field(..., description="The promotion, discount, offer, slogan, or messaging angle tested.")
    channel_or_format: Optional[str] = Field(default=None, description="Where or how it was deployed (e.g., Landing page, Ad copy, Email banner).")
    conditions_or_qualifiers: Optional[str] = Field(default=None, description="Stated conditions, minimum purchase, promo code, or eligibility restrictions.")
    performance_note: Optional[str] = Field(default=None, description="How well this offer or messaging performed, if mentioned.")
    source_pages: List[int] = Field(default_factory=list, description="Page numbers where this offer/message is documented.")
    supporting_text: Optional[str] = Field(default=None, description="Direct supporting passage or quote from the PDF.")


class PDFInsights(BaseModel):
    """
    Structured historical marketing insights extracted from an uploaded marketing document.
    """
    document_title_or_type: Optional[str] = Field(default=None, description="Title, header, or document category identified.")
    historical_channels: List[ChannelInsight] = Field(default_factory=list, description="Historical marketing channels and activities.")
    campaigns: List[CampaignInsight] = Field(default_factory=list, description="Past campaigns, objectives, dates, and results.")
    historical_budget_and_spend: Optional[str] = Field(default=None, description="Summary of past budgets, spends, or financial allocations.")
    key_results: List[PerformanceResult] = Field(default_factory=list, description="Reported metrics such as reach, impressions, leads, CAC, ROAS.")
    successful_activities: List[ActivityOutcome] = Field(default_factory=list, description="Activities and tactics that produced positive results.")
    unsuccessful_activities: List[ActivityOutcome] = Field(default_factory=list, description="Activities and tactics that underperformed or failed.")
    audience_learnings: List[AudienceLearning] = Field(default_factory=list, description="Learnings about customer segments, demographics, or behaviors.")
    offers_and_messaging: List[OfferOrMessagingInsight] = Field(default_factory=list, description="Promotions, discounts, creative approaches, and messaging tested.")
    missing_or_unreported_information: List[str] = Field(default_factory=list, description="Key strategic marketing gaps not mentioned in the document.")
    executive_summary: Optional[str] = Field(default=None, description="A cohesive narrative synthesis of the historical marketing insights.")


class PDFExtractionResponse(BaseModel):
    """
    Response model for PDF text and insight extraction.
    """
    is_marketing_relevant: bool = Field(
        ...,
        description="Whether the uploaded document contains marketing, business growth, or campaign information."
    )
    relevance_reason: Optional[str] = Field(
        default=None,
        description="Brief explanation of why the document was classified as marketing-relevant or not."
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Identified document category (e.g., Marketing Plan, Campaign Report, Invoice, Resume)."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Executive summary of the marketing insights (or explanation if non-marketing). Backward-compatible with frontend data.summary."
    )
    pdf_insights: Optional[PDFInsights] = Field(
        default=None,
        description="Structured marketing insights extracted from the PDF if relevant."
    )
    character_count: int = Field(
        ...,
        description="The total character count of the cleaned extracted text before LLM analysis."
    )
    page_count: int = Field(
        default=1,
        description="Total number of pages in the uploaded PDF document."
    )
    document_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Unique ID of the persisted document in uploaded_documents table."
    )
    was_summarized: bool = Field(
        default=True,
        description="Indicates whether the document underwent LLM insight extraction."
    )


