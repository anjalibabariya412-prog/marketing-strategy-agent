from typing import Dict, Optional
from pydantic import BaseModel, Field


class MarketingStrategy(BaseModel):
    """
    Represents the final structured marketing strategy synthesized by the agent.
    All fields are optional because filler should not be forced if information is insufficient.
    """
    business_overview: Optional[str] = Field(
        default=None,
        description="High-level overview of the business, its core market, and strategic context."
    )
    target_audience_insights: Optional[str] = Field(
        default=None,
        description="Deep insights into customer personas, pain points, motivations, and behaviors."
    )
    competitive_positioning: Optional[str] = Field(
        default=None,
        description="Market positioning relative to competitors and key differentiators."
    )
    value_proposition: Optional[str] = Field(
        default=None,
        description="Core messaging angles and unique value proposition statements."
    )
    marketing_channels_and_tactics: Optional[str] = Field(
        default=None,
        description="Recommended marketing channels (e.g. SEO, LinkedIn, Email) and tactical execution details."
    )
    customer_acquisition_approach: Optional[str] = Field(
        default=None,
        description="Funnel strategy and tactics for turning prospects into paying customers."
    )
    budget_considerations: Optional[str] = Field(
        default=None,
        description="Practical budget allocation advice and resource prioritization."
    )
    kpis: Optional[str] = Field(
        default=None,
        description="Key performance indicators and metrics to measure strategy success."
    )
    action_plan: Optional[str] = Field(
        default=None,
        description="Phased implementation roadmap and action steps."
    )
    additional_sections: Optional[Dict[str, str]] = Field(
        default=None,
        description="Domain-specific or custom strategic sections relevant to the business."
    )
