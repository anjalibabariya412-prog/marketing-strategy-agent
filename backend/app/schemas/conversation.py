from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.models.business_context import MarketingBudget
from backend.app.models.marketing_strategy import MarketingStrategy


class StartRequest(BaseModel):
    """
    Request payload to start a new marketing strategy conversation using structured business details.
    """
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The description or name of the business or company."
    )
    product_or_service: str = Field(
        ...,
        min_length=1,
        max_length=750,
        description="The primary product or service offered by the business."
    )
    marketing_goal: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The primary marketing goal or objective."
    )
    target_audience: str = Field(
        ...,
        min_length=1,
        max_length=750,
        description="The target audience or ideal customer profile."
    )
    budget_resources: Optional[MarketingBudget] = Field(
        default=None,
        description="Structured marketing budget with amount (>0) and currency ('INR' or 'USD')."
    )
    current_marketing_channels: Optional[str] = Field(
        default=None,
        max_length=500,
        description="The marketing channels currently in use by the business."
    )
    website_social_links: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Website or social media links provided by the user."
    )
    past_marketing_document: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Extracted or summarized text from a previous marketing plan or document uploaded by the user."
    )


    @field_validator("company_name", "product_or_service", "marketing_goal", "target_audience", mode="before")
    @classmethod
    def strip_and_validate_required_str(cls, v: str) -> str:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Field cannot be empty or whitespace-only.")
            return stripped
        return v

    @field_validator("current_marketing_channels", "website_social_links", mode="before")
    @classmethod
    def strip_and_validate_current_marketing_channels(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v

    @field_validator("past_marketing_document", mode="before")
    @classmethod
    def strip_and_validate_past_marketing_document(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v




class StartResponse(BaseModel):
    """
    Response returned when starting a new conversation or pausing for the first question.
    """
    thread_id: str = Field(
        ...,
        description="Unique thread identifier for tracking this conversation session."
    )
    status: str = Field(
        ...,
        description="Status of the graph workflow ('waiting_for_reply' or 'completed')."
    )
    question: Optional[str] = Field(
        default=None,
        description="The clarifying question to ask the user if waiting for a reply."
    )
    requirement_id: Optional[str] = Field(
        default=None,
        description="The requirement ID associated with the clarifying question."
    )
    session_intro: Optional[str] = Field(
        default=None,
        description="Dynamic session introduction sentence based on available context."
    )


class ReplyRequest(BaseModel):
    """
    Request payload to submit a user answer and continue an existing conversation.
    """
    thread_id: str = Field(
        ...,
        min_length=1,
        description="The active conversation thread identifier."
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The user's answer to the active clarifying question."
    )


class ReplyResponse(BaseModel):
    """
    Response returned after submitting a reply and advancing the graph.
    """
    thread_id: str = Field(
        ...,
        description="The active conversation thread identifier."
    )
    status: str = Field(
        ...,
        description="Status of the graph workflow ('waiting_for_reply' or 'completed')."
    )
    question: Optional[str] = Field(
        default=None,
        description="The next clarifying question if waiting for another reply."
    )
    requirement_id: Optional[str] = Field(
        default=None,
        description="The requirement ID associated with the next clarifying question."
    )


class StrategyResponse(BaseModel):
    """
    Response payload containing the final generated marketing strategy.
    """
    thread_id: str = Field(
        ...,
        description="The active conversation thread identifier."
    )
    status: str = Field(
        default="completed",
        description="Status indicating strategy generation is complete."
    )
    strategy: MarketingStrategy = Field(
        ...,
        description="The complete structured MarketingStrategy object."
    )
