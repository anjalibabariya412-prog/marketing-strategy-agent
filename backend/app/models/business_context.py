from typing import Optional
from pydantic import BaseModel, Field


class BusinessContext(BaseModel):
    """
    Represents the initial business context provided by the user at the start of a conversation.
    All fields are optional because a user may initially provide only partial information.
    """
    company_name: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="The name of the business or company."
    )
    product_or_service: Optional[str] = Field(
        default=None,
        max_length=750,
        description="The primary product or service offered by the business."
    )
    marketing_goal: Optional[str] = Field(
        default=None,
        max_length=500,
        description="The primary marketing goal or objective (e.g., increase lead generation, expand brand awareness, boost sales)."
    )
    target_audience: Optional[str] = Field(
        default=None,
        max_length=750,
        description="The target audience, ideal customer profile, or demographic targeted by the business."
    )
    budget_resources: Optional[str] = Field(
        default=None,
        max_length=200,
        description="The marketing budget and resources (money, time, team) available for marketing activities. Note: this refers to marketing spend, NOT the price of the product or service itself."
    )
    current_marketing_channels: Optional[str] = Field(
        default=None,
        max_length=500,
        description="The marketing channels currently in use by the business (e.g., Instagram, Google Ads, offline flyers)."
    )
    website_social_links: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Website or social media links provided by the user."
    )
    past_marketing_document: Optional[str] = Field(
        default=None,
        description="Extracted or summarized text from a previous marketing plan or document uploaded by the owner."
    )




