from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class CurrencyEnum(str, Enum):
    INR = "INR"
    USD = "USD"


class MarketingBudget(BaseModel):
    """
    Structured representation of marketing budget.
    Restricted strictly to INR or USD currencies and positive amounts.
    """
    amount: float = Field(
        ...,
        gt=0,
        description="Marketing budget amount, must be greater than 0."
    )
    currency: CurrencyEnum = Field(
        ...,
        description="Currency code for marketing budget. Restricted strictly to INR or USD."
    )

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper not in ("INR", "USD"):
                raise ValueError("Currency must be either 'INR' or 'USD'.")
            return v_upper
        return v

    def __str__(self) -> str:
        symbol = "₹" if self.currency == CurrencyEnum.INR else "$"
        amt_str = f"{self.amount:,.0f}" if float(self.amount).is_integer() else f"{self.amount:,.2f}"
        return f"{symbol}{amt_str}"


class BusinessContext(BaseModel):
    """
    Represents the initial business context provided by the user at the start of a conversation.
    All fields are optional because a user may initially provide only partial information.
    """
    company_name: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="The description or name of the business or company."
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
    budget_resources: Optional[MarketingBudget] = Field(
        default=None,
        description="Structured marketing budget with amount (>0) and currency ('INR' or 'USD')."
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
    competitors: List[str] = Field(
        default_factory=list,
        description="Extracted list of competitor names, brand names, or alternative solutions."
    )





