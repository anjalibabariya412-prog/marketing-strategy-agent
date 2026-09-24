from typing import Optional
from pydantic import BaseModel, Field


class BusinessContext(BaseModel):
    """
    Represents the initial business context provided by the user at the start of a conversation.
    All fields are optional because a user may initially provide only partial information.
    """
    company_name: Optional[str] = Field(
        default=None,
        description="The name of the business or company."
    )
    product_or_service: Optional[str] = Field(
        default=None,
        description="The primary product or service offered by the business."
    )
    marketing_goal: Optional[str] = Field(
        default=None,
        description="The primary marketing goal or objective (e.g., increase lead generation, expand brand awareness, boost sales)."
    )
    target_audience: Optional[str] = Field(
        default=None,
        description="The target audience, ideal customer profile, or demographic targeted by the business."
    )
