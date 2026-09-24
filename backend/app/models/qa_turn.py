from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class QATurn(BaseModel):
    """
    Represents a single question-and-answer exchange between the agent and the user during the conversation.
    """
    question: str = Field(
        ...,
        description="The question asked by the agent."
    )
    answer: str = Field(
        ...,
        description="The response provided by the user."
    )
    requirement_id: Optional[str] = Field(
        default=None,
        description="Optional ID linking this exchange to a specific InformationRequirement (e.g., 'pricing_model')."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="The UTC timestamp when this Q&A exchange occurred."
    )
