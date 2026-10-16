from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.agent.marketing_strategy import MarketingStrategy
from backend.app.schemas.agent.revision_state import RevisionState


class RevisionMessageRequest(BaseModel):
    """
    Request payload submitted by the user in the revision chatbot.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's message requesting strategy adjustments or answering follow-ups."
    )

    @field_validator("message", mode="before")
    @classmethod
    def strip_and_validate_message(cls, v: str) -> str:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("Message cannot be empty or whitespace-only.")
            return stripped
        return v


class RevisionMessageResponse(BaseModel):
    """
    Response returned by the revision assistant API.
    """
    reply: str = Field(
        ...,
        description="The assistant's conversational response proposing revisions or clarifying requirements."
    )
    status: str = Field(
        default="success",
        description="Operation status, e.g. 'success'."
    )
    message_id: Optional[int] = Field(
        default=None,
        description="ID of the stored assistant response message."
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the assistant reply."
    )
    revision_state: Optional[RevisionState] = Field(
        default=None,
        description="Structured revision state and proposal details."
    )


class RevisionMessageItem(BaseModel):
    """
    Individual message item in the revision conversation history.
    """
    id: int = Field(..., description="Unique message ID.")
    sender: str = Field(..., description="Sender role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message text content.")
    created_at: datetime = Field(..., description="Timestamp of message creation.")


class RevisionHistoryResponse(BaseModel):
    """
    Full revision conversation history for a given strategy request.
    """
    strategy_request_id: str = Field(..., description="Associated strategy request UUID.")
    messages: List[RevisionMessageItem] = Field(
        default_factory=list,
        description="Chronological list of revision conversation messages."
    )
    revision_state: Optional[RevisionState] = Field(
        default=None,
        description="Latest structured revision state for this strategy request."
    )


class RevisionStateResponse(BaseModel):
    """
    Dedicated response schema for fetching current revision state.
    """
    strategy_request_id: str = Field(..., description="Associated strategy request UUID.")
    revision_state: Optional[RevisionState] = Field(
        default=None,
        description="Current structured revision state."
    )


class RevisionApplyResponse(BaseModel):
    """
    Response returned when applying confirmed revisions to the saved strategy.
    """
    status: str = Field(default="success", description="Application status, e.g. 'success' or 'already_applied'")
    message: str = Field(..., description="Human-readable status summary")
    strategy_request_id: str = Field(..., description="Strategy request UUID")
    strategy: MarketingStrategy = Field(..., description="The complete updated MarketingStrategy")
    application_status: str = Field(..., description="'applied' or 'already_applied'")
    revision_state: RevisionState = Field(..., description="Updated structured revision state")
    checkpoint_synced: bool = Field(default=False, description="Whether LangGraph checkpoint was synchronized")
