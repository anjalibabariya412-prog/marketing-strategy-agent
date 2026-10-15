"""
Pydantic schemas for the strategy revision assistant API and structured revision state.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models.marketing_strategy import MarketingStrategy


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


class ProposedSectionChange(BaseModel):
    """
    Structured modification proposed for a specific strategy section.
    """
    section_id: str = Field(
        ...,
        description="Key of the affected section (e.g. 'budget_considerations', 'marketing_channels_and_tactics')"
    )
    section_name: str = Field(
        ...,
        description="Display name of the section (e.g. 'Budget Considerations & Allocation')"
    )
    proposed_change: str = Field(
        ...,
        description="Specific adjustment or modification proposed for this section"
    )
    unchanged_aspects: Optional[str] = Field(
        default=None,
        description="Aspects or sub-items of this section that will remain intact"
    )
    replacement_content: Optional[Any] = Field(
        default=None,
        description="Optional pre-validated replacement content for direct application"
    )


class RevisionState(BaseModel):
    """
    Structured representation of the current revision conversation state.
    """
    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="in_progress",
        description="Revision status: 'in_progress', 'clarifying', 'proposal_ready', 'confirmed', 'rejected', 'applied', 'failed'"
    )
    original_request: Optional[str] = Field(
        default=None,
        description="The original user request initiating this revision conversation"
    )
    normalized_intent: Optional[str] = Field(
        default=None,
        description="Normalized description of what the user wants to accomplish"
    )
    affected_sections: List[str] = Field(
        default_factory=list,
        description="List of standard section keys affected by the revision"
    )
    proposed_changes: List[ProposedSectionChange] = Field(
        default_factory=list,
        description="Detailed section-by-section proposed modifications"
    )
    unchanged_facts: List[str] = Field(
        default_factory=list,
        description="Business facts and sections confirmed to remain unchanged"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Essential pieces of information still needed before forming a proposal"
    )
    current_clarification_question: Optional[str] = Field(
        default=None,
        description="The active clarification question asked to the user, if clarifying"
    )
    is_ready_for_confirmation: bool = Field(
        default=False,
        description="True if a complete proposal is ready and awaiting user confirmation"
    )
    confirmation_status: str = Field(
        default="pending",
        description="Confirmation state: 'pending', 'confirmed', 'rejected', 'correction'"
    )
    assumptions_and_constraints: List[str] = Field(
        default_factory=list,
        description="Important assumptions or constraints applied to the proposal"
    )
    last_updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last revision state transition"
    )
    application_status: str = Field(
        default="not_applied",
        description="Application status: 'not_applied', 'applying', 'applied', 'failed'"
    )
    applied_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the revision was successfully applied to the strategy"
    )
    applied_version: Optional[int] = Field(
        default=None,
        description="Strategy version resulting from applying this revision"
    )
    application_error: Optional[str] = Field(
        default=None,
        description="Error message if application failed"
    )
    checkpoint_synced: Optional[bool] = Field(
        default=None,
        description="Whether the LangGraph checkpoint was verified as synchronized"
    )


class LLMRevisionAnalysisResult(BaseModel):
    """
    Structured output returned by the LLM reasoning step for revision analysis.
    """
    model_config = ConfigDict(extra="ignore")

    reply: str = Field(
        ...,
        description="Conversational, user-facing markdown response text"
    )
    intent_type: str = Field(
        default="new_request",
        description="Classified intent: 'new_request', 'clarification_answer', 'confirmation', 'rejection', 'correction', 'inquiry', 'out_of_context'"
    )
    status: str = Field(
        default="in_progress",
        description="Workflow state: 'clarifying', 'proposal_ready', 'confirmed', 'rejected', 'in_progress'"
    )
    normalized_intent: Optional[str] = Field(
        default=None,
        description="Normalized user revision intent"
    )
    affected_sections: List[str] = Field(
        default_factory=list,
        description="List of affected standard section keys"
    )
    proposed_changes: List[ProposedSectionChange] = Field(
        default_factory=list,
        description="Proposed changes per affected section"
    )
    unchanged_facts: List[str] = Field(
        default_factory=list,
        description="Facts and sections preserved unchanged"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Missing essential information items, if any"
    )
    current_clarification_question: Optional[str] = Field(
        default=None,
        description="Clarification question to ask the user, if any"
    )
    is_ready_for_confirmation: bool = Field(
        default=False,
        description="Whether a proposal is formed and awaits confirmation"
    )
    confirmation_status: str = Field(
        default="pending",
        description="Confirmation status: 'pending', 'confirmed', 'rejected', 'correction'"
    )
    assumptions_and_constraints: List[str] = Field(
        default_factory=list,
        description="Assumptions or constraints associated with the proposal"
    )


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

