from typing import List, Optional
from pydantic import BaseModel, Field

from .business_context import BusinessContext
from .information_requirement import InformationRequirement, RequirementStatus
from .qa_turn import QATurn
from .marketing_strategy import MarketingStrategy


class MarketingAgentState(BaseModel):
    """
    Central state object representing the full state of the Marketing Strategy Agent during a conversation session.
    Combines basic context, tracked information requirements, conversation Q&A history, active interaction status,
    sufficiency evaluation, and thread tracking.
    """
    business_context: BusinessContext = Field(
        default_factory=BusinessContext,
        description="The basic business context provided at the start of the session."
    )
    requirements: List[InformationRequirement] = Field(
        default_factory=list,
        description="List of all tracked information requirements (both base framework and dynamic)."
    )
    qa_history: List[QATurn] = Field(
        default_factory=list,
        description="Chronological log of all Q&A turns exchanged with the user."
    )
    current_question: Optional[str] = Field(
        default=None,
        description="The question currently awaiting a user response, if any."
    )
    active_requirement_id: Optional[str] = Field(
        default=None,
        description="The ID of the InformationRequirement that current_question is attempting to clarify, if any."
    )
    is_sufficient: bool = Field(
        default=False,
        description="True if sufficient information has been collected to generate a marketing strategy."
    )
    analysis_done: bool = Field(
        default=False,
        description="Tracks whether the one-time relevance analysis has already run for this conversation."
    )
    pending_question: Optional[str] = Field(
        default=None,
        description="The pre-planned next question generated in a merged turn call, waiting to be served."
    )
    pending_requirement_id: Optional[str] = Field(
        default=None,
        description="The requirement ID associated with pending_question."
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Unique session/conversation thread identifier for state persistence and LangGraph checkpointing."
    )
    final_strategy: Optional[MarketingStrategy] = Field(
        default=None,
        description="The final generated marketing strategy, populated once sufficient information has been collected and generate_strategy() has run."
    )


    def get_requirement_by_id(self, requirement_id: str) -> Optional[InformationRequirement]:
        """
        Search through self.requirements and return the matching InformationRequirement if found, or None if not found.
        """
        for req in self.requirements:
            if req.id == requirement_id:
                return req
        return None

    def get_missing_requirements(self) -> List[InformationRequirement]:
        """
        Return a list of all InformationRequirement items from self.requirements whose status is UNKNOWN.
        """
        return [req for req in self.requirements if req.status == RequirementStatus.UNKNOWN]

    def get_unresolved_must_haves(self) -> List[InformationRequirement]:
        """
        Return a list of all InformationRequirement items from self.requirements where is_must_have is True and status is UNKNOWN.
        """
        return [req for req in self.requirements if req.is_must_have and req.status == RequirementStatus.UNKNOWN]



