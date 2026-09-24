from backend.app.core.config import settings
from backend.app.models.agent_state import MarketingAgentState


def is_sufficient(state: MarketingAgentState) -> bool:
    """
    Determines whether sufficient information has been collected to stop asking questions
    and proceed to generating the marketing strategy.

    This function is purely deterministic rule-based logic with zero LLM or external API calls.

    Returns True if:
    1. state.get_unresolved_must_haves() returns an empty list (all must-have requirements are KNOWN, UNAVAILABLE, or NOT_RELEVANT), OR
    2. len(state.qa_history) >= settings.max_questions (the maximum allowed question count has been reached).

    Otherwise returns False.
    """
    unresolved_must_haves = state.get_unresolved_must_haves()
    if not unresolved_must_haves:
        return True

    if len(state.qa_history) >= settings.max_questions:
        return True

    return False
