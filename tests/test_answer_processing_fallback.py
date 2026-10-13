import os
import sys
from unittest.mock import patch

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
from backend.app.agent.answer_processing import (
    process_answer_for_requirement,
    process_answer_and_plan_next,
)
from backend.app.agent.question_selection import select_next_requirement
from backend.app.agent.sufficiency_check import is_sufficient
from backend.app.services.llm_service import LLMServiceError


def _create_test_state() -> MarketingAgentState:
    state = MarketingAgentState()
    req = InformationRequirement(
        id="target_market_location",
        title="Target Market Location",
        description="Geographic region of target customers",
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
    )
    state.requirements = [req]
    state.active_requirement_id = req.id
    state.current_question = "Which locations do you serve?"
    return state


def test_process_answer_fallback_on_llm_error():
    """
    Verifies that when LLM fails:
    1. Requirement status remains UNKNOWN (not marked KNOWN).
    2. Requirement value remains None (unvalidated answer not confirmed).
    3. User answer and question are preserved in qa_history.
    4. qa_turn.requirement_id is None so question selection does not skip it.
    5. active_requirement_id and current_question are cleared.
    6. is_sufficient() remains False.
    """
    state = _create_test_state()

    with patch("backend.app.agent.answer_processing.get_llm_response", side_effect=LLMServiceError("API timeout")):
        process_answer_for_requirement(state, "Ahmedabad and Gandhinagar")

    req = state.get_requirement_by_id("target_market_location")
    assert req is not None
    assert req.status == RequirementStatus.UNKNOWN, f"Expected UNKNOWN, got {req.status}"
    assert req.value is None, f"Expected None value, got {req.value}"

    assert len(state.qa_history) == 1
    turn = state.qa_history[0]
    assert turn.question == "Which locations do you serve?"
    assert turn.answer == "Ahmedabad and Gandhinagar"
    assert turn.requirement_id is None

    assert state.active_requirement_id is None
    assert state.current_question is None
    assert state.pending_requirement_id is None
    assert state.pending_question is None

    # Must-have is unresolved, so is_sufficient must not be True
    assert is_sufficient(state) is False

    # Requirement must not be treated as already resolved/asked by candidate selection
    candidates = [
        r for r in state.requirements
        if r.status == RequirementStatus.UNKNOWN
        and r.id not in {t.requirement_id for t in state.qa_history if t.requirement_id}
    ]
    assert "target_market_location" in [c.id for c in candidates]


def test_process_answer_fallback_on_json_decode_error():
    """
    Verifies fallback behavior when LLM returns invalid non-JSON output.
    """
    state = _create_test_state()

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value="Not a JSON string"):
        process_answer_for_requirement(state, "Ahmedabad and Gandhinagar")

    req = state.get_requirement_by_id("target_market_location")
    assert req.status == RequirementStatus.UNKNOWN
    assert req.value is None
    assert len(state.qa_history) == 1
    assert state.qa_history[0].requirement_id is None
    assert is_sufficient(state) is False


def test_merged_turn_fallback_propagates_safe_behavior():
    """
    Verifies that process_answer_and_plan_next delegates to the safe fallback
    when merged turn LLM call fails.
    """
    state = _create_test_state()

    with patch("backend.app.agent.answer_processing.get_llm_response", side_effect=LLMServiceError("Service unavailable")):
        process_answer_and_plan_next(state, "Ahmedabad and Gandhinagar")

    req = state.get_requirement_by_id("target_market_location")
    assert req.status == RequirementStatus.UNKNOWN
    assert req.value is None
    assert len(state.qa_history) == 1
    assert state.qa_history[0].requirement_id is None
    assert is_sufficient(state) is False


def test_successful_answer_processing_still_resolves():
    """
    Confirms that normal successful LLM classification still marks requirement as KNOWN.
    """
    state = _create_test_state()
    success_json = (
        '{"active_requirement": {"status": "known", "value": "Ahmedabad and Gandhinagar"}, '
        '"incidentally_known_requirements": []}'
    )

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=success_json):
        process_answer_for_requirement(state, "Ahmedabad and Gandhinagar")

    req = state.get_requirement_by_id("target_market_location")
    assert req.status == RequirementStatus.KNOWN
    assert req.value == "Ahmedabad and Gandhinagar"
    assert len(state.qa_history) == 1
    assert state.qa_history[0].requirement_id == "target_market_location"
    assert is_sufficient(state) is True
