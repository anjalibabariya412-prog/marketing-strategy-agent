import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.answer_processing import process_answer_and_plan_next, process_answer_for_requirement
from backend.app.core.config import settings


def test_merged_turn_state_fields():
    print("=== Testing Task M1.2: Merged Turn Call Optimization ===")

    # 1. Test state fields default and backward compatibility
    state = MarketingAgentState()
    assert state.pending_question is None
    assert state.pending_requirement_id is None
    print("✓ State pending_question and pending_requirement_id default to None.")

    # 2. Test state instantiation from legacy dict (without pending fields)
    legacy_dict = {
        "business_context": {"company_name": "Test Co"},
        "requirements": [],
        "qa_history": [],
        "is_sufficient": False,
        "analysis_done": True,
    }
    restored_state = MarketingAgentState.model_validate(legacy_dict)
    assert restored_state.pending_question is None
    assert restored_state.pending_requirement_id is None
    print("✓ Legacy saved state loads successfully without pending fields.")

    # 3. Test configuration flag
    assert hasattr(settings, "merged_turn_call_enabled")
    print(f"✓ Config settings.merged_turn_call_enabled = {settings.merged_turn_call_enabled}")

    print("=== Merged Turn Unit Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_merged_turn_state_fields()
