import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.gap_analysis import analyze_relevance
from backend.app.agent.question_selection import (
    select_next_requirement,
    generate_question,
    prepare_next_question,
    _get_default_fallback
)


def test_question_selection():
    print("=== Testing Task 5.1 & 5.2: Requirement Selection, Question Generation & State Wiring ===")

    state = MarketingAgentState()
    load_requirements_into_state(state)
    state.business_context = BusinessContext(
        company_name="CleanSeas Ocean Cleanup",
        product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
        marketing_goal="Recruit 500 volunteer cleanup captains across coastal cities",
        target_audience="Environmentally conscious college students aged 18-30"
    )

    analyze_relevance(state)
    selected_req = select_next_requirement(state)
    assert selected_req is not None, "Expected requirement selection to return a requirement."

    generated_q = generate_question(state, selected_req)
    assert generated_q is not None and len(generated_q) > 0, "Expected generated question string."

    wired_state = MarketingAgentState(business_context=state.business_context)
    load_requirements_into_state(wired_state)
    analyze_relevance(wired_state)

    q_text = prepare_next_question(wired_state)
    assert wired_state.active_requirement_id is not None
    assert wired_state.current_question is not None

    print("=== Question Selection Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_question_selection()
