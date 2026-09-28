import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.strategy_generation import generate_strategy


def test_strategy_generation():
    print("=== Testing Task 7.2: Strategy Generation ===")

    state = MarketingAgentState(
        business_context=BusinessContext(
            company_name="CleanSeas Ocean Cleanup",
            product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
            marketing_goal="Recruit 500 volunteer cleanup captains",
            target_audience="Environmentally conscious college students aged 18-30",
            budget_resources="$1,500 promotional budget and 2 full-time staff members."
        )
    )
    load_requirements_into_state(state)

    pain_req = state.get_requirement_by_id("customer_pain_points")
    if pain_req:
        pain_req.status = RequirementStatus.KNOWN
        pain_req.value = "Students want to help the environment but lack structured local cleanup groups."

    strategy = generate_strategy(state)
    assert strategy is not None
    assert strategy.business_overview is not None

    print("=== Strategy Generation Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_strategy_generation()
