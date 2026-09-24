import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.gap_analysis import analyze_relevance

def test_gap_analysis():
    print("--- Running Gap Analysis Test for Donation-Based Non-Profit ---")
    
    # 1. Initialize State and Load Requirements Library
    state = MarketingAgentState()
    load_requirements_into_state(state)

    # 2. Set Business Context for a Donation-Based Non-Profit
    state.business_context = BusinessContext(
        company_name="rare beatuty",
        product_or_service="selling makeup products",
        marketing_goal="increase sales",
        target_audience="women of age between 18-35"
    )

    print(f"Total Requirements Loaded: {len(state.requirements)}")
    print(f"Initial UNKNOWN count: {len(state.get_missing_requirements())}\n")

    # 3. Run Relevance Gap Analysis
    print("Executing analyze_relevance()...")
    analyze_relevance(state)

    # 4. Filter and display results
    not_relevant = [req for req in state.requirements if req.status == RequirementStatus.NOT_RELEVANT]
    still_unknown = [req for req in state.requirements if req.status == RequirementStatus.UNKNOWN]

    print("\n" + "=" * 50)
    print(f"RESULTS AFTER GAP ANALYSIS:")
    print(f"NOT_RELEVANT count ({len(not_relevant)}):")
    for req in not_relevant:
        print(f" - [{req.id}] {req.title}")

    print(f"\nSTILL UNKNOWN count ({len(still_unknown)}):")
    for req in still_unknown:
        print(f" - [{req.id}] {req.title}")
    print("=" * 50)

if __name__ == "__main__":
    test_gap_analysis()
