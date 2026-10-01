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


from unittest.mock import patch, MagicMock
from backend.app.models.online_presence import OnlinePresenceContext, OnlineSourceSummary
from backend.app.models.marketing_strategy import MarketingStrategy


def test_strategy_generation():
    print("=== Testing Task 7.2: Strategy Generation & OnlinePresenceContext Integration ===")

    # Case 1: Without OnlinePresenceContext (Backward compatibility check)
    state_no_op = MarketingAgentState(
        business_context=BusinessContext(
            company_name="CleanSeas Ocean Cleanup",
            product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
            marketing_goal="Recruit 500 volunteer cleanup captains",
            target_audience="Environmentally conscious college students aged 18-30",
            budget_resources="$1,500 promotional budget and 2 full-time staff members."
        )
    )
    load_requirements_into_state(state_no_op)

    pain_req = state_no_op.get_requirement_by_id("customer_pain_points")
    if pain_req:
        pain_req.status = RequirementStatus.KNOWN
        pain_req.value = "Students want to help the environment but lack structured local cleanup groups."

    # Intercept LLM prompt to verify OnlinePresenceContext is NOT present in prompt when None
    captured_prompts = []
    def mock_llm_call(prompt, **kwargs):
        captured_prompts.append(prompt)
        mock_strategy = MarketingStrategy(
            business_overview="CleanSeas Ocean Cleanup recruits college student volunteers for beach cleanups.",
            target_audience_insights="College students aged 18-30 interested in environmental action.",
            competitive_positioning="Leading student-led ocean cleanup initiative in the Pacific Northwest.",
            value_proposition="Empowering students to clean ocean beaches with provided tools and community leadership.",
            marketing_channels_and_tactics="Instagram campaigns and campus club partnerships.",
            customer_acquisition_approach="On-campus informational sessions and social media sign-up drives.",
            budget_considerations="$1,500 allocated across campus flyers ($300) and Instagram ads ($1,200).",
            kpis="Recruit 500 captains at $3 cost per captain.",
            action_plan="Week 1: Launch campus posters. Week 2: Run Instagram ad campaign.",
            additional_sections={"Assumptions to Verify": "Assuming $3 CPA on Instagram ads."}
        )
        return mock_strategy.model_dump_json()

    with patch("backend.app.agent.strategy_generation.get_llm_response", side_effect=mock_llm_call):
        strat_no_op = generate_strategy(state_no_op)
        assert strat_no_op is not None
        assert "ONLINE PRESENCE CONTEXT" not in captured_prompts[-1]
        print("✓ Case 1 Passed: Strategy generation succeeds without OnlinePresenceContext")

    # Case 2: With OnlinePresenceContext (Integration check)
    op_ctx = OnlinePresenceContext(
        overall_summary="CleanSeas has a strong Instagram visual footprint showing active beach cleanup events.",
        website=OnlineSourceSummary(
            summary="CleanSeas official portal with event registration and cleanup captain guides.",
            relevant_products_or_services=["Beach Cleanup Drives", "Captain Toolkit"],
            positioning_or_messaging=["Join the wave of ocean restoration."]
        ),
        instagram=OnlineSourceSummary(
            summary="Active Instagram page with 12k followers showcasing weekly cleanup photos.",
            marketing_content=["Weekly cleanup highlights", "Volunteer spotlight posts"],
            positioning_or_messaging=["#CleanSeasCaptains"]
        )
    )

    state_with_op = MarketingAgentState(
        business_context=BusinessContext(
            company_name="CleanSeas Ocean Cleanup",
            product_or_service="Free ocean plastic cleanup drives and community environmental workshops",
            marketing_goal="Recruit 500 volunteer cleanup captains",
            target_audience="Environmentally conscious college students aged 18-30",
            budget_resources="$1,500 promotional budget and 2 full-time staff members."
        ),
        online_presence_context=op_ctx
    )
    load_requirements_into_state(state_with_op)

    with patch("backend.app.agent.strategy_generation.get_llm_response", side_effect=mock_llm_call):
        strat_with_op = generate_strategy(state_with_op)
        assert strat_with_op is not None
        last_prompt = captured_prompts[-1]
        assert "ONLINE PRESENCE CONTEXT" in last_prompt
        assert "CleanSeas official portal with event registration" in last_prompt
        assert "#CleanSeasCaptains" in last_prompt
        print("✓ Case 2 Passed: Strategy generation receives and includes OnlinePresenceContext in LLM prompt")

    print("\n=== All Strategy Generation & OnlinePresenceContext Verification Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_strategy_generation()

