import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext, MarketingBudget
from backend.app.models.marketing_strategy import MarketingStrategy
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.strategy_generation import generate_strategy


def test_strategy_generation_without_user_competitors():
    state = MarketingAgentState(
        business_context=BusinessContext(
            company_name="CleanSeas Ocean Cleanup",
            product_or_service="Ocean plastic cleanup drives",
            marketing_goal="Recruit 500 volunteers",
            target_audience="College students aged 18-30",
            budget_resources=MarketingBudget(amount=1500, currency="USD")
        )
    )
    load_requirements_into_state(state)

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
        strat = generate_strategy(state)
        assert strat is not None
        last_prompt = captured_prompts[-1]
        assert "Competitors / Alternatives: Not provided" in last_prompt
        assert "COMPETITOR RESEARCH CONTEXT" not in last_prompt


def test_strategy_generation_with_user_provided_competitors():
    state = MarketingAgentState(
        business_context=BusinessContext(
            company_name="Homemade Meals",
            product_or_service="Daily fresh tiffin service",
            marketing_goal="Acquire 100 monthly subscribers",
            target_audience="Working professionals in IT hubs",
            budget_resources=MarketingBudget(amount=50000, currency="INR"),
            competitors=["Brand A", "Brand B"]
        )
    )
    load_requirements_into_state(state)

    captured_prompts = []
    def mock_llm_call(prompt, **kwargs):
        captured_prompts.append(prompt)
        mock_strategy = MarketingStrategy(
            business_overview="Homemade Meals provides fresh tiffin service.",
            target_audience_insights="Working IT professionals in tech parks.",
            competitive_positioning="Homemade Meals positions against Brand A and Brand B by offering daily home-style subscription meals.",
            value_proposition="Fresh, hygienic home-style daily tiffin delivered directly to office desks.",
            marketing_channels_and_tactics="WhatsApp marketing and tech park flyer distribution.",
            customer_acquisition_approach="Free trial lunch boxes distributed at IT hub offices.",
            budget_considerations="₹50,000 allocated for flyer distribution (₹10,000) and trial boxes (₹40,000).",
            kpis="Acquire 100 subscribers at ₹500 CPA.",
            action_plan="Week 1: Office flyer distribution. Week 2: Trial lunch box delivery.",
            additional_sections={"Assumptions to Verify": "Assuming ₹500 CPA based on trial box conversion rate."}
        )
        return mock_strategy.model_dump_json()

    with patch("backend.app.agent.strategy_generation.get_llm_response", side_effect=mock_llm_call):
        strat = generate_strategy(state)
        assert strat is not None
        last_prompt = captured_prompts[-1]
        assert "Competitors / Alternatives: Brand A, Brand B" in last_prompt
        assert "PUBLIC WEB RESEARCH" not in last_prompt
