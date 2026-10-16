import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.schemas.agent.marketing_strategy import MarketingStrategy


def test_marketing_strategy_model():
    print("=== Testing Task 7.1: MarketingStrategy Data Model ===")

    strategy_default = MarketingStrategy()
    fields = [
        "business_overview",
        "target_audience_insights",
        "competitive_positioning",
        "value_proposition",
        "marketing_channels_and_tactics",
        "customer_acquisition_approach",
        "budget_considerations",
        "kpis",
        "action_plan",
    ]
    for f in fields:
        assert getattr(strategy_default, f) is None

    assert not hasattr(strategy_default, "additional_sections")

    # Verify backward compatibility: extra fields like additional_sections are ignored without error
    strategy_populated = MarketingStrategy(
        business_overview="CleanSeas Ocean Cleanup non-profit",
        additional_sections={"Volunteer Strategy": "Gamified volunteer rewards"}
    )
    assert strategy_populated.business_overview == ["CleanSeas Ocean Cleanup non-profit"]
    assert not hasattr(strategy_populated, "additional_sections")

    print("=== MarketingStrategy Model Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_marketing_strategy_model()
