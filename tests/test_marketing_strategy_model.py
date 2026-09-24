import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.models.marketing_strategy import MarketingStrategy


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
        "additional_sections",
    ]
    for f in fields:
        assert getattr(strategy_default, f) is None

    strategy_populated = MarketingStrategy(
        business_overview="CleanSeas Ocean Cleanup non-profit",
        additional_sections={"Volunteer Strategy": "Gamified volunteer rewards"}
    )
    assert strategy_populated.business_overview == "CleanSeas Ocean Cleanup non-profit"
    assert strategy_populated.additional_sections["Volunteer Strategy"] == "Gamified volunteer rewards"

    print("=== MarketingStrategy Model Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_marketing_strategy_model()
