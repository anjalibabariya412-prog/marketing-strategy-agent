from typing import List
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
from backend.app.models.agent_state import MarketingAgentState

REQUIREMENTS_LIBRARY: List[InformationRequirement] = [
    InformationRequirement(
        id="target_market_location",
        title="Target Market & Geographic Scope",
        description=(
            "Understand where the business wants to compete or acquire customers, "
            "including target cities, regions, states, countries, local/regional/national/international reach, "
            "and priority geographic markets."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="customer_needs_buying_behavior",
        title="Customer Needs, Motivations & Buying Behavior",
        description=(
            "Understand what target customers need, value, prefer, consider, and what motivates "
            "or influences their purchase decisions."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="competitive_landscape",
        title="Competitive Landscape & Alternatives",
        description=(
            "Understand the competitive environment relevant to the business, including direct competitors, "
            "indirect competitors, alternative solutions, competitor strengths and weaknesses, and potential market gaps."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="usp_differentiation",
        title="Positioning, USP & Differentiation",
        description=(
            "Understand what makes the product, service, or business different or valuable, including strongest benefits, "
            "unique qualities, competitive advantages, positioning, and reasons customers should choose the offering."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="pricing_offer_structure",
        title="Pricing, Offers & Commercial Model",
        description=(
            "Understand the commercial structure of the offering, including pricing, price ranges, packages, plans, "
            "discounts, promotions, subscriptions, payment structures, and other relevant commercial offers or constraints."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="sales_conversion_journey",
        title="Sales & Conversion Journey",
        description=(
            "Understand how a potential customer moves from discovering the business to becoming a customer."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
]


def load_requirements_into_state(state: MarketingAgentState) -> None:
    """
    Populates the given MarketingAgentState instance with a fresh deep copy of 
    all InformationRequirement items from REQUIREMENTS_LIBRARY.
    
    Using a deep copy prevents state mutation leakage across different conversation sessions.
    """
    state.requirements = [req.model_copy(deep=True) for req in REQUIREMENTS_LIBRARY]

