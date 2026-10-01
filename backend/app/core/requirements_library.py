from typing import List
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
from backend.app.models.agent_state import MarketingAgentState

REQUIREMENTS_LIBRARY: List[InformationRequirement] = [
    # Must-Have Requirements (is_must_have=True)
    InformationRequirement(
        id="customer_pain_points",
        title="Customer Pain Points",
        description=(
            "The specific problem, frustration, or unmet need that drives someone to this business "
            "when one exists, and otherwise the occasion, motivation, or purpose that brings them "
            "(for example wanting a comfortable place to meet or work)."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="competitors",
        title="Competitors",
        description=(
            "Direct or indirect alternative options that customers might choose instead of "
            "this offering. Identifying competitors helps establish market differentiation and "
            "highlight unique competitive advantages."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="usp_differentiation",
        title="USP / Differentiation",
        description=(
            "The unique attributes, features, or positioning that make this business distinct "
            "from alternatives. Essential for establishing a clear competitive edge and value prop."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="pricing_model",
        title="Pricing Model",
        description=(
            "The price and payment structure of the product or service being sold to customers. "
            "Examples: product price, service fee, monthly subscription, yearly subscription, one-time payment, packages, etc. "
            "Note: This refers to what customers pay for the product/service, NOT the money available for marketing activities ('budget_resources')."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="sales_process",
        title="Sales / Conversion Process",
        description=(
            "The steps and mechanism through which a prospect transitions into a customer "
            "(e.g., online checkout, consultative sales call, or direct interaction). Helps "
            "in designing effective funnels and customer journeys."
        ),
        is_must_have=False,
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

