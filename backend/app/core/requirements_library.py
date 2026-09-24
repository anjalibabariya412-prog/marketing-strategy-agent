from typing import List
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
from backend.app.models.agent_state import MarketingAgentState

REQUIREMENTS_LIBRARY: List[InformationRequirement] = [
    # Must-Have Requirements (is_must_have=True)
    InformationRequirement(
        id="customer_pain_points",
        title="Customer Pain Points",
        description=(
            "The specific problems, frustrations, or unmet needs that drive a customer "
            "to seek out a solution. Understanding this is essential for crafting compelling "
            "value propositions, targeted messaging, and strategic positioning."
        ),
        is_must_have=True,
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
        id="current_marketing_channels",
        title="Current Marketing Channels",
        description=(
            "The marketing channels and promotional activities currently in use, along with "
            "their relative performance, preventing redundant efforts and leveraging active channels."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="budget_resources",
        title="Budget / Resources",
        description=(
            "The financial, time, or team resources available for marketing execution. Crucial "
            "for defining realistic, actionable, and scalable strategic recommendations."
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
            "The structure of how customers pay (e.g., one-time, subscription, tiered, "
            "usage-based, or donation), shaping market positioning, offer structure, and messaging."
        ),
        is_must_have=True,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),

    # Optional / Contextual Requirements (is_must_have=False)
    InformationRequirement(
        id="brand_tone",
        title="Brand Tone / Voice",
        description=(
            "The distinct personality, style, and tone of communication (e.g., formal, "
            "playful, authoritative, or accessible) used across marketing touchpoints. Useful "
            "for maintaining consistent content and creative direction."
        ),
        is_must_have=False,
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
    InformationRequirement(
        id="purchase_frequency",
        title="Purchase Frequency",
        description=(
            "How often customers typically buy or re-engage (e.g., one-time, occasional, "
            "or recurring), shaping whether strategy should emphasize acquisition or retention."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="geography",
        title="Geographic Focus",
        description=(
            "The geographic scope served by the business (e.g., local, regional, national, "
            "or global), influencing channel selection, targeting, and localization choices."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="sales_cycle_length",
        title="Sales Cycle Length",
        description=(
            "The typical timeframe required for a prospect to evaluate and complete a purchase "
            "decision, impacting campaign pacing, lead nurturing, and follow-up strategy."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="previous_marketing_results",
        title="Previous Marketing Results",
        description=(
            "Historical outcomes and performance data from past marketing campaigns, helping "
            "to avoid repeating past failures and build upon proven successes."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="seasonality",
        title="Seasonality",
        description=(
            "Fluctuations in customer demand or purchasing patterns tied to specific times "
            "of the year or recurring cycles, influencing promotional timing and strategy calendars."
        ),
        is_must_have=False,
        status=RequirementStatus.UNKNOWN,
        value=None,
        is_custom=False,
    ),
    InformationRequirement(
        id="customer_acquisition_method",
        title="How Customers Currently Find the Business",
        description=(
            "The primary channels through which new customers currently discover the business "
            "(e.g., referrals, search, social discovery, or foot traffic), highlighting organic strengths."
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

