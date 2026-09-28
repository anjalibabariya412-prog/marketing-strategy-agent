import json
import logging
from pydantic import ValidationError

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import RequirementStatus
from backend.app.models.marketing_strategy import MarketingStrategy
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)


class StrategyGenerationError(Exception):
    """Raised when marketing strategy generation or parsing fails."""
    pass


SYSTEM_PROMPT = (
    "You are a senior, highly pragmatic marketing strategy consultant creating an actionable, tailored marketing strategy for a client.\n\n"
    "Your goal is to synthesize the provided business context and collected facts into a structured, highly specific marketing strategy.\n\n"
    "CRITICAL GUIDELINES:\n"
    "1. SPECIFICITY OVER GENERALITY: Ground every recommendation in the specific business details, audience, and facts provided. "
    "Do NOT use vague, generic fluff or one-size-fits-all phrases (e.g. 'high-quality, affordable solutions' or 'leveraging social media'). "
    "Name concrete tactics, messaging angles, and channels relevant to THIS specific business.\n"
    "2. DISTINGUISH FACTS VS RECOMMENDATIONS: You must clearly distinguish between two types of content in every section:\n"
    "   - FACTS the business owner explicitly provided (from the KNOWN requirements/business context given to you) — state these plainly as established facts.\n"
    "   - RECOMMENDATIONS you are generating as a consultant — these must be phrased using clear recommendation language: 'should', 'could', 'consider', 'we recommend', 'an effective approach would be', etc. NEVER state a recommendation as if it were an existing fact or current practice of the business.\n"
    "   For example, if the business never mentioned using influencers, write 'Sweet Crumbs should consider partnering with local micro-influencers' — NEVER 'Sweet Crumbs works with micro-influencers' or 'Sweet Crumbs partners with 5 micro-influencers', since that falsely implies this is something already happening.\n"
    "   This rule applies especially to marketing_channels_and_tactics, customer_acquisition_approach, action_plan, and additional_sections, where new tactics are being proposed rather than summarized from known facts.\n"
    "3. NO FORCED FILLER: Fill in ONLY the standard strategy sections that are genuinely relevant and supportable given the known information. "
    "If a section is not applicable or cannot be supportably crafted from what is known (e.g. competitive positioning for a unique entity with no competitors), "
    "set that field to null/None rather than inventing generic content.\n"
    "4. WORK AROUND UNAVAILABLE INFO: If specific information was marked UNAVAILABLE (e.g. exact budget unknown), provide practical, "
    "adaptable strategic guidance rather than making up precise numbers.\n"
    "5. ADDITIONAL SECTIONS: If a specialized, domain-specific section would genuinely add high strategic value for this business "
    "(e.g., 'Volunteer Engagement Strategy' for a non-profit), include it in the 'additional_sections' key as a key-value pair. Otherwise, set 'additional_sections' to null.\n\n"
    "REQUIRED JSON STRUCTURE:\n"
    "You MUST respond ONLY with a JSON object with the following schema:\n"
    "{\n"
    '  "business_overview": "string or null",\n'
    '  "target_audience_insights": "string or null",\n'
    '  "competitive_positioning": "string or null",\n'
    '  "value_proposition": "string or null",\n'
    '  "marketing_channels_and_tactics": "string or null",\n'
    '  "customer_acquisition_approach": "string or null",\n'
    '  "budget_considerations": "string or null",\n'
    '  "kpis": "string or null",\n'
    '  "action_plan": "string or null",\n'
    '  "additional_sections": {"Section Title": "Section Content"} or null\n'
    "}\n"
    "Do NOT include markdown code fences or commentary outside the JSON object."
)


def generate_strategy(state: MarketingAgentState) -> MarketingStrategy:
    """
    Generates a structured MarketingStrategy object based on the state's business context,
    KNOWN requirement values, and UNAVAILABLE requirement statuses.

    Does NOT modify state directly.
    Raises StrategyGenerationError if generation or schema parsing fails.
    """
    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
    ]

    # Collect facts from KNOWN requirements
    known_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.KNOWN and req.value is not None
    ]
    known_lines = []
    for req in known_reqs:
        known_lines.append(f"- {req.title} [{req.id}]: {req.value}")

    # Collect facts from UNAVAILABLE requirements
    unavailable_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.UNAVAILABLE
    ]
    unavailable_lines = []
    for req in unavailable_reqs:
        unavailable_lines.append(f"- {req.title} [{req.id}]: User indicated this information is currently unavailable/undecided.")

    # Assemble user prompt
    prompt_parts = [
        "BUSINESS CONTEXT:",
        "\n".join(context_lines)
    ]

    if known_lines:
        prompt_parts.extend([
            "\nCOLLECTED FACTS (KNOWN INFORMATION):",
            "\n".join(known_lines)
        ])

    if unavailable_lines:
        prompt_parts.extend([
            "\nUNAVAILABLE INFORMATION (CLIENT DID NOT HAVE DATA):",
            "\n".join(unavailable_lines)
        ])

    prompt_parts.append(
        "\nSynthesize all available context above into a high-quality, tailored marketing strategy JSON object."
    )

    user_prompt = "\n".join(prompt_parts)

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        strategy = MarketingStrategy.model_validate(data)
        logger.info("Successfully generated and validated MarketingStrategy.")
        return strategy

    except (LLMServiceError, json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"Failed to generate strategy: {e}")
        raise StrategyGenerationError(f"Failed to generate marketing strategy: {e}") from e
