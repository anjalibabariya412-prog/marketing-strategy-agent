import json
import logging
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import RequirementStatus
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant analyzing information requirements for a business.\n"
    "Your goal is to evaluate missing marketing requirements against the known business context.\n"
    "Based ONLY on the known business context (do NOT guess user answers or invent information), determine if each "
    "requirement is relevant for this specific business.\n\n"
    "Mark a requirement as:\n"
    "- 'not_relevant': IF the business's fundamental nature explicitly makes this requirement inapplicable "
    "(e.g., 'pricing_model' for a free, donation-funded non-profit, or 'physical retail foot traffic' for a pure digital SaaS).\n"
    "- 'still_relevant': IF the requirement could reasonably apply to this business, even if the answer is currently unknown.\n\n"
    "You MUST respond ONLY with a JSON object conforming strictly to this structure:\n"
    "{\n"
    '  "results": [\n'
    '    {"id": "<requirement_id>", "relevance": "still_relevant" | "not_relevant"}\n'
    "  ]\n"
    "}\n"
    "Do NOT include markdown formatting or commentary outside the JSON object."
)


def analyze_relevance(state: MarketingAgentState) -> None:
    """
    Evaluates missing information requirements in state using LLM gap analysis.
    Updates the status of inapplicable requirements to RequirementStatus.NOT_RELEVANT.
    Leaves still-relevant requirements untouched (as UNKNOWN).
    """
    missing_reqs = state.get_missing_requirements()
    if not missing_reqs:
        return

    # Build business context summary
    ctx = state.business_context
    context_str = (
        f"Company Name: {ctx.company_name or 'Not provided'}\n"
        f"Product/Service: {ctx.product_or_service or 'Not provided'}\n"
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}\n"
        f"Target Audience: {ctx.target_audience or 'Not provided'}"
    )

    # Format missing requirements for LLM evaluation
    reqs_formatted = []
    for req in missing_reqs:
        reqs_formatted.append(f"- ID: {req.id}\n  Title: {req.title}\n  Description: {req.description}")

    reqs_str = "\n".join(reqs_formatted)

    user_prompt = (
        f"BUSINESS CONTEXT:\n{context_str}\n\n"
        f"MISSING REQUIREMENTS TO EVALUATE:\n{reqs_str}\n\n"
        "Evaluate each missing requirement ID listed above and output the JSON response containing the relevance judgment."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        results = data.get("results", [])

        for item in results:
            req_id = item.get("id")
            relevance = item.get("relevance")

            if relevance == "not_relevant" and req_id:
                target_req = state.get_requirement_by_id(req_id)
                if target_req:
                    target_req.status = RequirementStatus.NOT_RELEVANT

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Gap analysis failed: {e}. Leaving requirements unchanged.")
        print(f"Gap analysis error: {e}. Requirements left unchanged.")
