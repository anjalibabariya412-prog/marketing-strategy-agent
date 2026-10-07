import json
import logging
import re
from backend.app.core.config import settings
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import (
    InformationRequirement,
    RequirementStatus,
    create_dynamic_requirement,
    is_duplicate_requirement,
)
from backend.app.models.online_presence import format_online_presence_context
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

ID_REGEX = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def analyze_relevance(state: MarketingAgentState) -> bool:
    """
    Evaluates missing information requirements in state using LLM gap analysis.
    Updates the status of inapplicable requirements to RequirementStatus.NOT_RELEVANT.
    Optionally proposes a single custom extra requirement if extra_requirement_enabled is True.
    Returns True if LLM call and JSON parsing succeeded and results were applied, False otherwise.
    """
    missing_reqs = state.get_missing_requirements()
    if not missing_reqs:
        return True

    # Build business context summary including past marketing document and online presence context
    ctx = state.business_context
    op_str = format_online_presence_context(state.online_presence_context)

    context_str = (
        f"Company Name: {ctx.company_name or 'Not provided'}\n"
        f"Product/Service: {ctx.product_or_service or 'Not provided'}\n"
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}\n"
        f"Target Audience: {ctx.target_audience or 'Not provided'}\n"
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}\n"
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}\n"
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}\n"
        f"Online presence context (scraped/verified from website & social media): {op_str or 'Not provided'}"
    )

    # Format missing requirements for LLM evaluation including Must Have flag
    reqs_formatted = []
    for req in missing_reqs:
        must_have_str = "Yes" if req.is_must_have else "No"
        reqs_formatted.append(
            f"- ID: {req.id}\n"
            f"  Title: {req.title}\n"
            f"  Must Have: {must_have_str}\n"
            f"  Description: {req.description}"
        )

    reqs_str = "\n".join(reqs_formatted)

    system_prompt = (
        "You are an expert marketing strategy consultant analyzing information requirements for a business.\n"
        "Your goal is to evaluate missing marketing requirements against the known business context, past marketing document summary, "
        "and online presence context (scraped/verified from website and social media sources).\n\n"
        "CRITICAL RELEVANCE RULE — DO NOT MARK EVERYTHING RELEVANT:\n"
        "For every requirement, ask: 'Would knowing this information materially help create the marketing strategy for THIS specific business?'\n"
        "Do NOT mark every predefined requirement as relevant simply because it is in the library or generally useful for marketing.\n"
        "- If NO: Mark as 'not_relevant' (e.g. subscription_model or recurring sales cycle for a physical footwear store or single-purchase retail business).\n"
        "- If YES and information is ALREADY available in business context, past marketing document, or online presence: Mark as 'known' and provide a concise summary as 'value'.\n"
        "- If YES but information is missing: Keep as 'still_relevant'.\n\n"
        "IMPORTANT GROUNDEDNESS RULES:\n"
        "1. Only mark a requirement as 'known' or 'not_relevant' based on facts when the context explicitly supports it. Do NOT infer unsupported facts.\n"
        "2. TARGET MARKET LOCATION RULE: Do NOT infer the target market location merely from a business address, phone number, Instagram bio location, website domain, or city mentioned in a post. Target market location must be explicitly stated.\n\n"
        "DYNAMIC REQUIREMENT CREATION RULE:\n"
        "After evaluating existing requirements, perform a second check:\n"
        "'Is there an important piece of information needed for this specific business's marketing strategy that NONE of the existing requirements represents?'\n"
        "- If NO: Set 'dynamic_requirement' to null.\n"
        "- If YES: Create exactly ONE meaningful dynamic requirement for the most important uncovered strategic gap.\n"
        "ANTI-OVERGENERATION RULES FOR DYNAMIC REQUIREMENTS:\n"
        "Do NOT create a dynamic requirement:\n"
        "- just because an existing requirement could be phrased differently\n"
        "- when an existing requirement already covers the information\n"
        "- just to increase the number of questions\n"
        "- for information that is merely nice-to-have\n"
        "- when the information is already known\n"
        "- when the information is not strategically important\n"
        "Create a dynamic requirement ONLY when: strategically important + genuinely missing + specific to this business + not covered by any existing requirement.\n\n"
        "You MUST respond ONLY with a JSON object conforming strictly to this structure:\n"
        "{\n"
        '  "results": [\n'
        '    {"id": "<requirement_id>", "relevance": "still_relevant" | "not_relevant" | "known", "value": "<extracted string if known, else null>"}\n'
        '  ],\n'
        '  "dynamic_requirement": null OR {\n'
        '    "id": "<short_unique_id>",\n'
        '    "title": "<Clear Title>",\n'
        '    "description": "<Concise description of strategic gap>"\n'
        '  }\n'
        "}\n"
        "Do NOT include markdown formatting or commentary outside the JSON object."
    )

    user_prompt = (
        f"BUSINESS CONTEXT:\n{context_str}\n\n"
        f"MISSING REQUIREMENTS TO EVALUATE:\n{reqs_str}\n\n"
        "Evaluate each missing requirement ID listed above and output the JSON response containing the relevance judgment."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        results = data.get("results", [])

        for item in results:
            if isinstance(item, dict):
                req_id = item.get("id")
                relevance = item.get("relevance") or item.get("status")
                val = item.get("value")

                if req_id:
                    target_req = state.get_requirement_by_id(req_id)
                    if target_req:
                        if relevance == "not_relevant":
                            target_req.status = RequirementStatus.NOT_RELEVANT
                            logger.info(f"Requirement [{req_id}] marked NOT_RELEVANT by gap analysis.")
                        elif relevance == "known":
                            target_req.status = RequirementStatus.KNOWN
                            if val:
                                target_req.value = str(val).strip()
                            logger.info(f"Requirement [{req_id}] marked KNOWN by gap analysis with value: {val}")

        # Process dynamic requirement if proposed
        dyn_obj = data.get("dynamic_requirement")
        if isinstance(dyn_obj, dict):
            dyn_id = dyn_obj.get("id")
            dyn_title = dyn_obj.get("title")
            dyn_desc = dyn_obj.get("description")
            if dyn_id and dyn_title and dyn_desc:
                if not is_duplicate_requirement(dyn_id, dyn_title, state.requirements):
                    dyn_req = create_dynamic_requirement(
                        raw_id=dyn_id,
                        title=dyn_title,
                        description=dyn_desc,
                        is_must_have=True
                    )
                    state.requirements.append(dyn_req)
                    logger.info(f"Created dynamic requirement [{dyn_req.id}] during gap analysis: {dyn_req.title}")
                else:
                    logger.info(f"Proposed dynamic requirement [{dyn_id}] was a duplicate. Ignored.")

        return True

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Gap analysis failed: {e}. Leaving requirements unchanged.")
        print(f"Gap analysis error: {e}. Requirements left unchanged.")
        return False
