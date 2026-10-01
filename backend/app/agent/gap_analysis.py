import json
import logging
import re
from backend.app.core.config import settings
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
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

    # Build business context summary
    ctx = state.business_context
    context_str = (
        f"Company Name: {ctx.company_name or 'Not provided'}\n"
        f"Product/Service: {ctx.product_or_service or 'Not provided'}\n"
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}\n"
        f"Target Audience: {ctx.target_audience or 'Not provided'}\n"
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}\n"
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}\n"
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}"
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

    # Construct system prompt based on extra_requirement_enabled flag
    base_prompt = (
        "You are an expert marketing strategy consultant analyzing information requirements for a business.\n"
        "Your goal is to evaluate missing marketing requirements against the known business context and past marketing document summary.\n"
        "Based ONLY on the known business context and past marketing document summary (do NOT guess user answers or invent information), determine if each "
        "requirement is relevant for this specific business.\n\n"
        "Mark a requirement as:\n"
        "- 'not_relevant': IF the business's fundamental nature explicitly makes this requirement inapplicable "
        "(e.g., 'pricing_model' for a free, donation-funded non-profit, or 'physical retail foot traffic' for a pure digital SaaS), "
        "OR IF the previous marketing document summary already contains enough clear information about that requirement so it does not need to be asked again.\n"
        "- 'still_relevant': IF the requirement could reasonably apply to this business, even if the answer is currently unknown.\n\n"
    )

    if settings.extra_requirement_enabled:
        system_prompt = (
            base_prompt +
            "EXTRA REQUIREMENT: the fixed list may miss a topic that matters a lot for this business's "
            "marketing strategy, beyond what the business context and past marketing document summary already cover. "
            "Default to null; most businesses need none. Propose one only if: (a) it strongly shapes the strategy "
            "for this business, (b) nothing in the fixed list, the business context, or the past marketing document "
            "summary already covers it, (c) it is not the product's own price and not the marketing budget. "
            "Invent a short lowercase snake_case id, a short title, and a one or two sentence neutral description "
            "that works for any kind of audience (customers, patients, donors, members). Never propose more than one.\n\n"
            "You MUST respond ONLY with a JSON object conforming strictly to this structure:\n"
            "{\n"
            '  "results": [\n'
            '    {"id": "<requirement_id>", "relevance": "still_relevant" | "not_relevant"}\n'
            '  ],\n'
            '  "extra_requirement": null OR {"id": "<id>", "title": "<title>", "description": "<description>", "reasoning": "<reasoning>"}\n'
            "}\n"
            "Do NOT include markdown formatting or commentary outside the JSON object."
        )
    else:
        system_prompt = (
            base_prompt +
            "You MUST respond ONLY with a JSON object conforming strictly to this structure:\n"
            "{\n"
            '  "results": [\n'
            '    {"id": "<requirement_id>", "relevance": "still_relevant" | "not_relevant"}\n'
            '  ]\n'
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
                relevance = item.get("relevance")

                if relevance == "not_relevant" and req_id:
                    target_req = state.get_requirement_by_id(req_id)
                    if target_req:
                        target_req.status = RequirementStatus.NOT_RELEVANT

        # Handle optional extra_requirement if enabled
        if settings.extra_requirement_enabled:
            extra_req = data.get("extra_requirement")
            if not extra_req or not isinstance(extra_req, dict):
                logger.info("Extra requirement: none")
            else:
                extra_id = extra_req.get("id")
                extra_title = extra_req.get("title")
                extra_desc = extra_req.get("description")

                rejection_reason = None
                if not extra_id or not isinstance(extra_id, str) or not ID_REGEX.match(extra_id.strip()):
                    rejection_reason = "invalid id format"
                elif not extra_title or not isinstance(extra_title, str) or not (0 < len(extra_title.strip()) <= 60):
                    rejection_reason = "invalid title"
                elif not extra_desc or not isinstance(extra_desc, str) or not (0 < len(extra_desc.strip()) <= 300):
                    rejection_reason = "invalid description"
                elif any(r.is_custom for r in state.requirements):
                    rejection_reason = "custom requirement limit reached"
                elif len(state.get_unresolved_must_haves()) + 1 > settings.max_questions:
                    rejection_reason = "exceeds max_questions limit"

                if rejection_reason:
                    logger.info(f"Extra requirement: rejected: {rejection_reason}")
                else:
                    clean_id = extra_id.strip()
                    new_req = InformationRequirement(
                        id=clean_id,
                        title=extra_title.strip(),
                        description=extra_desc.strip(),
                        status=RequirementStatus.UNKNOWN,
                        value=None,
                        is_must_have=True,
                        is_custom=True,
                    )
                    state.requirements.append(new_req)
                    logger.info(f"Extra requirement: created custom {clean_id}")

        return True

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Gap analysis failed: {e}. Leaving requirements unchanged.")
        print(f"Gap analysis error: {e}. Requirements left unchanged.")
        return False
