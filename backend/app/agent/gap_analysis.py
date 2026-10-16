import json
import logging
import re
from backend.app.core.config import settings
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.information_requirement import (
    InformationRequirement,
    RequirementStatus,
    create_dynamic_requirement,
    is_duplicate_requirement,
)
from backend.app.schemas.agent.document_context import format_document_context
from backend.app.schemas.agent.online_presence import format_online_presence_context
from backend.app.services.llm_service import get_llm_response, LLMServiceError
from backend.app.prompts.gap_analysis import (
    GAP_ANALYSIS_SYSTEM_PROMPT,
    build_gap_analysis_prompt,
)

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

    # Build business context summary including past marketing document, online presence context, and structured historical document context
    ctx = state.business_context
    op_str = format_online_presence_context(state.online_presence_context)
    doc_str = format_document_context(state.document_context)

    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}",
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}",
    ]
    if ctx.past_marketing_document and not doc_str:
        context_lines.append(f"Past marketing document summary: {ctx.past_marketing_document}")
    if op_str:
        context_lines.append(f"Online presence context (scraped/verified from website & social media): {op_str}")
    if doc_str:
        context_lines.append(f"\nHISTORICAL MARKETING EVIDENCE (FROM UPLOADED DOCUMENT):\n{doc_str}")

    context_str = "\n".join(context_lines)

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

    system_prompt = GAP_ANALYSIS_SYSTEM_PROMPT
    user_prompt = build_gap_analysis_prompt(context_str, reqs_str)

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            operation="relevance_analysis",
            conversation_id=state.thread_id,
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
                            if req_id == "target_market_location" or target_req.is_must_have:
                                logger.warning(f"Requirement [{req_id}] is a mandatory must-have; cannot be marked NOT_RELEVANT.")
                                continue
                            target_req.status = RequirementStatus.NOT_RELEVANT
                            logger.info(f"Requirement [{req_id}] marked NOT_RELEVANT by gap analysis.")
                        elif relevance == "known":
                            if req_id == "target_market_location":
                                val_str = str(val or "").strip()
                                demographic_patterns = [
                                    r"^indian\s+consumers?(?:\s+aged?\s+\d+[\s–-]+\d+)?$",
                                    r"^indian\s+(?:users?|customers?|audience|people)(?:\s+aged?\s+\d+[\s–-]+\d+)?$",
                                    r"^consumers?\s+aged?\s+\d+[\s–-]+\d+$",
                                ]
                                is_demographic_only = any(re.match(p, val_str, re.IGNORECASE) for p in demographic_patterns)
                                if is_demographic_only or not val_str:
                                    logger.info(
                                        "target_market_location was marked known with demographic-only value '%s'; keeping UNKNOWN.",
                                        val_str
                                    )
                                    continue
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
