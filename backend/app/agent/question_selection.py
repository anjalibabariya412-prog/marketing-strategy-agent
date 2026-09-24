import json
import logging
from typing import Optional, List

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import InformationRequirement, RequirementStatus
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant selecting the next single information requirement to clarify with a client.\n"
    "Your goal is to evaluate the list of candidate missing requirements against the known business context and pick the SINGLE most strategic requirement to ask about next.\n\n"
    "Prioritization rules:\n"
    "1. Must-have requirements (Must Have: Yes) should generally be prioritized over optional ones.\n"
    "2. However, do NOT simply pick the first must-have item blindly. Reason about which specific must-have requirement provides the highest immediate value or clarity for THIS specific business and its current context.\n"
    "3. Occasionally, a non-must-have requirement (Must Have: No) may be unusually urgent or foundational for a particular business situation. You may pick a non-must-have item if you clearly reason why it is more critical right now than the remaining must-haves.\n\n"
    "You MUST respond ONLY with a JSON object in this exact shape:\n"
    "{\n"
    '  "selected_id": "<requirement_id>",\n'
    '  "reasoning": "<short explanation of why this requirement was selected next>"\n'
    "}\n"
    "Do NOT include markdown formatting or commentary outside the JSON object."
)


def _get_default_fallback(candidates: List[InformationRequirement]) -> InformationRequirement:
    """
    Returns the first must-have candidate if available, otherwise the first candidate in the list.
    """
    for req in candidates:
        if req.is_must_have:
            return req
    return candidates[0]


def select_next_requirement(state: MarketingAgentState) -> Optional[InformationRequirement]:
    """
    Selects the single highest-priority missing requirement from state.get_missing_requirements()
    that still has status=UNKNOWN.

    Uses LLM reasoning based on business context and already-known requirement values.
    Falls back safely to the first must-have candidate (or first candidate) if LLM fails or returns an invalid ID.
    """
    candidates = state.get_missing_requirements()
    if not candidates:
        return None

    # Build known business context summary
    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
    ]

    # Include already known requirement values if present
    known_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.KNOWN and req.value is not None
    ]
    if known_reqs:
        context_lines.append("\nALREADY KNOWN INFORMATION:")
        for req in known_reqs:
            context_lines.append(f"- {req.title}: {req.value}")

    context_str = "\n".join(context_lines)

    # Format missing candidate requirements
    reqs_formatted = []
    for req in candidates:
        must_have_label = "Yes" if req.is_must_have else "No"
        reqs_formatted.append(
            f"- ID: {req.id}\n"
            f"  Title: {req.title}\n"
            f"  Must Have: {must_have_label}\n"
            f"  Description: {req.description}"
        )

    reqs_str = "\n".join(reqs_formatted)

    user_prompt = (
        f"BUSINESS CONTEXT:\n{context_str}\n\n"
        f"CANDIDATE MISSING REQUIREMENTS:\n{reqs_str}\n\n"
        "Select the single most valuable requirement to ask about next for this business and output the JSON response."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        selected_id = data.get("selected_id")
        reasoning = data.get("reasoning", "No reasoning provided.")

        if selected_id:
            candidate_ids = {req.id for req in candidates}
            if selected_id in candidate_ids:
                selected_req = state.get_requirement_by_id(selected_id)
                if selected_req:
                    logger.info(f"Selected requirement '{selected_id}'. Reasoning: {reasoning}")
                    return selected_req

            logger.warning(
                f"LLM selected ID '{selected_id}' which is not in candidate list. Falling back to default priority."
            )
        else:
            logger.warning("LLM output missing 'selected_id'. Falling back to default priority.")

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"select_next_requirement error: {e}. Falling back to default priority.")

    fallback_req = _get_default_fallback(candidates)
    logger.info(f"Fallback selected requirement '{fallback_req.id}'.")
    return fallback_req


QUESTION_GEN_SYSTEM_PROMPT = (
    "You are a friendly, expert marketing strategy consultant speaking directly to a business owner.\n"
    "Your goal is to ask ONE clear, natural, conversational question to gather a specific piece of marketing information.\n\n"
    "Guidelines:\n"
    "1. Sound natural, warm, and conversational — NOT like a robotic form field label or survey question.\n"
    "2. Weave in specific details from the known business context (such as their company name, product, goal, or target audience) to make the question feel personalized and attentive.\n"
    "3. Focus on ONLY ONE specific information requirement at a time. Do not ask multiple unrelated questions.\n"
    "4. Return ONLY the question text itself. Do NOT surround it with quotation marks, preamble, or commentary."
)


def generate_question(state: MarketingAgentState, selected_requirement: InformationRequirement) -> str:
    """
    Generates a natural, context-aware conversational question for the selected information requirement.

    Uses LLM to frame the question personalized to the business context.
    Falls back gracefully to a simple template-based question if the LLM call fails.
    """
    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
    ]

    known_reqs = [
        req for req in state.requirements 
        if req.status == RequirementStatus.KNOWN and req.value is not None
    ]
    if known_reqs:
        context_lines.append("\nALREADY KNOWN INFORMATION:")
        for req in known_reqs:
            context_lines.append(f"- {req.title}: {req.value}")

    context_str = "\n".join(context_lines)

    user_prompt = (
        f"BUSINESS CONTEXT:\n{context_str}\n\n"
        f"INFORMATION TO COLLECT:\n"
        f"- Title: {selected_requirement.title}\n"
        f"- Description: {selected_requirement.description}\n\n"
        "Phrase ONE natural, personalized conversational question to ask the client to collect this specific information."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT
        )

        question = raw_response.strip().strip('"').strip("'").strip()
        if question:
            logger.info(f"Generated question for requirement '{selected_requirement.id}': {question}")
            return question

    except (LLMServiceError, Exception) as e:
        logger.error(f"generate_question error: {e}. Falling back to template question.")

    fallback_question = f"Could you tell me a bit about your {selected_requirement.title.lower()}?"
    logger.info(f"Fallback question used: {fallback_question}")
    return fallback_question


def prepare_next_question(state: MarketingAgentState) -> Optional[str]:
    """
    Selects the next requirement and generates the question text, updating state.current_question 
    and state.active_requirement_id accordingly.

    Returns the generated question string, or None if no missing requirements remain.
    """
    selected_req = select_next_requirement(state)
    if not selected_req:
        state.current_question = None
        state.active_requirement_id = None
        return None

    question_text = generate_question(state, selected_req)
    state.current_question = question_text
    state.active_requirement_id = selected_req.id
    return question_text
