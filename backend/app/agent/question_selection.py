import json
import logging
from typing import Optional, List

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

logger = logging.getLogger(__name__)

from backend.app.prompts.question_selection import (
    QUESTION_SELECTION_SYSTEM_PROMPT,
    QUESTION_GEN_SYSTEM_PROMPT,
    FALLBACK_QUESTIONS,
    FIRST_MESSAGE_SYSTEM_PROMPT,
    build_question_selection_prompt,
    build_question_generation_prompt,
    build_first_message_prompt,
    build_grounded_business_summary,
)

SYSTEM_PROMPT = QUESTION_SELECTION_SYSTEM_PROMPT



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
    Selects an existing UNKNOWN requirement from the fixed library based on business context,
    past marketing document, online presence context, and conversation history.

    Returns the selected InformationRequirement instance, or None if no missing requirements remain.
    """
    already_asked_or_resolved_ids = {
        req.id for req in state.requirements 
        if req.status != RequirementStatus.UNKNOWN
    }
    if state.qa_history:
        for turn in state.qa_history:
            if turn.requirement_id:
                already_asked_or_resolved_ids.add(turn.requirement_id)
    if state.active_requirement_id:
        already_asked_or_resolved_ids.add(state.active_requirement_id)

    candidates = [
        req for req in state.requirements 
        if req.status == RequirementStatus.UNKNOWN and req.id not in already_asked_or_resolved_ids
    ]

    # Check question limit
    if len(state.qa_history) >= settings.max_questions:
        logger.info("Question limit reached in select_next_requirement. Returning None.")
        return None

    if not candidates:
        logger.info("No remaining candidate requirements. Returning None.")
        return None

    # Build known business context summary
    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}",
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}",
    ]
    doc_str = format_document_context(state.document_context)
    if ctx.past_marketing_document and not doc_str:
        context_lines.append(f"Past marketing document summary: {ctx.past_marketing_document}")
    if doc_str:
        context_lines.append(f"\nHISTORICAL MARKETING EVIDENCE (FROM UPLOADED DOCUMENT):\n{doc_str}")

    op_str = format_online_presence_context(state.online_presence_context)
    if op_str:
        context_lines.append(f"\nONLINE PRESENCE CONTEXT (scraped/verified website & social media sources):\n{op_str}")

    # Include already known / resolved requirement values if present
    resolved_reqs = [
        req for req in state.requirements 
        if req.status != RequirementStatus.UNKNOWN
    ]
    if resolved_reqs:
        context_lines.append("\nALREADY RESOLVED REQUIREMENTS:")
        for req in resolved_reqs:
            if req.status == RequirementStatus.KNOWN and req.value:
                context_lines.append(f"- {req.title} ({req.id}): KNOWN -> {req.value}")
            elif req.status == RequirementStatus.UNAVAILABLE:
                context_lines.append(f"- {req.title} ({req.id}): UNAVAILABLE")
            elif req.status == RequirementStatus.NOT_RELEVANT:
                context_lines.append(f"- {req.title} ({req.id}): NOT_RELEVANT")

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

    reqs_str = "\n".join(reqs_formatted) if reqs_formatted else "None"

    user_prompt = build_question_selection_prompt(context_str, reqs_str)

    try:
        strat_id = state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None)
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="question_selection",
            conversation_id=state.thread_id,
            strategy_request_id=strat_id,
        )

        data = json.loads(raw_response)
        selected_id = data.get("selected_requirement_id") or data.get("selected_id")
        create_dyn = data.get("create_dynamic")
        reasoning = data.get("reasoning", "No reasoning provided.")
        generated_q = data.get("question")

        # 1. Check existing requirement selection against candidates
        if selected_id and isinstance(selected_id, str) and selected_id != "none":
            candidate_ids = {req.id for req in candidates}
            if selected_id in candidate_ids:
                selected_req = state.get_requirement_by_id(selected_id)
                if selected_req:
                    if generated_q and isinstance(generated_q, str):
                        setattr(selected_req, "_generated_question", generated_q.strip().strip('"').strip("'"))
                    logger.info(f"Selected existing requirement '{selected_id}'. Reasoning: {reasoning}")
                    return selected_req

        # 2. Check dynamic requirement creation if proposed
        if isinstance(create_dyn, dict):
            dyn_id = create_dyn.get("id")
            dyn_title = create_dyn.get("title")
            dyn_desc = create_dyn.get("description")
            if dyn_id and dyn_title and dyn_desc:
                if not is_duplicate_requirement(dyn_id, dyn_title, state.requirements):
                    dyn_req = create_dynamic_requirement(
                        raw_id=dyn_id,
                        title=dyn_title,
                        description=dyn_desc,
                        is_must_have=True
                    )
                    state.requirements.append(dyn_req)
                    if generated_q and isinstance(generated_q, str):
                        setattr(dyn_req, "_generated_question", generated_q.strip().strip('"').strip("'"))
                    logger.info(f"Created & selected dynamic requirement '{dyn_req.id}'. Reasoning: {reasoning}")
                    return dyn_req
                else:
                    logger.info(f"Proposed dynamic requirement [{dyn_id}] in selection was duplicate. Skipping.")

        if not selected_id or selected_id == "none":
            logger.info(f"LLM decided no further requirements needed. Reasoning: {reasoning}")
            return None

        logger.warning(f"LLM output selected_requirement_id='{selected_id}' was invalid or not in candidates. Falling back to default candidate.")

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"select_next_requirement error: {e}. Falling back to default priority.")

    if candidates:
        fallback_req = _get_default_fallback(candidates)
        logger.info(f"Fallback selected requirement '{fallback_req.id}'.")
        return fallback_req

    return None





def generate_question(state: MarketingAgentState, selected_requirement: InformationRequirement) -> str:
    """
    Generates a natural, context-aware conversational question for the selected information requirement.

    Uses LLM to frame the question personalized to the business context and conversation history.
    Falls back gracefully to a simple template-based question if the LLM call fails.
    """
    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}",
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}",
    ]
    doc_str = format_document_context(state.document_context)
    if ctx.past_marketing_document and not doc_str:
        context_lines.append(f"Past marketing document summary: {ctx.past_marketing_document}")
    if doc_str:
        context_lines.append(f"\nHISTORICAL MARKETING EVIDENCE (FROM UPLOADED DOCUMENT):\n{doc_str}")

    op_str = format_online_presence_context(state.online_presence_context)
    if op_str:
        context_lines.append(f"\nONLINE PRESENCE CONTEXT (scraped/verified website & social media sources):\n{op_str}")

    # Include resolved requirements status and values (both KNOWN and UNAVAILABLE)
    resolved_reqs = [
        req for req in state.requirements 
        if req.status != RequirementStatus.UNKNOWN
    ]
    if resolved_reqs:
        context_lines.append("\nRESOLVED REQUIREMENTS STATUS & VALUES:")
        for req in resolved_reqs:
            if req.status == RequirementStatus.KNOWN and req.value:
                context_lines.append(f"- {req.title} ({req.id}): KNOWN -> {req.value}")
            elif req.status == RequirementStatus.UNAVAILABLE:
                context_lines.append(f"- {req.title} ({req.id}): UNAVAILABLE (No direct competitors / info unavailable / not applicable)")

    # Include previous conversation Q&A history
    if state.qa_history:
        context_lines.append("\nPREVIOUS CONVERSATION Q&A HISTORY (do NOT repeat these questions or exact phrasing):")
        for turn in state.qa_history:
            context_lines.append(f"Q: \"{turn.question}\"")
            context_lines.append(f"A: \"{turn.answer}\"")

    context_str = "\n".join(context_lines)

    user_prompt = build_question_generation_prompt(
        context_str,
        selected_requirement.id,
        selected_requirement.title,
        selected_requirement.description,
    )

    try:
        strat_id = state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None)
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            temperature=0.7,
            operation="question_generation",
            conversation_id=state.thread_id,
            strategy_request_id=strat_id,
        )

        question = raw_response.strip().strip('"').strip("'").strip()
        if "\n" in question:
            lines = [line.strip().strip('"').strip("'").strip() for line in question.split("\n") if line.strip()]
            if lines:
                question = lines[0]

        if question:
            logger.info(f"Generated question for requirement '{selected_requirement.id}': {question}")
            return question

    except (LLMServiceError, Exception) as e:
        logger.error(f"generate_question error: {e}. Falling back to template question.")

    fallback_raw = FALLBACK_QUESTIONS.get(
        selected_requirement.id,
        f"What details can you share about your {selected_requirement.title.lower()}?"
    )
    if ctx.product_or_service or ctx.company_name:
        name = ctx.product_or_service or ctx.company_name
        fallback_question = fallback_raw.replace("your business", name).replace("your offering", name).replace("your products or services", f"your {name}")
    else:
        fallback_question = fallback_raw

    logger.info(f"Fallback question used: {fallback_question}")
    return fallback_question


def prepare_next_question(state: MarketingAgentState) -> Optional[str]:
    """
    Selects or creates the next requirement and generates the question text,
    updating state.current_question and state.active_requirement_id accordingly.

    Returns the generated question string, or None if no missing requirements remain.
    """
    selected_req = select_next_requirement(state)
    if not selected_req:
        state.current_question = None
        state.active_requirement_id = None
        return None

    pre_gen_q = getattr(selected_req, "_generated_question", None)
    if pre_gen_q and isinstance(pre_gen_q, str) and 0 < len(pre_gen_q.strip()) <= 300:
        question_text = pre_gen_q.strip()
    else:
        question_text = generate_question(state, selected_req)

    state.current_question = question_text
    state.active_requirement_id = selected_req.id
    return question_text


def _build_first_message_fallback(state: MarketingAgentState, selected_req: InformationRequirement) -> str:
    """
    Builds a grounded, safe fallback for the first contextual message if LLM generation fails or returns invalid JSON.
    Dynamically generates the question for the selected requirement.
    """
    ctx = state.business_context
    op_ctx = state.online_presence_context
    has_website = bool(op_ctx and op_ctx.website and op_ctx.website.summary)
    summary = build_grounded_business_summary(
        company=ctx.company_name,
        product=ctx.product_or_service,
        goal=ctx.marketing_goal,
        has_website=has_website,
    )
    question = generate_question(state, selected_req)
    greeting = "welcome! Let's get started on your marketing plan. Here is my initial summary of your profile:"
    return f"{greeting}\n\n{summary}\n\n{question}"


def generate_first_contextual_message(state: MarketingAgentState) -> str:
    """
    Generates the initial contextual agent message combining grounded business understanding
    and a connected question targeting the single highest-value missing requirement.

    Sets state.current_question, state.active_requirement_id, and state.first_message_generated = True.
    Returns the combined message text string.
    """
    already_asked_or_resolved_ids = {
        req.id for req in state.requirements
        if req.status != RequirementStatus.UNKNOWN
    }
    candidates = [
        req for req in state.requirements
        if req.status == RequirementStatus.UNKNOWN and req.id not in already_asked_or_resolved_ids
    ]

    if not candidates:
        logger.info("generate_first_contextual_message: No candidates missing. Marking first_message_generated=True.")
        state.first_message_generated = True
        return prepare_next_question(state) or ""

    ctx = state.business_context
    context_lines = [
        f"Company Name: {ctx.company_name or 'Not provided'}",
        f"Product/Service: {ctx.product_or_service or 'Not provided'}",
        f"Marketing Goal: {ctx.marketing_goal or 'Not provided'}",
        f"Target Audience: {ctx.target_audience or 'Not provided'}",
        f"Marketing Budget/Resources: {ctx.budget_resources or 'Not provided'}",
        f"Current Marketing Channels: {ctx.current_marketing_channels or 'Not provided'}",
    ]
    doc_str = format_document_context(state.document_context)
    if ctx.past_marketing_document and not doc_str:
        context_lines.append(f"Past marketing document summary: {ctx.past_marketing_document}")
    if doc_str:
        context_lines.append(f"\nHISTORICAL MARKETING EVIDENCE (FROM UPLOADED DOCUMENT):\n{doc_str}")

    op_str = format_online_presence_context(state.online_presence_context)
    if op_str:
        context_lines.append(f"\nONLINE PRESENCE CONTEXT (scraped/verified website & social media sources):\n{op_str}")

    resolved_reqs = [
        req for req in state.requirements
        if req.status != RequirementStatus.UNKNOWN
    ]
    if resolved_reqs:
        context_lines.append("\nALREADY RESOLVED REQUIREMENTS:")
        for req in resolved_reqs:
            if req.status == RequirementStatus.KNOWN and req.value:
                context_lines.append(f"- {req.title} ({req.id}): KNOWN -> {req.value}")

    context_str = "\n".join(context_lines)

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

    user_prompt = build_first_message_prompt(context_str, reqs_str)

    try:
        strat_id = state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None)
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=FIRST_MESSAGE_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="first_message_generation",
            conversation_id=state.thread_id,
            strategy_request_id=strat_id,
        )

        data = json.loads(raw_response)
        context_summary = (data.get("context_summary") or "").strip()
        selected_id = data.get("selected_requirement_id") or data.get("selected_id")
        question = (data.get("question") or "").strip().strip('"').strip("'").strip()
        reasoning = data.get("reasoning", "No reasoning provided.")

        candidate_ids = {req.id for req in candidates}
        if selected_id and selected_id in candidate_ids and context_summary and question:
            selected_req = state.get_requirement_by_id(selected_id)
            if selected_req:
                greeting = "Hi! Let's get started on your marketing plan. Here is my initial summary of your profile:"
                full_message = f"{greeting}\n\n{context_summary}\n\n{question}"

                state.current_question = full_message
                state.active_requirement_id = selected_req.id
                state.first_message_generated = True
                logger.info(f"Generated first contextual agent message for requirement '{selected_req.id}'. Reasoning: {reasoning}")
                return full_message
        else:
            logger.warning(f"LLM output for first contextual message missing fields or invalid selected_id '{selected_id}'. Using fallback.")

    except Exception as e:
        logger.error(f"Error in generate_first_contextual_message: {e}. Using fallback.")

    fallback_req = _get_default_fallback(candidates)
    full_message = _build_first_message_fallback(state, fallback_req)
    state.current_question = full_message
    state.active_requirement_id = fallback_req.id
    state.first_message_generated = True
    return full_message
