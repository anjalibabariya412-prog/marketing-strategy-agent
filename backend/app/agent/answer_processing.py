import json
import logging
from typing import Optional, List

from backend.app.core.config import settings
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.information_requirement import (
    RequirementStatus,
    create_dynamic_requirement,
    is_duplicate_requirement,
)
from backend.app.schemas.agent.document_context import format_document_context
from backend.app.schemas.agent.online_presence import format_online_presence_context
from backend.app.schemas.agent.qa_turn import QATurn
from backend.app.services.llm_service import get_llm_response, LLMServiceError
from backend.app.services.tavily_research_service import classify_competitor_response

logger = logging.getLogger(__name__)

from backend.app.prompts.answer_processing import (
    INITIAL_CONTEXT_SYSTEM_PROMPT,
    ANSWER_CLASSIFICATION_SYSTEM_PROMPT,
    MERGED_TURN_SYSTEM_PROMPT,
    COMPETITOR_EXTRACTION_SYSTEM_PROMPT,
    build_initial_context_prompt,
    build_competitor_extraction_prompt,
    build_answer_classification_prompt,
    build_merged_turn_prompt,
)



def extract_competitors(user_answer: str, conversation_id: Optional[str] = None) -> List[str]:
    """
    Extracts structured competitor names/alternatives from a natural language user response using LLM.
    Preserves exact user wording when a competitor cannot be named specifically (e.g. 'two local tiffin services').
    Avoids duplicate names and returns [] if the user states they have no competitors or if none are present.
    """
    if not user_answer or not user_answer.strip():
        return []

    user_prompt = build_competitor_extraction_prompt(user_answer)

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=COMPETITOR_EXTRACTION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="competitor_extraction",
            conversation_id=conversation_id,
        )

        data = json.loads(raw_response)
        extracted = data.get("competitors", [])
        if not isinstance(extracted, list):
            return []

        clean_list = []
        seen = set()
        for item in extracted:
            if isinstance(item, str) and item.strip():
                clean_item = item.strip()
                lower_item = clean_item.lower()
                if lower_item not in seen:
                    seen.add(lower_item)
                    clean_list.append(clean_item)

        return clean_list

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"extract_competitors failed: {e}. Returning empty list.")
        return []


def update_state_competitors(state: MarketingAgentState, competitor_list: List[str]) -> None:
    """
    Appends new non-duplicate competitor names to state.business_context.competitors.
    """
    if not competitor_list:
        return

    existing_set = {c.lower() for c in state.business_context.competitors}
    for comp in competitor_list:
        if comp and comp.strip():
            clean = comp.strip()
            if clean.lower() not in existing_set:
                existing_set.add(clean.lower())
                state.business_context.competitors.append(clean)


def extract_initial_context(user_message: str) -> BusinessContext:
    """
    Extracts BusinessContext fields from a user's initial free-text introduction message.
    Leaves unmentioned fields as None.
    Returns an empty BusinessContext on failure.
    """
    if not user_message or not user_message.strip():
        return BusinessContext()

    user_prompt = build_initial_context_prompt(user_message)

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=INITIAL_CONTEXT_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="initial_context_extraction",
        )

        data = json.loads(raw_response)
        context = BusinessContext(
            company_name=data.get("company_name"),
            product_or_service=data.get("product_or_service"),
            marketing_goal=data.get("marketing_goal"),
            target_audience=data.get("target_audience")
        )
        comps = extract_competitors(user_message)
        if comps:
            context.competitors = comps
        logger.info(f"Extracted initial context: {context.model_dump()}")
        return context

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"extract_initial_context failed: {e}. Returning empty BusinessContext.")
        return BusinessContext()


def process_answer_for_requirement(state: MarketingAgentState, user_answer: str) -> None:
    """
    Processes the user's answer to the active requirement question (state.active_requirement_id).
    Classifies the active requirement as 'known' or 'unavailable' and updates its status/value.
    Additionally evaluates the user's answer against all other UNKNOWN requirements in state;
    if any are incidentally answered, updates their status to KNOWN and stores their values.
    Records a single QATurn for the active question and clears active question tracking.
    """
    active_req_id = state.active_requirement_id
    question_text = state.current_question

    if not active_req_id or not user_answer or not user_answer.strip():
        logger.warning("process_answer_for_requirement called without active_requirement_id or user_answer.")
        return

    req = state.get_requirement_by_id(active_req_id)
    req_title = req.title if req else active_req_id
    req_desc = req.description if req else ""

    # Collect other UNKNOWN candidate requirements to check for incidental information
    other_candidates = [
        r for r in state.get_missing_requirements()
        if r.id != active_req_id
    ]

    other_reqs_formatted = []
    for candidate in other_candidates:
        other_reqs_formatted.append(
            f"- ID: {candidate.id}\n"
            f"  Title: {candidate.title}\n"
            f"  Description: {candidate.description}"
        )
    other_reqs_str = "\n".join(other_reqs_formatted) if other_reqs_formatted else "None"

    user_prompt = build_answer_classification_prompt(
        active_req_id=active_req_id,
        req_title=req_title,
        req_desc=req_desc,
        question_text=question_text,
        other_reqs_str=other_reqs_str,
        user_answer=user_answer,
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=ANSWER_CLASSIFICATION_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="answer_classification",
            conversation_id=state.thread_id,
        )

        data = json.loads(raw_response)

        # 1. Process ACTIVE requirement evaluation
        active_eval = data.get("active_requirement", {})
        status_str = str(active_eval.get("status", "known")).lower()
        extracted_val = active_eval.get("value")

        if status_str == "unavailable":
            final_status = RequirementStatus.UNAVAILABLE
            final_value = None
        else:
            final_status = RequirementStatus.KNOWN
            final_value = extracted_val or user_answer.strip()

        if req:
            req.status = final_status
            req.value = final_value
            logger.info(f"Requirement [{active_req_id}] updated to {final_status.value} with value: {final_value}")

        if active_req_id in ("competitive_landscape", "competitors") or "competitor" in str(active_req_id).lower():
            comps = extract_competitors(user_answer, conversation_id=state.thread_id)
            if comps:
                update_state_competitors(state, comps)

            # For all competitor response categories, mark requirement as resolved so user is not asked again
            if req:
                req.status = RequirementStatus.KNOWN
                req.value = user_answer.strip()

        # 2. Process INCIDENTALLY KNOWN requirements
        incidentals = data.get("incidentally_known_requirements", [])
        if isinstance(incidentals, list):
            for item in incidentals:
                if isinstance(item, dict):
                    inc_id = item.get("requirement_id")
                    inc_val = item.get("value")
                    if inc_id and inc_val and inc_id != active_req_id:
                        inc_req = state.get_requirement_by_id(inc_id)
                        if inc_req and inc_req.status == RequirementStatus.UNKNOWN:
                            inc_req.status = RequirementStatus.KNOWN
                            inc_req.value = str(inc_val).strip()
                            logger.info(f"Requirement [{inc_id}] incidentally updated to KNOWN with value: {inc_val}")
                            if inc_id in ("competitive_landscape", "competitors") or "competitor" in str(inc_id).lower():
                                inc_comps = extract_competitors(str(inc_val))
                                if inc_comps:
                                    update_state_competitors(state, inc_comps)
                                inc_req.status = RequirementStatus.KNOWN
                                inc_req.value = str(inc_val).strip()

        # 3. Record single QATurn in qa_history for the actual turn
        qa_turn = QATurn(
            question=question_text or "",
            answer=user_answer.strip(),
            requirement_id=active_req_id
        )
        state.qa_history.append(qa_turn)

        # 4. Clear active question tracking
        state.current_question = None
        state.active_requirement_id = None

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(
            f"process_answer_for_requirement error: {e}. "
            f"Keeping requirement [{active_req_id}] unresolved (status=UNKNOWN)."
        )

        if req:
            req.status = RequirementStatus.UNKNOWN
            req.value = None

        qa_turn = QATurn(
            question=question_text or "",
            answer=user_answer.strip(),
            requirement_id=None
        )
        state.qa_history.append(qa_turn)

        state.current_question = None
        state.active_requirement_id = None
        state.pending_question = None
        state.pending_requirement_id = None


def process_answer_and_plan_next(state: MarketingAgentState, user_answer: str) -> None:
    """
    Executes a single merged LLM call to:
    1. Process the user's answer for the active requirement and any incidentally answered requirements.
    2. Plan and phrase the next question from the remaining UNKNOWN requirements.

    Stores the next question and requirement ID in state.pending_question and state.pending_requirement_id.
    Falls back gracefully to process_answer_for_requirement(state, user_answer) if the LLM call fails or JSON is invalid.
    Records exactly ONE QATurn in every case.
    """
    active_req_id = state.active_requirement_id
    question_text = state.current_question

    if not active_req_id or not user_answer or not user_answer.strip():
        logger.info(
            f"Merged turn call skipped (no active_req_id or user_answer). "
            f"Fallback to process_answer_for_requirement. Active req: '{active_req_id}'"
        )
        state.pending_question = None
        state.pending_requirement_id = None
        process_answer_for_requirement(state, user_answer)
        return

    req = state.get_requirement_by_id(active_req_id)
    req_title = req.title if req else active_req_id
    req_desc = req.description if req else ""

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
        r for r in state.requirements 
        if r.status != RequirementStatus.UNKNOWN and r.id != active_req_id
    ]
    if resolved_reqs:
        context_lines.append("\nALREADY KNOWN / RESOLVED REQUIREMENTS:")
        for r in resolved_reqs:
            if r.status == RequirementStatus.KNOWN and r.value:
                context_lines.append(f"- {r.title} ({r.id}): KNOWN -> {r.value}")
            elif r.status == RequirementStatus.UNAVAILABLE:
                context_lines.append(f"- {r.title} ({r.id}): UNAVAILABLE")

    # Include previous conversation Q&A history
    if state.qa_history:
        context_lines.append("\nPREVIOUS CONVERSATION Q&A HISTORY (do NOT repeat these questions or exact phrasing):")
        for turn in state.qa_history:
            context_lines.append(f"Q: \"{turn.question}\"")
            context_lines.append(f"A: \"{turn.answer}\"")

    context_str = "\n".join(context_lines)

    # Collect resolved and already-asked requirement IDs to exclude from next selection
    already_asked_or_resolved_ids = {
        r.id for r in state.requirements 
        if r.status != RequirementStatus.UNKNOWN
    }
    already_asked_or_resolved_ids.add(active_req_id)
    if state.qa_history:
        for turn in state.qa_history:
            if turn.requirement_id:
                already_asked_or_resolved_ids.add(turn.requirement_id)

    # Collect other UNKNOWN candidate requirements
    other_candidates = [
        r for r in state.requirements
        if r.status == RequirementStatus.UNKNOWN and r.id not in already_asked_or_resolved_ids
    ]

    other_reqs_formatted = []
    for candidate in other_candidates:
        must_have_label = "Yes" if candidate.is_must_have else "No"
        other_reqs_formatted.append(
            f"- ID: {candidate.id}\n"
            f"  Title: {candidate.title}\n"
            f"  Must Have: {must_have_label}\n"
            f"  Description: {candidate.description}"
        )
    other_reqs_str = "\n".join(other_reqs_formatted) if other_reqs_formatted else "None"

    user_prompt = build_merged_turn_prompt(
        context_str=context_str,
        active_req_id=active_req_id,
        req_title=req_title,
        req_desc=req_desc,
        question_text=question_text,
        user_answer=user_answer,
        other_reqs_str=other_reqs_str,
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=MERGED_TURN_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            operation="merged_turn_processing",
            conversation_id=state.thread_id,
        )

        data = json.loads(raw_response)
        active_eval = data.get("active_requirement")

        if not isinstance(active_eval, dict) or "status" not in active_eval:
            logger.info(
                f"Merged turn call outcome (c): LLM response missing valid 'active_requirement'. "
                f"Active req: '{active_req_id}'. Fallback to process_answer_for_requirement()."
            )
            state.pending_question = None
            state.pending_requirement_id = None
            process_answer_for_requirement(state, user_answer)
            return

        # Active requirement evaluation succeeds -> Apply answer results
        status_str = str(active_eval.get("status", "known")).lower()
        extracted_val = active_eval.get("value")

        if status_str == "unavailable":
            final_status = RequirementStatus.UNAVAILABLE
            final_value = None
        else:
            final_status = RequirementStatus.KNOWN
            final_value = extracted_val or user_answer.strip()

        if req:
            req.status = final_status
            req.value = final_value

        if active_req_id in ("competitive_landscape", "competitors") or "competitor" in str(active_req_id).lower():
            comps = extract_competitors(user_answer, conversation_id=state.thread_id)
            if comps:
                update_state_competitors(state, comps)

            # For all competitor response categories, mark requirement as resolved so user is not asked again
            if req:
                req.status = RequirementStatus.KNOWN
                req.value = user_answer.strip()

        # Process incidental requirements
        incidental_ids = []
        incidentals = data.get("incidentally_known_requirements", [])
        if isinstance(incidentals, list):
            for item in incidentals:
                if isinstance(item, dict):
                    inc_id = item.get("requirement_id")
                    inc_val = item.get("value")
                    if inc_id and inc_val and inc_id != active_req_id:
                        inc_req = state.get_requirement_by_id(inc_id)
                        if inc_req and inc_req.status == RequirementStatus.UNKNOWN:
                            inc_req.status = RequirementStatus.KNOWN
                            inc_req.value = str(inc_val).strip()
                            incidental_ids.append(inc_id)
                            if inc_id in ("competitive_landscape", "competitors") or "competitor" in str(inc_id).lower():
                                inc_comps = extract_competitors(str(inc_val))
                                if inc_comps:
                                    update_state_competitors(state, inc_comps)
                                inc_req.status = RequirementStatus.KNOWN
                                inc_req.value = str(inc_val).strip()

        # Record exactly ONE QATurn for the actual answered turn
        qa_turn = QATurn(
            question=question_text or "",
            answer=user_answer.strip(),
            requirement_id=active_req_id
        )
        state.qa_history.append(qa_turn)

        # Clear active question tracking
        state.current_question = None
        state.active_requirement_id = None

        # Evaluate the 'next' requirement proposal
        next_obj = data.get("next")
        valid_next = False
        selected_id = None
        next_q = None

        if isinstance(next_obj, dict):
            proposed_action = str(next_obj.get("action", "")).lower()
            proposed_q = next_obj.get("question")
            proposed_q_clean = (
                proposed_q.strip().strip('"').strip("'").strip()
                if isinstance(proposed_q, str)
                else None
            )

            # Check question limit
            if len(state.qa_history) < settings.max_questions and proposed_q_clean and 0 < len(proposed_q_clean) <= 300:
                proposed_id = next_obj.get("selected_id")
                create_dyn = next_obj.get("create_dynamic")

                if proposed_id and isinstance(proposed_id, str) and proposed_id != "none":
                    target_req = state.get_requirement_by_id(proposed_id)
                    if (
                        target_req 
                        and target_req.status == RequirementStatus.UNKNOWN 
                        and proposed_id not in already_asked_or_resolved_ids
                        and proposed_id not in incidental_ids
                    ):
                        valid_next = True
                        selected_id = target_req.id
                        next_q = proposed_q_clean
                elif isinstance(create_dyn, dict):
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
                            valid_next = True
                            selected_id = dyn_req.id
                            next_q = proposed_q_clean
                            logger.info(f"Created & planned dynamic requirement '{dyn_req.id}' in merged turn.")
                        else:
                            logger.info(f"Proposed dynamic requirement [{dyn_id}] in merged turn was duplicate. Skipping.")

        if valid_next:
            state.pending_requirement_id = selected_id
            state.pending_question = next_q
            logger.info(
                f"Merged turn call outcome (a): Successfully processed active req [{active_req_id}] "
                f"(incidentals: {incidental_ids}) and planned pending next req [{selected_id}]."
            )
        else:
            state.pending_requirement_id = None
            state.pending_question = None
            logger.info(
                f"Merged turn call outcome (b): Successfully processed active req [{active_req_id}] "
                f"(incidentals: {incidental_ids}) but next selection was invalid/missing. Pending fields cleared."
            )

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.info(
            f"Merged turn call outcome (c): LLM call or parsing failed ({e}). "
            f"Active req: '{active_req_id}'. Fallback to process_answer_for_requirement()."
        )
        state.pending_question = None
        state.pending_requirement_id = None
        process_answer_for_requirement(state, user_answer)
