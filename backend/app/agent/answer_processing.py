import json
import logging
from typing import Optional

from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.business_context import BusinessContext
from backend.app.models.information_requirement import RequirementStatus
from backend.app.models.qa_turn import QATurn
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

INITIAL_CONTEXT_SYSTEM_PROMPT = (
    "You are an expert marketing strategy assistant extracting structured business context from a client's initial message.\n"
    "Your goal is to extract any mentioned information for the following 4 fields:\n"
    "1. company_name: The name of the company or business.\n"
    "2. product_or_service: The product or service offered.\n"
    "3. marketing_goal: The primary marketing goal, objective, or target outcome.\n"
    "4. target_audience: The target customer profile, audience, or demographic.\n\n"
    "RULES:\n"
    "- If a field is NOT mentioned or implied in the message, set its value to null.\n"
    "- Do NOT invent, assume, or fabricate any information not present in the user's message.\n"
    "- Output MUST be a JSON object with keys: 'company_name', 'product_or_service', 'marketing_goal', 'target_audience'."
)

ANSWER_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an expert marketing strategy assistant analyzing a client's message responding to a specific clarifying question.\n\n"
    "You are provided with:\n"
    "1. The ACTIVE requirement being asked about (ID, title, description, and exact question asked).\n"
    "2. A list of OTHER currently missing candidate requirements (ID, title, description).\n"
    "3. The client's response message.\n\n"
    "CRITICAL PRICING MODEL RULE:\n"
    "- Use the semantic meaning and conversation context of the user's answer, NOT just the presence of a monetary value.\n"
    "- pricing_model = the price and payment structure of the product or service being sold to customers (e.g., product price, service fee, monthly subscription, yearly subscription, one-time payment, packages).\n"
    "- If the active requirement is pricing_model but the user's answer refers to internal marketing spend/budget rather than product pricing, set active_requirement status to 'unavailable' (value: null).\n\n"
    "Your tasks:\n"
    "TASK 1 - EVALUATE ACTIVE REQUIREMENT:\n"
    "- Classify the client's answer for the ACTIVE requirement as 'known' (if they provided concrete information for it) or 'unavailable' (if they indicated they don't know, haven't decided, or did not answer it).\n"
    "- If 'known', extract a concise summary of their answer as 'value'. If 'unavailable', set 'value' to null.\n\n"
    "TASK 2 - EVALUATE OTHER CANDIDATE REQUIREMENTS:\n"
    "- Scan the client's message against the list of OTHER missing candidate requirements.\n"
    "- If the message incidentally reveals concrete information for any of these OTHER requirements (even if not explicitly asked about), extract that requirement's ID and a concise summary as 'value'.\n"
    "- Do NOT include requirements that were not mentioned or implied.\n\n"
    "OUTPUT FORMAT:\n"
    "You MUST respond ONLY with a JSON object in this exact schema:\n"
    "{\n"
    '  "active_requirement": {\n'
    '    "status": "known" | "unavailable",\n'
    '    "value": "<concise extracted value string or null>"\n'
    "  },\n"
    '  "incidentally_known_requirements": [\n'
    "    {\n"
    '      "requirement_id": "<requirement_id>",\n'
    '      "value": "<concise extracted value string>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Do NOT include markdown formatting outside the JSON object."
)


from backend.app.agent.question_selection import (
    SYSTEM_PROMPT as QUESTION_SELECT_SYSTEM_PROMPT,
    QUESTION_GEN_SYSTEM_PROMPT,
)

_ANSWER_RULES = ANSWER_CLASSIFICATION_SYSTEM_PROMPT.split("OUTPUT FORMAT:")[0].strip()

_PRIORITIZATION_SECTION = (
    "REQUIREMENT SELECTION PRIORITIZATION RULES:\n"
    "1. Must-have requirements (Must Have: Yes) should generally be prioritized over optional ones.\n"
    "2. However, do NOT simply pick the first must-have item blindly. Reason about which specific requirement provides the highest immediate value or clarity for THIS specific business and its current context.\n"
    "3. Occasionally, a non-must-have requirement (Must Have: No) may be unusually urgent or foundational for a particular business situation. You may pick a non-must-have item if you clearly reason why it is more critical right now than the remaining must-haves.\n"
    "4. Do NOT select a requirement for 'next' if it is already adequately covered by the business context or past marketing document summary."
)

_QUESTION_PHRASING_RULES = QUESTION_GEN_SYSTEM_PROMPT.split("9. OUTPUT FORMAT:")[0].strip()

MERGED_TURN_SYSTEM_PROMPT = (
    f"{_ANSWER_RULES}\n\n"
    f"{_PRIORITIZATION_SECTION}\n\n"
    f"{_QUESTION_PHRASING_RULES}\n\n"
    "MERGED TURN OUTPUT FORMAT:\n"
    "You MUST respond ONLY with a JSON object conforming strictly to this exact schema:\n"
    "{\n"
    '  "active_requirement": {\n'
    '    "status": "known" | "unavailable",\n'
    '    "value": "<concise extracted value string or null>"\n'
    "  },\n"
    '  "incidentally_known_requirements": [\n'
    "    {\n"
    '      "requirement_id": "<requirement_id>",\n'
    '      "value": "<concise extracted value string>"\n'
    "    }\n"
    "  ],\n"
    '  "next": {\n'
    '    "selected_id": "<requirement_id>",\n'
    '    "question": "<short direct conversational question>",\n'
    '    "reasoning": "<short explanation of why this requirement was selected next>"\n'
    "  }\n"
    "}\n"
    "IMPORTANT FOR 'next': 'selected_id' MUST be chosen ONLY from candidate requirements that remain UNKNOWN after applying the active and incidentally answered requirements. If no candidate requirements remain UNKNOWN, set 'next' to null.\n"
    "Do NOT include markdown formatting or commentary outside the JSON object."
)



def extract_initial_context(user_message: str) -> BusinessContext:
    """
    Extracts BusinessContext fields from a user's initial free-text introduction message.
    Leaves unmentioned fields as None.
    Returns an empty BusinessContext on failure.
    """
    if not user_message or not user_message.strip():
        return BusinessContext()

    user_prompt = f"USER MESSAGE:\n\"{user_message.strip()}\"\n\nExtract business context fields and return the JSON object."

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=INITIAL_CONTEXT_SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        context = BusinessContext(
            company_name=data.get("company_name"),
            product_or_service=data.get("product_or_service"),
            marketing_goal=data.get("marketing_goal"),
            target_audience=data.get("target_audience")
        )
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

    user_prompt = (
        f"ACTIVE REQUIREMENT TO CLARIFY:\n"
        f"- ID: {active_req_id}\n"
        f"- Title: {req_title}\n"
        f"- Description: {req_desc}\n"
        f"- Question Asked: \"{question_text or 'Not provided'}\"\n\n"
        f"OTHER MISSING CANDIDATE REQUIREMENTS:\n{other_reqs_str}\n\n"
        f"CLIENT RESPONSE MESSAGE:\n\"{user_answer.strip()}\"\n\n"
        "Evaluate the response against the active requirement and other candidates, then output the JSON object."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=ANSWER_CLASSIFICATION_SYSTEM_PROMPT,
            response_format={"type": "json_object"}
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
        logger.error(f"process_answer_for_requirement error: {e}. Applying fallback status=KNOWN.")

        if req:
            req.status = RequirementStatus.KNOWN
            req.value = user_answer.strip()

        qa_turn = QATurn(
            question=question_text or "",
            answer=user_answer.strip(),
            requirement_id=active_req_id
        )
        state.qa_history.append(qa_turn)

        state.current_question = None
        state.active_requirement_id = None


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
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}",
    ]


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

    user_prompt = (
        f"BUSINESS CONTEXT & HISTORY:\n{context_str}\n\n"
        f"ACTIVE REQUIREMENT BEING ANSWERED:\n"
        f"- ID: {active_req_id}\n"
        f"- Title: {req_title}\n"
        f"- Description: {req_desc}\n"
        f"- Question Asked: \"{question_text or 'Not provided'}\"\n\n"
        f"CLIENT RESPONSE MESSAGE:\n\"{user_answer.strip()}\"\n\n"
        f"OTHER MISSING CANDIDATE REQUIREMENTS FOR NEXT QUESTION:\n{other_reqs_str}\n\n"
        "Evaluate the user's answer for the active requirement and any incidental requirements, and select/phrase the next question from the remaining UNKNOWN candidates. Output the JSON object."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=MERGED_TURN_SYSTEM_PROMPT,
            response_format={"type": "json_object"}
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
            proposed_id = next_obj.get("selected_id")
            proposed_q = next_obj.get("question")
            if proposed_id and proposed_q and isinstance(proposed_id, str) and isinstance(proposed_q, str):
                proposed_q_clean = proposed_q.strip().strip('"').strip("'").strip()
                target_req = state.get_requirement_by_id(proposed_id)
                # Validation: must be a requirement that is STILL UNKNOWN after updates and NOT in already_asked_or_resolved_ids
                if (
                    target_req 
                    and target_req.status == RequirementStatus.UNKNOWN 
                    and proposed_id not in already_asked_or_resolved_ids
                    and proposed_id not in incidental_ids
                ):
                    if 0 < len(proposed_q_clean) <= 300:
                        valid_next = True
                        selected_id = target_req.id
                        next_q = proposed_q_clean

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
