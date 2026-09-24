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
    "You are an expert marketing strategy assistant analyzing a client's answer to a specific information requirement question.\n\n"
    "Your task is to classify the client's answer into one of two statuses:\n"
    "1. 'known': The client provided concrete information or a meaningful answer. Extract a clean, concise summary of their answer as 'value'.\n"
    "2. 'unavailable': The client indicated they do not have this information, haven't decided yet, or it is not available (e.g., 'we don't have a budget yet', 'not sure', 'no fixed plan', 'don't know'). In this case, 'value' MUST be null.\n\n"
    "Output MUST be a JSON object in this exact format:\n"
    "{\n"
    '  "status": "known" | "unavailable",\n'
    '  "value": "<concise extracted value string or null>"\n'
    "}"
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
    Classifies the answer as 'known' or 'unavailable', updates requirement status/value,
    records a QATurn in state.qa_history, and clears active question tracking.
    """
    active_req_id = state.active_requirement_id
    question_text = state.current_question

    if not active_req_id or not user_answer:
        logger.warning("process_answer_for_requirement called without active_requirement_id or user_answer.")
        return

    req = state.get_requirement_by_id(active_req_id)
    req_title = req.title if req else active_req_id
    req_desc = req.description if req else ""

    user_prompt = (
        f"REQUIREMENT TO CLARIFY:\n"
        f"- ID: {active_req_id}\n"
        f"- Title: {req_title}\n"
        f"- Description: {req_desc}\n\n"
        f"QUESTION ASKED:\n\"{question_text or 'Not provided'}\"\n\n"
        f"USER ANSWER:\n\"{user_answer.strip()}\"\n\n"
        "Classify the user answer and output the JSON response."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=ANSWER_CLASSIFICATION_SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        status_str = str(data.get("status", "known")).lower()
        extracted_val = data.get("value")

        if status_str == "unavailable":
            final_status = RequirementStatus.UNAVAILABLE
            final_value = None
        else:
            final_status = RequirementStatus.KNOWN
            final_value = extracted_val or user_answer.strip()

        # 1. Update target requirement status and value
        if req:
            req.status = final_status
            req.value = final_value
            logger.info(f"Requirement [{active_req_id}] updated to {final_status.value} with value: {final_value}")

        # 2. Record QATurn in qa_history
        qa_turn = QATurn(
            question=question_text or "",
            answer=user_answer.strip(),
            requirement_id=active_req_id
        )
        state.qa_history.append(qa_turn)

        # 3. Clear active question fields
        state.current_question = None
        state.active_requirement_id = None

    except (LLMServiceError, json.JSONDecodeError, Exception) as e:
        logger.error(f"process_answer_for_requirement error: {e}. Applying fallback status=KNOWN.")

        # Defensive fallback: mark requirement as KNOWN with raw answer
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
