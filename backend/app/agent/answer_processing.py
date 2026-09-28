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
    "CRITICAL DISAMBIGUATION RULE (pricing_model vs budget_resources):\n"
    "- Use the semantic meaning and conversation context of the user's answer, NOT just the presence of a monetary value.\n"
    "- pricing_model = the price and payment structure of the product or service being sold to customers. Examples: product price, service fee, monthly subscription, yearly subscription, one-time payment, packages, etc.\n"
    "- budget_resources = the money/resources available to the business for marketing activities such as advertising, content, influencers, campaigns, etc.\n"
    "  * Example: 'Our cake costs ₹800' or '$50 per session' -> pricing_model\n"
    "  * Example: 'We can spend ₹30,000 on marketing' or 'rs 30000 budget' -> budget_resources\n"
    "- If the currently active requirement is pricing_model but the user's answer clearly refers to marketing budget (e.g. 'rs 30000 for ads/marketing'), do NOT incorrectly classify it as product pricing. Set active_requirement status to 'unavailable' (value: null) and list budget_resources with its extracted value in incidentally_known_requirements.\n\n"
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
