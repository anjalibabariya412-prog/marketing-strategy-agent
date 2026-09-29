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
    "2. However, do NOT simply pick the first must-have item blindly. Reason about which specific requirement provides the highest immediate value or clarity for THIS specific business and its current context.\n"
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
    if not candidates:
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
    "You are a friendly, experienced human marketing consultant having a warm, direct conversation with a business owner.\n"
    "Your goal is to ask ONE short, simple, direct question to collect a specific piece of missing information.\n\n"
    "CRITICAL RULES FOR QUESTION PHRASING:\n"
    "1. CONCEPT TRANSLATION: The requirement title/ID is an internal system concept. Do NOT blindly convert or mechanically copy the requirement title verbatim into a question. Ask about the underlying practical information needed.\n"
    "2. BUSINESS CONTEXT & BRAND INTEGRATION: Weave the specific company name, product, or service name naturally into the question whenever it improves clarity and sounds conversational (e.g. asking 'Which channels are you currently using to promote [Company/Product]?' or 'How is [Product] currently priced?' instead of generic placeholders like 'your business' or 'your offering'). Do NOT force the name if it creates awkward repetition (such as 'your [Company] business'). Never invent unprovided facts.\n"
    "3. GROUNDED IN CONTEXT & CONTEXT-APPROPRIATE FRAMING: Stay strictly grounded in the conversation context. Do NOT introduce ungrounded assumptions, specific customer journey stages, or assume customers have problems, frustrations, or dissatisfaction.\n"
    "   - For 'customer_pain_points': Interpret this broadly as understanding customer needs, preferences, motivations, decision factors, or problems, depending on the actual business and offering. Choose the framing based on the business context. Do NOT assume customers have problems, frustrations, dissatisfaction, or challenges. Use 'challenges/problems' ONLY when the business context supports it (e.g., software or services solving explicit pain points). For businesses where customers are making a preference or purchase choice (e.g., a café, home decor, food/lifestyle), ask what target customers typically look for, prefer, or consider when choosing that type of offering.\n"
    "4. PREVIOUS QUESTIONS & AVOID REPETITION: Review the previous conversation Q&A history. Do NOT repeat questions, exact template phrases, sentence structures, or recurring openings that have already been used. Vary the natural phrasing based on the context and the specific information being collected.\n"
    "5. COMPETITOR AWARENESS: Never reference competitors if the user indicated there are no direct competitors or if 'competitors' status is UNAVAILABLE. For USP/Differentiation when no competitors exist, ask naturally about key software/offering features, unique capabilities, or main strengths.\n"
    "6. CLARITY & ACCURACY OVER FORCED VARIATION: Prioritize clarity, naturalness, and business relevance. Ask only for the information needed by the selected requirement without inventing extra details.\n"
    "7. CONCISE & DIRECT: Keep the question concise, normally under 20 words. No formal preambles (e.g., 'Could you tell me...', 'Can you elaborate...'), no bulleted lists, and no raw system jargon.\n"
    "8. NO ASSUMPTION OF PROBLEMS: Do not assume the customer has a problem, frustration or challenge unless the business context suggests one. For businesses where people are simply choosing among pleasant options (for example a cafe, a bakery, a gift shop), ask about what they are looking for, what occasion brings them, or what makes them choose this place, instead of asking what problem or challenge they have.\n"
    "9. OUTPUT FORMAT: Return ONLY the raw question text. No quotes, no markdown, no preamble.\n"
    "10. NATURAL VARIATION: Do not rely on a single question template for any requirement. The same requirement may be asked in different ways depending on the business context. Vary the question structure naturally while preserving the exact information being requested. Avoid repeatedly starting questions with phrases such as 'What is the biggest...', 'What are the main...', 'What challenges...', or 'What frustrations...'.\n"
)

FALLBACK_QUESTIONS = {
    "customer_pain_points": "What do your target customers typically look for or consider when choosing your offering?",
    "competitors": "Which other brands do your customers usually compare you with?",
    "usp_differentiation": "What are the main strengths or key features of your offering?",
    "current_marketing_channels": "Where are you currently promoting your business?",
    "pricing_model": "How do you structure your pricing or fees?",
    "brand_tone": "How would you describe the tone of your brand?",
    "sales_process": "How do customers usually buy from you?",
    "purchase_frequency": "How often do your customers typically buy from you?",
    "geography": "Where are your main customers located?",
    "sales_cycle_length": "How long does it usually take a customer to decide to buy?",
    "previous_marketing_results": "What marketing tactics have worked best for you so far?",
    "seasonality": "Are there specific times of year when your sales peak?",
    "customer_acquisition_method": "How do new customers currently find your business?",
}


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

    user_prompt = (
        f"BUSINESS CONTEXT & CONVERSATION HISTORY:\n{context_str}\n\n"
        f"INFORMATION TO COLLECT:\n"
        f"- Requirement: {selected_requirement.title} ({selected_requirement.id})\n"
        f"- Description: {selected_requirement.description}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Use natural, conversational wording tailored to this specific business and product ({ctx.product_or_service or ctx.company_name or 'the business'}).\n"
        f"- Integrate the known business name ({ctx.company_name}) or product/service name ({ctx.product_or_service}) naturally into the question where it improves clarity, avoiding generic phrases like 'your business' or 'your offering' when the specific name fits far better.\n"
        f"- Do NOT force the business name into every phrase if it sounds unnatural or redundant.\n"
        f"- Stay grounded: do NOT invent ungrounded assumptions or journey stages (like 'before hiring you' or 'during checkout').\n"
        f"- For customer_pain_points: Choose the appropriate framing based on the business type. Do NOT assume customers have problems or challenges. Use 'challenges/problems' only when the business context supports it. For preference or purchase choices, ask what customers typically look for, prefer, or consider when choosing this type of offering.\n"
        f"- For pricing: ask using business-appropriate terms naturally referencing the product or service.\n"
        f"- For USP/differentiation: if competitors are marked unavailable or no competitors exist, ask about key features/strengths WITHOUT mentioning competitors.\n"
        f"- Do NOT copy template questions verbatim; generate a fresh, contextually natural question.\n\n"
        f"Write ONE short, natural, 1-sentence direct question to ask the owner of {ctx.company_name or 'this business'} to gather this information."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            temperature=0.7
        )

        question = raw_response.strip().strip('"').strip("'").strip()
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
        fallback_question = fallback_raw.replace("your business", name).replace("your offering", name)
    else:
        fallback_question = fallback_raw

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
