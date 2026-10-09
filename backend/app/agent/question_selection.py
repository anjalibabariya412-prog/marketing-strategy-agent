import json
import logging
from typing import Optional, List

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

SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant selecting the next single information requirement to clarify with a client.\n"
    "Your goal is to evaluate candidate missing requirements against the known business context, past marketing document, online presence context (scraped website/social media), and conversation history.\n\n"
    "REQUIREMENT SELECTION RULES:\n"
    "1. Prefer selecting an existing requirement ID from the provided CANDIDATE MISSING REQUIREMENTS list.\n"
    "2. ONLY if an important strategic information gap exists for this specific business that is NOT covered by any existing requirement, you may propose ONE dynamic requirement using 'create_dynamic'.\n"
    "3. Priority requirements (target_market_location, customer_needs_buying_behavior, competitive_landscape, usp_differentiation, pricing_offer_structure, sales_conversion_journey) should be evaluated systematically.\n"
    "4. GROUNDEDNESS & RELEVANCE RULES:\n"
    "   - target_market_location: Do NOT infer the target market location from a business address, phone number, website domain, or Instagram bio location. Select 'target_market_location' unless the intended marketing target region/geography is explicitly stated in context.\n"
    "   - customer_needs_buying_behavior: Focus on what customers look for, value, prefer, or consider. Do NOT assume every business has customer 'pain points' (e.g. for bakeries/retail/salons, ask what customers look for or value; for SaaS/B2B, problems/challenges can be relevant).\n"
    "   - competitive_landscape: Understand direct/indirect competitors and alternative solutions. If competitors are already mentioned in context, do NOT ask for competitor names again; ask how customers compare or what alternatives they consider. If the user explicitly states there are no direct competitors, do not repeatedly ask for direct competitors.\n"
    "   - usp_differentiation: Adapt wording to the specific business/offering type (shoes: footwear collections; bakery: cakes/desserts; SaaS: software features; salon: services; skincare: products).\n"
    "   - pricing_offer_structure: Identify offering type first (physical products/shoes, skincare, bakery, SaaS, courses, services). Phrase the question naturally for that type (shoes/physical products: price ranges across collections or items; skincare: product price ranges; bakery: typical product price range; SaaS: pricing plans or subscription pricing; courses: course fees; services: service pricing or packages). Do NOT force 'package', 'plan', or 'structure' when unnatural.\n"
    "   - sales_conversion_journey: Adapt wording to business type (shoes/retail: online store checkout or store visit; bakery: ordering process; salon: booking; SaaS: enquiry to purchase). Do NOT use generic sales jargon like 'sales cycle' when unnatural.\n"
    "5. DYNAMIC REQUIREMENT CREATION RULES:\n"
    "   - Do NOT create a dynamic requirement if an existing requirement already covers the topic, if nice-to-have, if already known, or if rephrasing an existing requirement.\n"
    "   - Create a dynamic requirement ONLY when: strategically important + genuinely missing + specific to this business + not covered by any existing requirement.\n"
    "6. QUESTION GENERATION:\n"
    "   - Generate ONE short, simple, direct, natural, business-specific question using the actual product/service type (5-15 words preferred, max 20 words).\n"
    "   - NO formal preambles ('Could you please tell me...', 'Can you elaborate...'). Start directly with 'What...', 'Who...', 'Which...', 'How...'.\n\n"
    "OUTPUT FORMAT:\n"
    "You MUST respond ONLY with a JSON object in this exact schema:\n"
    "{\n"
    '  "selected_requirement_id": "<existing_requirement_id from candidates list or null>",\n'
    '  "create_dynamic": null OR {\n'
    '    "id": "<short_unique_id>",\n'
    '    "title": "<Clear Title>",\n'
    '    "description": "<Concise description of strategic gap>"\n'
    '  },\n'
    '  "question": "<short direct conversational question (5-15 words)>",\n'
    '  "reasoning": "<short explanation>"\n'
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
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}",
    ]

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

    user_prompt = (
        f"BUSINESS CONTEXT:\n{context_str}\n\n"
        f"CANDIDATE MISSING REQUIREMENTS:\n{reqs_str}\n\n"
        "Select an existing requirement from the candidates list above and generate the question, or propose a dynamic requirement if an uncovered strategic gap exists."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            response_format={"type": "json_object"}
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


QUESTION_GEN_SYSTEM_PROMPT = (
    "You are a practical marketing strategy consultant asking a business owner quick discovery questions.\n"
    "Your goal is to ask ONE short, simple, direct, business-specific, and meaningful question to collect the exact information needed by the selected requirement.\n\n"
    "CRITICAL RULES FOR QUESTION PHRASING:\n"
    "1. OFFERING TYPE IDENTIFICATION & DYNAMIC CONTEXT PHRASING:\n"
    "   - First, identify the specific offering/business type from context (e.g., physical products/shoes, skincare, bakery, SaaS, courses, consultancy/services).\n"
    "   - Frame the question using the actual product/service type rather than mechanically converting the requirement title into generic 'product or service' questions.\n"
    "2. SINGLE QUESTION ONLY: Ask exactly ONE question. Do not ask multiple things in one question. Do not add preambles or explanations before or after.\n"
    "3. SHORT AND CONCISE (5-15 WORDS PREFERRED, MAX 20 WORDS): Keep the question short, simple, and direct.\n"
    "4. SIMPLE EVERYDAY ENGLISH: Use simple everyday English. Avoid complicated vocabulary and unnecessary marketing jargon (e.g. 'leveraging', 'positioned', 'sales cycle', 'value prop').\n"
    "5. NO FORMAL PREAMBLES: NEVER start questions with formal phrases such as 'Could you please tell me...', 'Can you elaborate on...', 'I would like to understand...'. Start directly with 'What...', 'Who...', 'Which...', 'How...'.\n"
    "6. REQUIREMENT-SPECIFIC ADAPTATION:\n"
    "   - target_market_location -> Adapt to business/product type (e.g. 'Which cities or regions do you want to target with your [products/services]?').\n"
    "   - customer_needs_buying_behavior -> Adapt to business type (Bakery: 'What do customers look for when choosing a cake for a celebration?'; Shoes: 'What matters most to customers when choosing your shoes?'; SaaS: 'What motivates businesses to look for a solution like yours?'; Salon: 'What do clients consider most when choosing your salon?').\n"
    "   - competitive_landscape -> Adapt to business type (e.g. 'Who are your main competitors or alternatives customers consider?' or if competitors are known, 'What do customers choose your [product/service] for instead of those options?').\n"
    "   - usp_differentiation -> Adapt to business type (Shoes: 'What makes your shoe collections different from other footwear brands?'; Bakery: 'What makes your cakes and desserts different from other bakeries?'; SaaS: 'What makes your software different from competing solutions?').\n"
    "   - pricing_offer_structure -> Identify the offering type first and phrase the question naturally for that type. Do NOT force 'package', 'plan', or 'structure' when unnatural:\n"
    "     * Physical products / Shoes: Ask about price range across products, collections, or items (e.g. 'What are the price ranges across your different shoe collections?').\n"
    "     * Skincare / Cosmetics: Ask about product price ranges (e.g. 'What is the price range across your skincare products?').\n"
    "     * Bakery / Food: Ask about typical product price range (e.g. 'What is the typical price range for your cakes and baked goods?').\n"
    "     * SaaS / Software: Ask about pricing plans or subscription pricing (e.g. 'What are your monthly subscription plans or pricing tiers?').\n"
    "     * Courses / Education: Ask about course fees or pricing (e.g. 'What are the fees for your courses?').\n"
    "     * Services / Consulting: Ask about service pricing or packages (e.g. 'What is the price range for your consulting services?').\n"
    "   - sales_conversion_journey -> Adapt to business type (Ecommerce/Shoes: 'How do customers usually buy your shoes online or in-store?'; Bakery: 'How do customers usually place an order with you?'; SaaS: 'How does a potential customer move from enquiry to purchase?'; Salon: 'How do clients usually book your services?'). Do NOT use generic sales jargon like 'sales cycle' when unnatural.\n"
    "7. CONTEXT PERSONALIZATION: Use the specific business name or product type naturally.\n"
    "8. AVOID REPETITION: Review previous Q&A history and do NOT repeat previously asked questions.\n"
    "9. OUTPUT FORMAT:\n"
    "Return ONLY raw question text. No quotation marks, no markdown, no explanation."
)

FALLBACK_QUESTIONS = {
    "target_market_location": "Which location(s) do you want to target?",
    "customer_needs_buying_behavior": "What do your customers look for when choosing your offering?",
    "competitive_landscape": "Who are your main competitors or alternative options customers consider?",
    "usp_differentiation": "What makes your offering different from other options?",
    "pricing_offer_structure": "What are the price ranges across your products or services?",
    "sales_conversion_journey": "How do customers usually place an order or buy from you?",
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
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}",
    ]

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

    user_prompt = (
        f"BUSINESS CONTEXT & CONVERSATION HISTORY:\n{context_str}\n\n"
        f"REQUIREMENT TO CLARIFY:\n"
        f"- ID: {selected_requirement.id}\n"
        f"- Title: {selected_requirement.title}\n"
        f"- Description: {selected_requirement.description}\n\n"
        f"INSTRUCTIONS:\n"
        f"- First identify the type of offering (shoes, skincare, bakery, SaaS, courses, services, etc.) from context.\n"
        f"- Write ONE short, simple, direct, context-specific question (5–15 words preferred, MAX 20 WORDS).\n"
        f"- Use the actual product/service type naturally instead of generic 'product or service'. For pricing_offer_structure, ask naturally for the offering type (e.g. shoe collection price ranges, skincare product prices, SaaS subscription tiers, course fees) without forcing 'package' or 'plan'.\n"
        f"- Ask ONLY for the specific information needed by '{selected_requirement.title}'.\n"
        f"- Use simple everyday English. NO formal preambles ('Could you please...', 'Can you elaborate...').\n"
        f"- Return ONLY raw question text. No quotes, no markdown, no explanation."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            temperature=0.7
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


FIRST_MESSAGE_SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant starting a discovery conversation with a client.\n"
    "Your task is to generate the VERY FIRST agent message. The message MUST contain TWO connected parts inside ONE single agent response:\n\n"
    "PART 1: BUSINESS UNDERSTANDING (2–4 sentences max)\n"
    "- Summarize what you actually know about the client's business based ONLY on the provided context.\n"
    "- Adapt strictly to available sources:\n"
    "  * If website analysis is available: You may reference findings from their website/products.\n"
    "  * If NO website is available: DO NOT mention 'website', 'your website', or 'found on your website'. Rely on initial business details, social media, or uploaded documents instead.\n"
    "  * If social media is available (and no website): Reference social findings if explicitly present in extracted data.\n"
    "  * If a past marketing document/PDF is provided: Use information from it.\n"
    "- GROUNDING MANDATE: DO NOT invent facts, competitors, pricing, locations, USP, customer demographics, pain points, product claims, or numbers not explicitly present in context.\n\n"
    "PART 2: CONNECTED TRANSITION & ONE QUESTION\n"
    "- Select ONE missing requirement from the provided CANDIDATE MISSING REQUIREMENTS list that is most strategically valuable based on what is already known vs missing.\n"
    "- Use a natural transition phrase (e.g., 'With that in mind, ...', 'To understand your positioning better, ...', 'Based on this, ...', 'To help shape the strategy around your business, ...', 'Given what I\\'ve seen so far, ...').\n"
    "- Ask exactly ONE clear, direct, business-specific question targeting that selected requirement.\n"
    "- Do NOT ask multiple requirements in one question.\n\n"
    "OUTPUT FORMAT:\n"
    "You MUST respond ONLY with a JSON object in this exact schema:\n"
    "{\n"
    '  "context_summary": "<2-4 sentences summarizing business context strictly grounded in facts>",\n'
    '  "transition": "<Natural transition phrase e.g. With that in mind,>",\n'
    '  "selected_requirement_id": "<ID of selected requirement from candidates list>",\n'
    '  "question": "<ONE clear question targeting the selected requirement>",\n'
    '  "reasoning": "<Short explanation for selecting this requirement>"\n'
    "}\n"
    "Do NOT include markdown formatting or commentary outside the JSON object."
)


def _build_first_message_fallback(state: MarketingAgentState, selected_req: InformationRequirement) -> str:
    """
    Builds a grounded, safe fallback for the first contextual message if LLM generation fails or returns invalid JSON.
    """
    ctx = state.business_context
    company = ctx.company_name
    product = ctx.product_or_service
    goal = ctx.marketing_goal

    op_ctx = state.online_presence_context
    has_website = bool(op_ctx and op_ctx.website and op_ctx.website.summary)

    if has_website and company and product:
        summary = f"I can see that {company} offers {product} based on your online presence."
    elif company and product and goal:
        summary = f"I understand that {company} offers {product} and is focused on {goal}."
    elif company and product:
        summary = f"I understand that {company} provides {product}."
    elif product and goal:
        summary = f"I understand that your business offers {product} with a primary goal of {goal}."
    elif product:
        summary = f"I understand that your business provides {product}."
    elif company:
        summary = f"I understand that {company} is aiming to develop a personalized marketing strategy."
    else:
        summary = "I have reviewed the initial business details you provided."

    transitions_and_questions = {
        "usp_differentiation": (
            "To help shape the strategy around your business,",
            "what would you say is the most important thing that differentiates your offering from alternatives?"
        ),
        "customer_needs_buying_behavior": (
            "To understand your target audience better,",
            "what do your customers value most when choosing your product or service?"
        ),
        "competitive_landscape": (
            "To position your brand effectively,",
            "who are the main competitors or alternatives your potential customers consider?"
        ),
        "target_market_location": (
            "Given what I've seen so far,",
            "which geographic locations or regions do you want to target?"
        ),
        "pricing_offer_structure": (
            "With that in mind,",
            "what are the typical price ranges across your products or services?"
        ),
        "sales_conversion_journey": (
            "To understand your customer flow,",
            "how do potential customers usually place an order or buy from you?"
        ),
    }

    t_and_q = transitions_and_questions.get(
        selected_req.id,
        ("To help shape the strategy around your business,", f"what key details can you share regarding your {selected_req.title.lower()}?")
    )
    transition, question = t_and_q

    greeting = "Hi! Let's get started on your marketing plan. Here is my initial summary of your profile:"
    return f"{greeting}\n\n{summary}\n\n{transition} {question}"


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
        f"Past marketing document summary: {ctx.past_marketing_document or 'Not provided'}",
    ]

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

    user_prompt = (
        f"BUSINESS CONTEXT & ONLINE PRESENCE:\n{context_str}\n\n"
        f"CANDIDATE MISSING REQUIREMENTS:\n{reqs_str}\n\n"
        "Generate the first contextual agent message (Part 1: business understanding summary, Part 2: connected transition + question targeting ONE requirement)."
    )

    try:
        raw_response = get_llm_response(
            prompt=user_prompt,
            system_prompt=FIRST_MESSAGE_SYSTEM_PROMPT,
            response_format={"type": "json_object"}
        )

        data = json.loads(raw_response)
        context_summary = (data.get("context_summary") or "").strip()
        transition = (data.get("transition") or "").strip()
        selected_id = data.get("selected_requirement_id") or data.get("selected_id")
        question = (data.get("question") or "").strip()
        reasoning = data.get("reasoning", "No reasoning provided.")

        candidate_ids = {req.id for req in candidates}
        if selected_id and selected_id in candidate_ids and context_summary and question:
            selected_req = state.get_requirement_by_id(selected_id)
            if selected_req:
                if transition:
                    transition_clean = transition.rstrip(",")
                    if not transition.endswith(",") and not transition.endswith("."):
                        transition_clean = transition + ","
                    else:
                        transition_clean = transition

                    if question and question[0].isupper() and transition_clean.endswith(","):
                        question_clean = question[0].lower() + question[1:]
                    else:
                        question_clean = question

                    combined_question = f"{transition_clean} {question_clean}"
                else:
                    combined_question = question

                greeting = "Hi! Let's get started on your marketing plan. Here is my initial summary of your profile:"
                full_message = f"{greeting}\n\n{context_summary}\n\n{combined_question}"

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
