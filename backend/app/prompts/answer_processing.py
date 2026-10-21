"""
Prompt definitions and prompt builders for answer processing, classification,
merged turn handling, initial context extraction, and competitor extraction.
"""

from typing import Optional
from backend.app.prompts.question_selection import QUESTION_GEN_SYSTEM_PROMPT

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
    "CRITICAL PRICING & OFFER RULE:\n"
    "- Use the semantic meaning and conversation context of the user's answer, NOT just the presence of a monetary value.\n"
    "- pricing_offer_structure = the commercial structure, pricing, prices, tiers, fees, or packages of the product or service being sold to customers.\n"
    "- If the active requirement is pricing_offer_structure but the user's answer refers to internal marketing spend/budget rather than product/service pricing, set active_requirement status to 'unavailable' (value: null).\n\n"
    "CRITICAL TARGET MARKET LOCATION RULE:\n"
    "- When the active requirement is target_market_location:\n"
    "  * Any user response specifying geographic targeting — including specific cities (e.g., Mumbai, Pune, Ahmedabad), states, regions (e.g., South India, Delhi NCR), Tier-1 / Tier-2 / Tier-3 cities, national scope (e.g., 'Pan-India', 'All India', 'nationwide'), or international regions — is a valid concrete location.\n"
    "  * Classify status as 'known' and extract their stated geographic target as 'value'.\n"
    "  * Only mark as 'unavailable' if the user explicitly states they do not know, have not decided, or refuse to specify a geographic target.\n"
    "  * Do NOT mark as 'unavailable' if the user provided 'Pan-India', a city name, or a region.\n\n"
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

_ANSWER_RULES = ANSWER_CLASSIFICATION_SYSTEM_PROMPT.split("OUTPUT FORMAT:")[0].strip()

_PRIORITIZATION_SECTION = (
    "REQUIREMENT SELECTION PRIORITIZATION RULES:\n"
    "1. Prefer selecting an existing UNKNOWN candidate requirement ID from the provided candidate list.\n"
    "2. ONLY IF no remaining candidate requirement covers an important strategic information gap for this specific business, you may propose creating ONE dynamic requirement using 'create_dynamic'.\n"
    "3. Priority items (target_market_location, customer_needs_buying_behavior, competitive_landscape, usp_differentiation, pricing_offer_structure, sales_conversion_journey) should be evaluated systematically.\n"
    "4. GROUNDEDNESS & NO DUPLICATE RULE:\n"
    "   - target_market_location: Do NOT infer target market location from demographic audience mentions (e.g. 'Indian consumers aged 18–35') or business address. Select 'target_market_location' unless the user has explicitly stated their target geographic scope (e.g. cities, states, regions, Tier-1/Tier-2/Tier-3 cities, or Pan-India).\n"
    "   - Do NOT select a requirement if its information is already adequately covered by business context, past marketing document summary, online presence context (website/social media scrape), or previous user answers.\n"
    "   - PRIORITY / CONFLICT RULE: User-provided explicit information strictly overrides scraped online presence information if they conflict.\n"
    "   - Do NOT ask for information already covered by online presence context (e.g. product features/USP, pricing, or current marketing channels).\n"
    "   - HISTORICAL EVIDENCE VS CURRENT INPUTS: Historical document evidence reflects past campaigns and performance. Do NOT confuse historical PDF numbers (e.g. past budget, past campaigns) with current business parameters. Do NOT ask redundant questions about past campaigns already documented in the PDF, but DO ask questions needed for the current strategy if current information is missing.\n"
    "5. DYNAMIC REQUIREMENT CREATION RULES:\n"
    "   - Do NOT create a dynamic requirement if an existing requirement covers the topic, if nice-to-have, if already known, or if rephrasing an existing requirement.\n"
    "   - Create a dynamic requirement ONLY when: strategically important + genuinely missing + specific to this business + not covered by existing requirements."
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
    '  "next": null OR {\n'
    '    "selected_id": "<existing_requirement_id from candidates list or null>",\n'
    '    "create_dynamic": null OR {"id": "<short_id>", "title": "<Title>", "description": "<Desc>"},\n'
    '    "question": "<short direct conversational question>",\n'
    '    "reasoning": "<short explanation>"\n'
    "  }\n"
    "}\n"
    "IMPORTANT FOR 'next': Choose 'selected_id' from remaining UNKNOWN candidate requirements, OR use 'create_dynamic' if an uncovered strategic gap exists. If no further questions are needed, set 'next' to null.\n"
    "Do NOT include markdown formatting or commentary outside the JSON object."
)

COMPETITOR_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert marketing strategy assistant extracting competitor names, brand names, and competitor websites from a client's response.\n\n"
    "RULES:\n"
    "1. Extract ONLY competitor names, company names, brand names, competitor website URLs/domains, or specific alternative business options explicitly mentioned in the user's text.\n"
    "2. Do NOT invent, fabricate, or assume any company names, brand names, or competitors not present in the user's input.\n"
    "3. Preserve competitor websites or domain URLs if provided (e.g., 'https://sleepyowl.co' or 'bluetokai.com').\n"
    "4. Do NOT include the client's OWN company or brand name as a competitor.\n"
    "5. Do NOT classify generic home activities, behavioral substitutes, or DIY concepts (e.g., 'making coffee at home', 'brewing at home', 'doing it yourself') as competitor brands.\n"
    "6. If the user mentions generic unnamed competitors (e.g., 'two local tiffin services', 'local bakeries'), preserve the user's descriptive wording as an item in the list.\n"
    "7. If the user explicitly states they have no competitors or direct alternatives (e.g., 'No competitors', 'We have no direct competitors', 'None'), return an empty list: [].\n"
    "8. Avoid duplicate competitor names or URLs (case-insensitive).\n"
    "9. Output MUST be a JSON object with key 'competitors': list of strings.\n\n"
    "Example 1:\n"
    "User answer: \"Swiggy, Zomato and two local tiffin services\"\n"
    "Output: {\"competitors\": [\"Swiggy\", \"Zomato\", \"two local tiffin services\"]}\n\n"
    "Example 2:\n"
    "User answer: \"Our main rivals are Blue Tokai and https://sleepyowl.co, though people also make coffee at home.\"\n"
    "Output: {\"competitors\": [\"Blue Tokai\", \"https://sleepyowl.co\"]}\n\n"
    "Example 3:\n"
    "User answer: \"We don't have any direct competitors in our area.\"\n"
    "Output: {\"competitors\": []}"
)


def build_initial_context_prompt(user_message: str) -> str:
    """
    Builds the user prompt for extracting structured business context from initial message.
    """
    return f"USER MESSAGE:\n\"{user_message.strip()}\"\n\nExtract business context fields and return the JSON object."


def build_competitor_extraction_prompt(user_answer: str) -> str:
    """
    Builds the user prompt for extracting competitor names and alternatives.
    """
    return f"USER ANSWER:\n\"{user_answer.strip()}\"\n\nExtract competitor names or alternatives and return the JSON object."


def build_answer_classification_prompt(
    active_req_id: str,
    req_title: str,
    req_desc: str,
    question_text: Optional[str],
    other_reqs_str: str,
    user_answer: str,
) -> str:
    """
    Builds the user prompt for evaluating a user's answer against the active requirement and candidates.
    """
    return (
        f"ACTIVE REQUIREMENT TO CLARIFY:\n"
        f"- ID: {active_req_id}\n"
        f"- Title: {req_title}\n"
        f"- Description: {req_desc}\n"
        f"- Question Asked: \"{question_text or 'Not provided'}\"\n\n"
        f"OTHER MISSING CANDIDATE REQUIREMENTS:\n{other_reqs_str}\n\n"
        f"CLIENT RESPONSE MESSAGE:\n\"{user_answer.strip()}\"\n\n"
        "Evaluate the response against the active requirement and other candidates, then output the JSON object."
    )


def build_merged_turn_prompt(
    context_str: str,
    active_req_id: str,
    req_title: str,
    req_desc: str,
    question_text: Optional[str],
    user_answer: str,
    other_reqs_str: str,
) -> str:
    """
    Builds the user prompt for merged turn processing (answering + planning next question).
    """
    return (
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
