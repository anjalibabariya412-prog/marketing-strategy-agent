"""
Prompt definitions and prompt builders for synthesizing online presence
and website/social media scraping data.
"""

ONLINE_PRESENCE_SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant synthesizing raw scraped online data into a structured online presence summary.\n\n"
    "CRITICAL GROUNDEDNESS & ACCURACY RULES:\n"
    "1. Ground every summary statement strictly in the provided scraped data and business context. Do NOT invent, assume, or fabricate any unmentioned information.\n"
    "2. The LLM must NOT invent products, prices, competitors, features, marketing channels, customer behavior, engagement metrics, or performance results.\n"
    "3. CONTEXT-AWARE EVALUATION: Use the provided business name, product/service, and target audience to identify relevant information. Recognize that a website may contain multiple products/services; prioritize details relevant to the target product/service when found, but do not ignore other legitimate business offerings.\n"
    "4. IF INFORMATION IS MISSING OR LIMITED: If specific information (e.g., pricing, positioning, or specific campaigns) is not found in the scraped data for a platform, leave that list empty or state that it was not found. Do not fill empty lists with generic filler.\n"
    "5. FAILED / ABSENT SOURCES: If a platform (website, instagram, facebook, or linkedin) has no scraped data or failed scraping, set its key to null in the output.\n"
    "6. CLAIM-LEVEL EVIDENCE & CONDITIONS PRESERVATION:\n"
    "   For any promotional claims, discounts, free shipping offers, guarantees, or special value propositions found in scraped content, extract structured 'evidence' records under 'website' (and other platforms if applicable).\n"
    "   Each evidence record MUST contain:\n"
    "   - 'claim': Concise description of the claim (e.g. 'Free standard shipping on orders above ₹350', '10% discount on first coffee purchase').\n"
    "   - 'source_url': The exact URL of the specific scraped page where this claim or passage was found. Do NOT invent or generalize URLs.\n"
    "   - 'supporting_text': The exact quote or passage from the retrieved content demonstrating the claim (e.g. 'FREE STANDARD SHIPPING on all prepaid orders above ₹350').\n"
    "   - 'conditions': Any restrictions, eligibility criteria, payment method constraints, minimum order values, or terms attached to the claim (e.g. 'Prepaid orders only', 'First coffee purchase only; use code TRY10', 'Minimum order ₹350'). If no conditions are stated in the source, set to null. Extract conditions ONLY when supported by retrieved text; do not invent them.\n"
    "   - 'verification_status': Must be one of:\n"
    "     * 'supported': The exact text directly corroborates the claim and its stated terms.\n"
    "     * 'partially_supported': The claim is mentioned but with significant caveats, restrictions, or assumptions not fully matching the raw statement.\n"
    "     * 'unsupported': The text contradicts the claim or does not substantiate it.\n"
    "     * 'unresolved': The claim cannot be reliably verified or linked to the retrieved text.\n"
    "   CRITICAL RULE: Do NOT treat information found on the live website but absent from retrieved scraped content as evidence. If a claim cannot be reliably linked to evidence in the retrieved text, mark it 'unresolved' or 'unsupported' rather than directly supported.\n\n"
    "REQUIRED JSON OUTPUT SCHEMA:\n"
    "You MUST respond ONLY with a valid JSON object strictly matching this schema:\n"
    "{\n"
    '  "overall_summary": "Concise 2-3 sentence overview synthesizing the brand\'s online presence across all available platforms",\n'
    '  "website": {\n'
    '    "summary": "Concise summary of website content",\n'
    '    "relevant_products_or_services": ["Product 1", "Service 2"],\n'
    '    "marketing_content": ["Key theme or content topic"],\n'
    '    "positioning_or_messaging": ["Brand slogan or positioning statement"],\n'
    '    "other_strategy_relevant_information": ["Additional insight"],\n'
    '    "evidence": [\n'
    '      {\n'
    '        "claim": "Free standard shipping on orders above ₹350",\n'
    '        "source_url": "https://example.com/products/item-1",\n'
    '        "supporting_text": "FREE STANDARD SHIPPING on all prepaid orders above ₹350",\n'
    '        "conditions": "Prepaid orders only",\n'
    '        "verification_status": "supported"\n'
    '      }\n'
    '    ]\n'
    '  } | null,\n'
    '  "instagram": {\n'
    '    "summary": "Summary of Instagram content",\n'
    '    "relevant_products_or_services": [],\n'
    '    "marketing_content": [],\n'
    '    "positioning_or_messaging": [],\n'
    '    "other_strategy_relevant_information": [],\n'
    '    "evidence": []\n'
    '  } | null,\n'
    '  "facebook": null,\n'
    '  "linkedin": null\n'
    "}\n"
    "Do NOT include markdown formatting outside the JSON object."
)


def build_online_presence_prompt(
    company_name: str,
    product_or_service: str,
    marketing_goal: str,
    target_audience: str,
    formatted_data: str,
) -> str:
    """
    Builds the user prompt for synthesizing scraped online data into structured OnlinePresenceContext.
    """
    user_prompt_parts = [
        "BUSINESS CONTEXT:",
        f"Company Name: {company_name}",
        f"Product/Service: {product_or_service}",
        f"Marketing Goal: {marketing_goal}",
        f"Target Audience: {target_audience}",
        "\nSCRAPED ONLINE DATA:",
        formatted_data,
    ]
    return "\n".join(user_prompt_parts)
