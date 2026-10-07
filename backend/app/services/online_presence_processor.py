import json
import logging
from typing import List, Optional

from backend.app.models.business_context import BusinessContext
from backend.app.models.online_presence import OnlinePresenceContext, OnlineSourceSummary
from backend.app.models.scraped_result import ApifyScrapeResult
from backend.app.services.llm_service import get_llm_response, LLMServiceError

logger = logging.getLogger(__name__)

ONLINE_PRESENCE_SYSTEM_PROMPT = (
    "You are an expert marketing strategy consultant synthesizing raw scraped online data into a structured online presence summary.\n\n"
    "CRITICAL GROUNDEDNESS & ACCURACY RULES:\n"
    "1. Ground every summary statement strictly in the provided scraped data and business context. Do NOT invent, assume, or fabricate any unmentioned information.\n"
    "2. The LLM must NOT invent products, prices, competitors, features, marketing channels, customer behavior, engagement metrics, or performance results.\n"
    "3. CONTEXT-AWARE EVALUATION: Use the provided business name, product/service, and target audience to identify relevant information. Recognize that a website may contain multiple products/services; prioritize details relevant to the target product/service when found, but do not ignore other legitimate business offerings.\n"
    "4. IF INFORMATION IS MISSING OR LIMITED: If specific information (e.g., pricing, positioning, or specific campaigns) is not found in the scraped data for a platform, leave that list empty or state that it was not found. Do not fill empty lists with generic filler.\n"
    "5. FAILED / ABSENT SOURCES: If a platform (website, instagram, facebook, or linkedin) has no scraped data or failed scraping, set its key to null in the output.\n\n"
    "REQUIRED JSON OUTPUT SCHEMA:\n"
    "You MUST respond ONLY with a valid JSON object strictly matching this schema:\n"
    "{\n"
    '  "overall_summary": "Concise 2-3 sentence overview synthesizing the brand\'s online presence across all available platforms",\n'
    '  "website": {\n'
    '    "summary": "Concise summary of website content",\n'
    '    "relevant_products_or_services": ["Product 1", "Service 2"],\n'
    '    "marketing_content": ["Key theme or content topic"],\n'
    '    "positioning_or_messaging": ["Brand slogan or positioning statement"],\n'
    '    "other_strategy_relevant_information": ["Additional insight"]\n'
    '  } | null,\n'
    '  "instagram": {\n'
    '    "summary": "Summary of Instagram content",\n'
    '    "relevant_products_or_services": [],\n'
    '    "marketing_content": [],\n'
    '    "positioning_or_messaging": [],\n'
    '    "other_strategy_relevant_information": []\n'
    '  } | null,\n'
    '  "facebook": null,\n'
    '  "linkedin": null\n'
    "}\n"
    "Do NOT include markdown formatting outside the JSON object."
)


def _format_instagram_item(item: dict) -> str:
    """
    Formats a single scraped Instagram item using specifically selected strategy fields:
    caption, hashtags, mentions, timestamp, likesCount, ownerUsername, ownerFullName,
    type, videoViewCount, paidPartnership, taggedUsers, url.

    Excludes comment clutter (commentsCount, latestComments, etc.) and media/technical properties.
    """
    parts = []

    # Owner info
    owner = item.get("owner")
    owner_username = item.get("ownerUsername") or (owner.get("username") if isinstance(owner, dict) else None)
    owner_fullname = item.get("ownerFullName") or (owner.get("fullName") if isinstance(owner, dict) else None)
    if owner_username or owner_fullname:
        author_str = f"Author: {owner_fullname or owner_username}"
        if owner_username and owner_fullname:
            author_str += f" (@{owner_username})"
        parts.append(author_str)

    # Post type & timestamp
    post_type = item.get("type")
    if post_type:
        parts.append(f"Type: {post_type}")

    timestamp = item.get("timestamp")
    if timestamp:
        parts.append(f"Timestamp: {timestamp}")

    # Caption / Text
    caption = item.get("caption") or item.get("text")
    if caption and isinstance(caption, str):
        parts.append(f"Caption: {caption.strip()}")

    # Hashtags
    hashtags = item.get("hashtags")
    if hashtags and isinstance(hashtags, list):
        clean_tags = [str(t).strip() for t in hashtags if t]
        if clean_tags:
            parts.append(f"Hashtags: {', '.join(clean_tags)}")

    # Mentions
    mentions = item.get("mentions")
    if mentions and isinstance(mentions, list):
        clean_mentions = [str(m).strip() for m in mentions if m]
        if clean_mentions:
            parts.append(f"Mentions: {', '.join(clean_mentions)}")

    # Tagged Users
    tagged = item.get("taggedUsers")
    if tagged and isinstance(tagged, list):
        tagged_names = []
        for u in tagged:
            if isinstance(u, dict):
                uname = u.get("username") or u.get("fullName")
                if uname:
                    tagged_names.append(str(uname))
            elif isinstance(u, str):
                tagged_names.append(u.strip())
        if tagged_names:
            parts.append(f"Tagged Users: {', '.join(tagged_names)}")

    # Engagement Metrics
    likes = item.get("likesCount")
    if likes is not None and isinstance(likes, (int, float)):
        parts.append(f"Likes: {int(likes)}")

    views = item.get("videoViewCount")
    if views is not None and isinstance(views, (int, float)):
        parts.append(f"Video Views: {int(views)}")

    # Paid Partnership
    paid = item.get("paidPartnership")
    if paid is None:
        paid = item.get("isPaidPartnership")
    if paid is not None:
        parts.append(f"Paid Partnership: {bool(paid)}")

    # URL
    url = item.get("url")
    if url and isinstance(url, str):
        parts.append(f"URL: {url.strip()}")

    return " | ".join(parts)


def _format_scraped_data_for_prompt(scraped_results: List[ApifyScrapeResult]) -> str:
    """
    Formats raw scraped dataset items from Apify results into a clean, text-based prompt snippet per platform.
    Uses platform-specific formatting for Instagram while retaining generic extraction for other sources.
    """
    platform_texts = {}

    for res in scraped_results:
        if not res.success or not res.data:
            continue

        platform = res.platform
        items_snippets = []

        for item in res.data[:10]:  # Limit to top 10 items per dataset
            if platform == "instagram" and isinstance(item, dict):
                formatted = _format_instagram_item(item)
                if formatted:
                    items_snippets.append(formatted)
            elif isinstance(item, dict):
                # Extract text fields dynamically for other platforms
                parts = []
                for key in ("title", "text", "caption", "post_text", "description", "name", "url"):
                    val = item.get(key)
                    if val and isinstance(val, str):
                        parts.append(f"{key}: {val.strip()}")
                if parts:
                    items_snippets.append(" | ".join(parts[:4]))
            elif isinstance(item, str):
                items_snippets.append(item.strip())

        if items_snippets:
            platform_texts[platform] = "\n".join(items_snippets[:8])

    if not platform_texts:
        return "No valid scraped data content available."

    formatted_sections = []
    for platform, text_content in platform_texts.items():
        formatted_sections.append(f"--- PLATFORM: {platform.upper()} ---\n{text_content}")

    return "\n\n".join(formatted_sections)


def process_online_presence(
    scraped_results: List[ApifyScrapeResult],
    business_context: BusinessContext
) -> Optional[OnlinePresenceContext]:
    """
    Processes raw Apify scrape results using LLM to extract a structured, grounded OnlinePresenceContext.
    Returns None if no scraped results succeeded or if processing fails.
    """
    if not scraped_results:
        return None

    successful_results = [r for r in scraped_results if r.success and r.data]
    if not successful_results:
        logger.info("No successful Apify scrape results to process for OnlinePresenceContext.")
        return None

    formatted_data = _format_scraped_data_for_prompt(successful_results)
    if formatted_data == "No valid scraped data content available.":
        return None

    company_name = business_context.company_name or "Not provided"
    product_or_service = business_context.product_or_service or "Not provided"
    marketing_goal = business_context.marketing_goal or "Not provided"
    target_audience = business_context.target_audience or "Not provided"

    user_prompt_parts = [
        "BUSINESS CONTEXT:",
        f"Company Name: {company_name}",
        f"Product/Service: {product_or_service}",
        f"Marketing Goal: {marketing_goal}",
        f"Target Audience: {target_audience}",
        "\nSCRAPED ONLINE DATA:",
        formatted_data
    ]
    user_prompt = "\n".join(user_prompt_parts)

    try:
        logger.info("Calling LLM to process scraped online presence into OnlinePresenceContext...")
        llm_output = get_llm_response(
            system_prompt=ONLINE_PRESENCE_SYSTEM_PROMPT,
            prompt=user_prompt,
            temperature=0.2
        )

        cleaned_output = llm_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
        cleaned_output = cleaned_output.strip()

        data = json.loads(cleaned_output)
        context_obj = OnlinePresenceContext.model_validate(data)
        logger.info("Successfully generated and validated OnlinePresenceContext.")
        return context_obj

    except Exception as e:
        logger.error(f"Failed to process online presence with LLM: {e}")
        return None
