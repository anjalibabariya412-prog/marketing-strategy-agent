from datetime import datetime, timezone
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.app.models.business_context import BusinessContext
from backend.app.models.online_presence import (
    MarketingEvidence,
    OnlinePresenceContext,
    OnlineSourceSummary,
    VerificationStatus,
)
from backend.app.models.parsed_url import ParsedURL
from backend.app.models.scraped_result import ApifyScrapeResult
from backend.app.services.llm_service import get_llm_response, LLMServiceError
from backend.app.db.session import SessionLocal
from backend.app.db.orm.online_presence_source import OnlinePresenceSource

logger = logging.getLogger(__name__)

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


def _format_instagram_profile_item_for_prompt(item: dict) -> str:
    """Formats normalized Instagram profile data into a concise summary line for the LLM prompt."""
    parts = ["[Instagram Profile]"]
    username = item.get("username")
    full_name = item.get("fullName")
    if username or full_name:
        author = f"Username: @{username}" if username else ""
        if full_name:
            author = f"{author} ({full_name})" if author else f"Name: {full_name}"
        parts.append(author)

    bio = item.get("biography")
    if bio and isinstance(bio, str):
        parts.append(f"Bio: {bio.strip()}")

    category = item.get("businessCategoryName")
    if category and isinstance(category, str):
        parts.append(f"Category: {category.strip()}")

    followers = item.get("followersCount")
    if followers is not None and isinstance(followers, (int, float)):
        parts.append(f"Followers: {int(followers)}")

    follows = item.get("followsCount")
    if follows is not None and isinstance(follows, (int, float)):
        parts.append(f"Following: {int(follows)}")

    posts_count = item.get("postsCount")
    if posts_count is not None and isinstance(posts_count, (int, float)):
        parts.append(f"Total Posts: {int(posts_count)}")

    return " | ".join(parts)


def _format_instagram_post_item_for_prompt(item: dict) -> str:
    """Formats normalized Instagram post data into a concise summary line for the LLM prompt."""
    parts = ["[Instagram Post]"]
    post_type = item.get("type")
    if post_type:
        parts.append(f"Type: {post_type}")

    timestamp = item.get("timestamp")
    if timestamp:
        parts.append(f"Timestamp: {timestamp}")

    caption = item.get("caption")
    if caption and isinstance(caption, str):
        parts.append(f"Caption: {caption.strip()}")

    hashtags = item.get("hashtags")
    if hashtags and isinstance(hashtags, list):
        clean_tags = [str(t).strip() for t in hashtags if t]
        if clean_tags:
            parts.append(f"Hashtags: {', '.join(clean_tags)}")

    likes = item.get("likesCount")
    if likes is not None and isinstance(likes, (int, float)):
        parts.append(f"Likes: {int(likes)}")

    comments = item.get("commentsCount")
    if comments is not None and isinstance(comments, (int, float)):
        parts.append(f"Comments: {int(comments)}")

    views = item.get("videoViewCount")
    if views is not None and isinstance(views, (int, float)):
        parts.append(f"Video Views: {int(views)}")

    url = item.get("url")
    if url and isinstance(url, str):
        parts.append(f"URL: {url.strip()}")

    return " | ".join(parts)


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


def _format_facebook_page_item_for_prompt(item: dict) -> str:
    """Formats normalized Facebook page data into a concise summary line for the LLM prompt."""
    parts = ["[Facebook Page]"]
    page_name = item.get("pageName") or item.get("username")
    if page_name:
        parts.append(f"Page Name: {page_name}")
    category = item.get("category")
    if category and isinstance(category, str):
        parts.append(f"Category: {category.strip()}")
    about = item.get("about")
    if about and isinstance(about, str):
        parts.append(f"About: {about.strip()}")
    followers = item.get("followersCount")
    if followers is not None and isinstance(followers, (int, float)):
        parts.append(f"Followers: {int(followers)}")
    likes = item.get("likesCount")
    if likes is not None and isinstance(likes, (int, float)):
        parts.append(f"Page Likes: {int(likes)}")
    website = item.get("website")
    if website and isinstance(website, str):
        parts.append(f"Website: {website.strip()}")
    return " | ".join(parts)


def _format_facebook_post_item_for_prompt(item: dict) -> str:
    """Formats normalized Facebook post data into a concise summary line for the LLM prompt."""
    parts = ["[Facebook Post]"]
    post_type = item.get("type")
    if post_type:
        parts.append(f"Type: {post_type}")
    timestamp = item.get("timestamp")
    if timestamp:
        parts.append(f"Timestamp: {timestamp}")
    text = item.get("text")
    if text and isinstance(text, str):
        parts.append(f"Content: {text.strip()}")
    likes = item.get("likesCount")
    if likes is not None and isinstance(likes, (int, float)):
        parts.append(f"Likes: {int(likes)}")
    comments = item.get("commentsCount")
    if comments is not None and isinstance(comments, (int, float)):
        parts.append(f"Comments: {int(comments)}")
    shares = item.get("sharesCount")
    if shares is not None and isinstance(shares, (int, float)):
        parts.append(f"Shares: {int(shares)}")
    url = item.get("url")
    if url and isinstance(url, str):
        parts.append(f"URL: {url.strip()}")
    return " | ".join(parts)


def _format_website_page_item_for_prompt(item: dict) -> str:
    """Formats a scraped website page into a structured text snippet preserving page URL and markdown content."""
    url = item.get("url") or item.get("source_url") or "URL not specified"
    title = item.get("title") or ""
    markdown_content = (item.get("markdown") or item.get("text") or item.get("content") or "").strip()

    parts = [f"[Website Page URL: {url}]"]
    if title and isinstance(title, str) and title.strip():
        parts.append(f"Page Title: {title.strip()}")
    if markdown_content and isinstance(markdown_content, str) and markdown_content.strip():
        parts.append(f"Page Content:\n{markdown_content.strip()}")
    return "\n".join(parts)


def _format_scraped_data_for_prompt(scraped_results: List[ApifyScrapeResult]) -> str:
    """
    Formats raw scraped dataset items from Apify/Firecrawl results into a clean, text-based prompt snippet per platform.
    Uses platform-specific formatting for Website, Instagram, and Facebook while retaining generic extraction for other sources.
    """
    platform_texts = {}

    for res in scraped_results:
        if not res.success or not res.data:
            continue

        platform = res.platform
        items_snippets = []

        if platform == "website":
            if isinstance(res.data, list):
                for page in res.data[:10]:
                    if isinstance(page, dict):
                        snippet = _format_website_page_item_for_prompt(page)
                        if snippet:
                            items_snippets.append(snippet)
                    elif isinstance(page, str):
                        items_snippets.append(page.strip())
            elif isinstance(res.data, dict):
                snippet = _format_website_page_item_for_prompt(res.data)
                if snippet:
                    items_snippets.append(snippet)
        elif platform == "instagram":
            if isinstance(res.data, list):
                for entry in res.data:
                    if isinstance(entry, dict) and entry.get("type") == "profile":
                        for prof in entry.get("items", []):
                            if isinstance(prof, dict):
                                snippet = _format_instagram_profile_item_for_prompt(prof)
                                if snippet:
                                    items_snippets.append(snippet)
                    elif isinstance(entry, dict) and entry.get("type") == "posts":
                        for post in entry.get("items", [])[:10]:
                            if isinstance(post, dict):
                                snippet = _format_instagram_post_item_for_prompt(post)
                                if snippet:
                                    items_snippets.append(snippet)
                    elif isinstance(entry, dict):
                        snippet = _format_instagram_item(entry)
                        if snippet:
                            items_snippets.append(snippet)
            elif isinstance(res.data, dict):
                snippet = _format_instagram_item(res.data)
                if snippet:
                    items_snippets.append(snippet)
        elif platform == "facebook":
            if isinstance(res.data, list):
                for entry in res.data:
                    if isinstance(entry, dict) and entry.get("type") == "page":
                        for pg in entry.get("items", []):
                            if isinstance(pg, dict):
                                snippet = _format_facebook_page_item_for_prompt(pg)
                                if snippet:
                                    items_snippets.append(snippet)
                    elif isinstance(entry, dict) and entry.get("type") == "posts":
                        for post in entry.get("items", [])[:10]:
                            if isinstance(post, dict):
                                snippet = _format_facebook_post_item_for_prompt(post)
                                if snippet:
                                    items_snippets.append(snippet)
                    elif isinstance(entry, dict):
                        snippet = _format_facebook_post_item_for_prompt(entry)
                        if snippet:
                            items_snippets.append(snippet)
            elif isinstance(res.data, dict):
                snippet = _format_facebook_page_item_for_prompt(res.data)
                if snippet:
                    items_snippets.append(snippet)
        else:
            for item in res.data[:10]:  # Limit to top 10 items per dataset
                if isinstance(item, dict):
                    # Extract text fields dynamically for other platforms (markdown preferred, text as fallback)
                    parts = []
                    for key in ("title", "markdown", "text", "caption", "post_text", "description", "name", "url"):
                        val = item.get(key)
                        if val and isinstance(val, str):
                            parts.append(f"{key}: {val.strip()}")
                    if parts:
                        items_snippets.append(" | ".join(parts[:4]))
                elif isinstance(item, str):
                    items_snippets.append(item.strip())

        if items_snippets:
            separator = "\n\n--- NEXT PAGE ---\n\n" if platform == "website" else "\n"
            platform_texts[platform] = separator.join(items_snippets[:8])

    if not platform_texts:
        return "No valid scraped data content available."

    formatted_sections = []
    for platform, text_content in platform_texts.items():
        formatted_sections.append(f"--- PLATFORM: {platform.upper()} ---\n{text_content}")

    return "\n\n".join(formatted_sections)


def _extract_instagram_data(
    scraped_results: List[ApifyScrapeResult],
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts normalized profile dict (7 fields) and list of post dicts (8 fields)
    from Instagram scrape results while keeping them strictly separated.
    """
    profile_data: Optional[Dict[str, Any]] = None
    posts_data: List[Dict[str, Any]] = []

    for res in scraped_results:
        if res.platform == "instagram" and res.success and res.data:
            if isinstance(res.data, list):
                for entry in res.data:
                    if isinstance(entry, dict) and entry.get("type") == "profile":
                        items = entry.get("items", [])
                        if items and isinstance(items, list) and isinstance(items[0], dict):
                            profile_data = items[0]
                    elif isinstance(entry, dict) and entry.get("type") == "posts":
                        items = entry.get("items", [])
                        if isinstance(items, list):
                            posts_data.extend([p for p in items if isinstance(p, dict)])
                    elif isinstance(entry, dict):
                        if "username" in entry and profile_data is None:
                            profile_data = entry
                        elif "caption" in entry or "likesCount" in entry:
                            posts_data.append(entry)
            elif isinstance(res.data, dict):
                if res.data.get("type") == "profile":
                    items = res.data.get("items", [])
                    if items and isinstance(items, list) and isinstance(items[0], dict):
                        profile_data = items[0]
                elif res.data.get("type") == "posts":
                    items = res.data.get("items", [])
                    if isinstance(items, list):
                        posts_data.extend([p for p in items if isinstance(p, dict)])
                elif "profile" in res.data or "posts" in res.data:
                    profile_data = res.data.get("profile")
                    posts_data = res.data.get("posts", []) or []

    return profile_data, posts_data


def _extract_facebook_data(
    scraped_results: List[ApifyScrapeResult],
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts normalized page dict (7 fields) and list of post dicts (7 fields)
    from Facebook scrape results while keeping them strictly separated.
    """
    page_data: Optional[Dict[str, Any]] = None
    posts_data: List[Dict[str, Any]] = []

    for res in scraped_results:
        if res.platform == "facebook" and res.success and res.data:
            if isinstance(res.data, list):
                for entry in res.data:
                    if isinstance(entry, dict) and entry.get("type") == "page":
                        items = entry.get("items", [])
                        if items and isinstance(items, list) and isinstance(items[0], dict):
                            page_data = items[0]
                    elif isinstance(entry, dict) and entry.get("type") == "posts":
                        items = entry.get("items", [])
                        if isinstance(items, list):
                            posts_data.extend([p for p in items if isinstance(p, dict)])
                    elif isinstance(entry, dict):
                        if "pageName" in entry and page_data is None:
                            page_data = entry
                        elif "text" in entry or "sharesCount" in entry:
                            posts_data.append(entry)
            elif isinstance(res.data, dict):
                if res.data.get("type") == "page":
                    items = res.data.get("items", [])
                    if items and isinstance(items, list) and isinstance(items[0], dict):
                        page_data = items[0]
                elif res.data.get("type") == "posts":
                    items = res.data.get("items", [])
                    if isinstance(items, list):
                        posts_data.extend([p for p in items if isinstance(p, dict)])
                elif "profile" in res.data or "posts" in res.data or "page" in res.data:
                    page_data = res.data.get("page") or res.data.get("profile")
                    posts_data = res.data.get("posts", []) or []

    return page_data, posts_data


def _sanitize_evidence_list(
    raw_evidence: Any,
    fallback_url: Optional[str] = None
) -> List[MarketingEvidence]:
    """
    Validates and safely coerces raw LLM-produced evidence records into MarketingEvidence instances.
    Safely skips or repairs malformed entries so processing never crashes.
    """
    if not raw_evidence or not isinstance(raw_evidence, list):
        return []

    sanitized: List[MarketingEvidence] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue

        raw_claim = item.get("claim")
        if not raw_claim or not str(raw_claim).strip():
            continue
        claim = str(raw_claim).strip()

        source_url = str(item.get("source_url") or fallback_url or "").strip()
        raw_supporting = item.get("supporting_text")
        supporting_text = str(raw_supporting).strip() if raw_supporting else ""

        conditions = item.get("conditions")
        raw_status = item.get("verification_status")

        # If supporting_text is missing or empty, status should be UNRESOLVED
        if not supporting_text:
            verification_status = VerificationStatus.UNRESOLVED
        elif raw_status:
            try:
                verification_status = MarketingEvidence.normalize_status(raw_status)
            except Exception:
                verification_status = VerificationStatus.UNRESOLVED
        else:
            verification_status = VerificationStatus.SUPPORTED

        try:
            ev = MarketingEvidence(
                claim=claim,
                source_url=source_url,
                supporting_text=supporting_text,
                conditions=conditions,
                verification_status=verification_status,
            )
            sanitized.append(ev)
        except Exception as val_err:
            logger.warning(f"Skipping malformed evidence entry: {item} - Error: {val_err}")

    return sanitized


def process_online_presence(
    scraped_results: List[ApifyScrapeResult],
    business_context: BusinessContext,
    strategy_request_id: Optional[Any] = None,
    conversation_id: Optional[Any] = None,
) -> Optional[OnlinePresenceContext]:
    """
    Processes raw Apify scrape results using LLM to extract a structured, grounded OnlinePresenceContext.
    Attaches normalized Instagram profile and post data while preserving their strict separation.
    Returns None if no scraped results succeeded or if processing fails.
    """
    if not scraped_results:
        return None

    successful_results = [r for r in scraped_results if r.success and r.data]
    if not successful_results:
        logger.info("No successful Apify scrape results to process for OnlinePresenceContext.")
        return None

    instagram_profile, instagram_posts = _extract_instagram_data(successful_results)

    facebook_page, facebook_posts = _extract_facebook_data(successful_results)

    formatted_data = _format_scraped_data_for_prompt(successful_results)
    if formatted_data == "No valid scraped data content available.":
        # If we have extracted Instagram or Facebook data even without formatted prompt, construct context
        if instagram_profile or instagram_posts or facebook_page or facebook_posts:
            summary_parts = []
            if instagram_profile or instagram_posts:
                summary_parts.append("Instagram presence")
            if facebook_page or facebook_posts:
                summary_parts.append("Facebook presence")
            summary_text = f"Scraped {', '.join(summary_parts)}."

            return OnlinePresenceContext(
                overall_summary=summary_text,
                instagram=OnlineSourceSummary(
                    summary=f"Instagram presence for @{instagram_profile.get('username')}" if instagram_profile and instagram_profile.get("username") else "Scraped Instagram presence.",
                    profile=instagram_profile,
                    posts=instagram_posts,
                ) if (instagram_profile or instagram_posts) else None,
                facebook=OnlineSourceSummary(
                    summary=f"Facebook presence for {facebook_page.get('pageName') or facebook_page.get('username')}" if facebook_page and (facebook_page.get("pageName") or facebook_page.get("username")) else "Scraped Facebook presence.",
                    profile=facebook_page,
                    posts=facebook_posts,
                ) if (facebook_page or facebook_posts) else None,
            )
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
            operation="online_presence_analysis",
            strategy_request_id=strategy_request_id,
            conversation_id=conversation_id,
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
        if not isinstance(data, dict):
            raise ValueError(f"LLM output is not a JSON dictionary: {type(data)}")

        # Find primary website URL for fallback if needed
        primary_website_url = None
        for r in successful_results:
            if r.platform == "website":
                primary_website_url = r.original_url or r.normalized_url
                break

        # Sanitize evidence inside website if present
        if "website" in data and isinstance(data["website"], dict):
            data["website"]["evidence"] = _sanitize_evidence_list(
                data["website"].get("evidence"), fallback_url=primary_website_url
            )

        # Sanitize top-level evidence if present
        if "evidence" in data:
            data["evidence"] = _sanitize_evidence_list(
                data.get("evidence"), fallback_url=primary_website_url
            )

        context_obj = OnlinePresenceContext.model_validate(data)

        # Synchronize website evidence to top-level context_obj.evidence if not present
        if context_obj.website and context_obj.website.evidence:
            existing_keys = {(e.claim, e.source_url) for e in context_obj.evidence}
            for ev in context_obj.website.evidence:
                if (ev.claim, ev.source_url) not in existing_keys:
                    context_obj.evidence.append(ev)
                    existing_keys.add((ev.claim, ev.source_url))

        # Attach normalized Instagram profile and post datasets to context
        if instagram_profile or instagram_posts:
            if context_obj.instagram is None:
                summary_text = (
                    f"Instagram presence for @{instagram_profile.get('username')}"
                    if instagram_profile and instagram_profile.get("username")
                    else "Synthesized Instagram presence."
                )
                context_obj.instagram = OnlineSourceSummary(
                    summary=summary_text,
                    profile=instagram_profile,
                    posts=instagram_posts,
                )
            else:
                context_obj.instagram.profile = instagram_profile
                context_obj.instagram.posts = instagram_posts

        # Attach normalized Facebook page and post datasets to context
        if facebook_page or facebook_posts:
            if context_obj.facebook is None:
                summary_text = (
                    f"Facebook presence for {facebook_page.get('pageName') or facebook_page.get('username')}"
                    if facebook_page and (facebook_page.get("pageName") or facebook_page.get("username"))
                    else "Synthesized Facebook presence."
                )
                context_obj.facebook = OnlineSourceSummary(
                    summary=summary_text,
                    profile=facebook_page,
                    posts=facebook_posts,
                )
            else:
                context_obj.facebook.profile = facebook_page
                context_obj.facebook.posts = facebook_posts

        logger.info(
            f"Successfully generated and validated OnlinePresenceContext with "
            f"{len(context_obj.all_evidence)} verified evidence record(s)."
        )
        return context_obj

    except Exception as e:
        logger.error(f"Failed to process online presence with LLM: {e}")
        if instagram_profile or instagram_posts or facebook_page or facebook_posts:
            logger.info("Falling back to constructing OnlinePresenceContext with extracted social media data.")
            summary_parts = []
            if instagram_profile or instagram_posts:
                summary_parts.append("Instagram presence")
            if facebook_page or facebook_posts:
                summary_parts.append("Facebook presence")
            summary_text = f"Scraped {', '.join(summary_parts)}."

            return OnlinePresenceContext(
                overall_summary=summary_text,
                instagram=OnlineSourceSummary(
                    summary=f"Instagram presence for @{instagram_profile.get('username')}" if instagram_profile and instagram_profile.get("username") else "Scraped Instagram presence.",
                    profile=instagram_profile,
                    posts=instagram_posts,
                ) if (instagram_profile or instagram_posts) else None,
                facebook=OnlineSourceSummary(
                    summary=f"Facebook presence for {facebook_page.get('pageName') or facebook_page.get('username')}" if facebook_page and (facebook_page.get("pageName") or facebook_page.get("username")) else "Scraped Facebook presence.",
                    profile=facebook_page,
                    posts=facebook_posts,
                ) if (facebook_page or facebook_posts) else None,
            )
        return None


def save_online_presence_sources(
    strategy_request_id: uuid.UUID,
    parsed_urls: List[ParsedURL],
    scraped_results: List[ApifyScrapeResult],
    online_presence_ctx: Optional[OnlinePresenceContext] = None,
) -> None:
    """
    Persists scraping and summary results for online presence sources into the
    online_presence_sources database table.
    Enforces unique constraint per strategy_request_id and platform.
    """
    if not strategy_request_id or not parsed_urls:
        return

    seen_platforms = set()
    records_to_insert: List[OnlinePresenceSource] = []

    for parsed_url, res in zip(parsed_urls, scraped_results):
        platform = parsed_url.platform if parsed_url and parsed_url.platform else "unknown"

        # Enforce unique constraint: 1 row per platform per strategy request
        if platform in seen_platforms:
            logger.info(
                f"Skipping duplicate platform '{platform}' for strategy_request_id '{strategy_request_id}'."
            )
            continue
        seen_platforms.add(platform)

        original_url = parsed_url.original_url if parsed_url else (res.original_url if res else "")
        normalized_url = parsed_url.normalized_url if parsed_url else (res.normalized_url if res else None)

        # Determine scrape_status & error_message
        success = res.success if res else False
        error_msg = (
            res.error if res and res.error else (parsed_url.error if parsed_url and parsed_url.error else None)
        )

        if success:
            scrape_status = "success"
        elif error_msg and any(kw in error_msg.lower() for kw in ("skip", "not supported", "invalid")):
            scrape_status = "skipped"
        else:
            scrape_status = "failed"

        # Determine platform summary from OnlinePresenceContext if available
        platform_summary = None
        if online_presence_ctx:
            if platform == "website" and online_presence_ctx.website:
                platform_summary = online_presence_ctx.website.summary
            elif platform == "instagram" and online_presence_ctx.instagram:
                platform_summary = online_presence_ctx.instagram.summary
            elif platform == "facebook" and online_presence_ctx.facebook:
                platform_summary = online_presence_ctx.facebook.summary
            elif platform == "linkedin" and online_presence_ctx.linkedin:
                platform_summary = online_presence_ctx.linkedin.summary

        # Determine details JSON structure
        details_json = None
        if res and res.data is not None:
            if platform == "instagram" and online_presence_ctx and online_presence_ctx.instagram and (
                online_presence_ctx.instagram.profile or online_presence_ctx.instagram.posts
            ):
                details_json = {
                    "profile": online_presence_ctx.instagram.profile,
                    "posts": online_presence_ctx.instagram.posts or [],
                }
            elif platform == "facebook" and online_presence_ctx and online_presence_ctx.facebook and (
                online_presence_ctx.facebook.profile or online_presence_ctx.facebook.posts
            ):
                details_json = {
                    "profile": online_presence_ctx.facebook.profile,
                    "posts": online_presence_ctx.facebook.posts or [],
                }
            elif platform == "website":
                evidence_data = []
                if online_presence_ctx:
                    if online_presence_ctx.website and online_presence_ctx.website.evidence:
                        evidence_data = [ev.model_dump() for ev in online_presence_ctx.website.evidence]
                    elif online_presence_ctx.evidence:
                        evidence_data = [ev.model_dump() for ev in online_presence_ctx.evidence]

                if isinstance(res.data, list):
                    details_json = {"items": res.data}
                    if evidence_data:
                        details_json["evidence"] = evidence_data
                elif isinstance(res.data, dict):
                    details_json = dict(res.data)
                    if evidence_data:
                        details_json["evidence"] = evidence_data
                else:
                    details_json = {"raw": str(res.data)}
                    if evidence_data:
                        details_json["evidence"] = evidence_data
            elif isinstance(res.data, dict):
                details_json = res.data
            elif isinstance(res.data, list):
                details_json = {"items": res.data}
            else:
                details_json = {"raw": str(res.data)}

        scraped_at_val = datetime.now(timezone.utc) if success else None

        record = OnlinePresenceSource(
            strategy_request_id=strategy_request_id,
            platform=platform,
            original_url=original_url,
            normalized_url=normalized_url,
            scrape_status=scrape_status,
            error_message=error_msg,
            summary=platform_summary,
            details=details_json,
            scraped_at=scraped_at_val,
        )
        records_to_insert.append(record)

    if not records_to_insert:
        return

    try:
        with SessionLocal() as db:
            for record in records_to_insert:
                db.add(record)
            db.commit()
            logger.info(
                f"Successfully persisted {len(records_to_insert)} OnlinePresenceSource record(s) "
                f"for StrategyRequest '{strategy_request_id}'."
            )
    except Exception as db_err:
        logger.error(
            f"Failed to persist OnlinePresenceSource records for StrategyRequest '{strategy_request_id}': {db_err}"
        )
