import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from apify_client import ApifyClientAsync

from backend.app.core.config import settings
from backend.app.core.progress_notifier import emit_progress
from backend.app.schemas.agent.parsed_url import ParsedURL
from backend.app.schemas.agent.scraped_result import ApifyScrapeResult
from backend.app.services.firecrawl_service import scrape_website_url


logger = logging.getLogger(__name__)


ACTOR_MAPPING: Dict[str, Dict[str, Any]] = {
    "instagram": {
        "actor_id": "apify/instagram-scraper",
        "build_input": lambda url: {
            "directUrls": [url],
            "resultsLimit": 7,
        },
    },
    "instagram_profile": {
        "actor_id": "apify/instagram-profile-scraper",
        "build_input": lambda url: {
            "usernames": [url],
        },
    },
    "facebook": {
        "actor_id": "apify/facebook-posts-scraper",
        "build_input": lambda url: {
            "startUrls": [{"url": url}],
            "maxPosts": 7,
        },
    },
    "facebook_page": {
        "actor_id": "apify/facebook-pages-scraper",
        "build_input": lambda url: {
            "startUrls": [{"url": url}],
        },
    },
    "linkedin": {
        "actor_id": "automation-lab/linkedin-company-posts-scraper",
        "build_input": lambda url: {
            "companyUrls": [url],
            "maxPostsPerCompany": 5,
        },
    },
}


def normalize_instagram_profile_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw Apify Instagram profile item to standard schema fields:
    username, fullName, biography, followersCount, followsCount, postsCount, businessCategoryName
    """
    if not isinstance(item, dict):
        return {
            "username": None,
            "fullName": None,
            "biography": None,
            "followersCount": None,
            "followsCount": None,
            "postsCount": None,
            "businessCategoryName": None,
        }

    return {
        "username": item.get("username"),
        "fullName": item.get("fullName") if "fullName" in item else item.get("full_name"),
        "biography": item.get("biography") if "biography" in item else item.get("bio"),
        "followersCount": item.get("followersCount") if "followersCount" in item else item.get("followers_count"),
        "followsCount": (
            item.get("followsCount")
            if "followsCount" in item
            else (item.get("follows_count") if "follows_count" in item else item.get("followingCount"))
        ),
        "postsCount": (
            item.get("postsCount")
            if "postsCount" in item
            else (item.get("posts_count") if "posts_count" in item else item.get("mediaCount"))
        ),
        "businessCategoryName": (
            item.get("businessCategoryName")
            if "businessCategoryName" in item
            else (item.get("business_category_name") if "business_category_name" in item else item.get("categoryName"))
        ),
    }


def normalize_instagram_post_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw Apify Instagram post item to standard schema fields:
    caption, hashtags, type, likesCount, commentsCount, videoViewCount, timestamp, url
    """
    if not isinstance(item, dict):
        return {
            "caption": None,
            "hashtags": [],
            "type": None,
            "likesCount": None,
            "commentsCount": None,
            "videoViewCount": None,
            "timestamp": None,
            "url": None,
        }

    caption_val = item.get("caption") if "caption" in item else item.get("text")
    hashtags_val = item.get("hashtags") if "hashtags" in item else []
    if hashtags_val is None:
        hashtags_val = []

    return {
        "caption": caption_val,
        "hashtags": hashtags_val if isinstance(hashtags_val, list) else [hashtags_val],
        "type": item.get("type"),
        "likesCount": item.get("likesCount") if "likesCount" in item else item.get("likes_count"),
        "commentsCount": item.get("commentsCount") if "commentsCount" in item else item.get("comments_count"),
        "videoViewCount": item.get("videoViewCount") if "videoViewCount" in item else item.get("video_view_count"),
        "timestamp": item.get("timestamp"),
        "url": item.get("url"),
    }


def normalize_facebook_page_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw Apify Facebook page item to standard schema fields:
    pageName, username, about, category, followersCount, likesCount, website
    """
    if not isinstance(item, dict):
        return {
            "pageName": None,
            "username": None,
            "about": None,
            "category": None,
            "followersCount": None,
            "likesCount": None,
            "website": None,
        }

    followers_val = None
    if "followersCount" in item:
        followers_val = item.get("followersCount")
    elif "followers_count" in item:
        followers_val = item.get("followers_count")
    elif "followers" in item:
        followers_val = item.get("followers")
    elif isinstance(item.get("likes_and_followers"), dict):
        followers_val = item["likes_and_followers"].get("followers_count")

    likes_val = None
    if "likesCount" in item:
        likes_val = item.get("likesCount")
    elif "likes_count" in item:
        likes_val = item.get("likes_count")
    elif "likes" in item:
        likes_val = item.get("likes")
    elif "page_likes" in item:
        likes_val = item.get("page_likes")
    elif isinstance(item.get("likes_and_followers"), dict):
        likes_val = item["likes_and_followers"].get("likes_count")

    return {
        "pageName": (
            item.get("pageName")
            if "pageName" in item
            else (item.get("title") if "title" in item else (item.get("name") if "name" in item else item.get("page_name")))
        ),
        "username": (
            item.get("username")
            if "username" in item
            else (item.get("user_name") if "user_name" in item else item.get("handle"))
        ),
        "about": (
            item.get("about")
            if "about" in item
            else (item.get("about_text") if "about_text" in item else (item.get("description") if "description" in item else item.get("bio")))
        ),
        "category": (
            item.get("category")
            if "category" in item
            else (item.get("categories") if "categories" in item else item.get("business_category"))
        ),
        "followersCount": followers_val,
        "likesCount": likes_val,
        "website": (
            item.get("website")
            if "website" in item
            else (item.get("website_url") if "website_url" in item else item.get("url"))
        ),
    }


def normalize_facebook_post_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw Apify Facebook post item to standard schema fields:
    text, type, likesCount, commentsCount, sharesCount, timestamp, url
    """
    if not isinstance(item, dict):
        return {
            "text": None,
            "type": None,
            "likesCount": None,
            "commentsCount": None,
            "sharesCount": None,
            "timestamp": None,
            "url": None,
        }

    return {
        "text": (
            item.get("text")
            if "text" in item
            else (item.get("message") if "message" in item else (item.get("postText") if "postText" in item else (item.get("caption") if "caption" in item else item.get("content"))))
        ),
        "type": (
            item.get("type")
            if "type" in item
            else (item.get("postType") if "postType" in item else item.get("mediaType"))
        ),
        "likesCount": (
            item.get("likesCount")
            if "likesCount" in item
            else (item.get("likes_count") if "likes_count" in item else (item.get("likes") if "likes" in item else (item.get("reactionCount") if "reactionCount" in item else item.get("reactionsCount"))))
        ),
        "commentsCount": (
            item.get("commentsCount")
            if "commentsCount" in item
            else (item.get("comments_count") if "comments_count" in item else item.get("comments"))
        ),
        "sharesCount": (
            item.get("sharesCount")
            if "sharesCount" in item
            else (item.get("shares_count") if "shares_count" in item else item.get("shares"))
        ),
        "timestamp": (
            item.get("timestamp")
            if "timestamp" in item
            else (item.get("time") if "time" in item else (item.get("createdTime") if "createdTime" in item else (item.get("date") if "date" in item else item.get("publishedAt"))))
        ),
        "url": (
            item.get("url")
            if "url" in item
            else (item.get("postUrl") if "postUrl" in item else (item.get("link") if "link" in item else item.get("canonicalUrl")))
        ),
    }


async def _run_apify_actor(
    client: ApifyClientAsync,
    actor_id: str,
    run_input: dict,
) -> List[Any]:
    """Helper function to execute a single Apify Actor call and fetch dataset items."""
    run = await client.actor(actor_id).call(run_input=run_input)
    if not run:
        raise RuntimeError(f"Apify actor '{actor_id}' returned empty run response")

    if isinstance(run, dict):
        dataset_id = run.get("default_dataset_id") or run.get("defaultDatasetId")
    else:
        dataset_id = (
            getattr(run, "default_dataset_id", None)
            or getattr(run, "defaultDatasetId", None)
            or (run.get("default_dataset_id") if hasattr(run, "get") else None)
            or (run.get("defaultDatasetId") if hasattr(run, "get") else None)
        )

    if not dataset_id:
        raise RuntimeError(f"Apify actor '{actor_id}' run missing default dataset ID")

    dataset_client = client.dataset(dataset_id)
    items_page = await dataset_client.list_items()

    if hasattr(items_page, "items"):
        return items_page.items or []
    elif isinstance(items_page, dict):
        return items_page.get("items", []) or []
    return []


async def scrape_single_parsed_url(
    parsed_url: ParsedURL,
) -> ApifyScrapeResult:
    """
    Scrape a single ParsedURL using the corresponding provider:
    - Firecrawl for website platform
    - Apify Actors for social platforms (instagram profile + posts, facebook, linkedin)

    Handles:
    - Invalid URLs
    - Unsupported platforms
    - Missing API tokens
    - Execution errors
    - Missing dataset ID / empty datasets
    """

    # ---------------------------------------------------------
    # 1. Validate parsed URL
    # ---------------------------------------------------------
    if not parsed_url.is_valid or not parsed_url.normalized_url:
        logger.warning(
            f"Skipping scraping for invalid URL: '{parsed_url.original_url}'"
        )
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=parsed_url.error or "Invalid URL string skipped",
        )

    # ---------------------------------------------------------
    # 2. Dispatch to Firecrawl for website platform
    # ---------------------------------------------------------
    platform = parsed_url.platform
    if platform == "website":
        return await scrape_website_url(parsed_url)

    # ---------------------------------------------------------
    # 3. Check supported platform
    # ---------------------------------------------------------
    if platform not in ACTOR_MAPPING:
        logger.warning(
            f"Skipping Apify scraping for unsupported platform '{platform}' (URL: '{parsed_url.original_url}')"
        )
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=f"Platform '{platform}' is not supported by Apify scraping service",
        )

    # ---------------------------------------------------------
    # 4. Get Apify API token
    # ---------------------------------------------------------
    token = settings.apify_api_key or settings.apify_api_token
    if not token:
        logger.error("Apify API token is not configured in application settings.")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error="Apify API token is missing or not configured",
        )

    target_url = parsed_url.normalized_url
    client = ApifyClientAsync(token=token)

    # ---------------------------------------------------------
    # 5. Execution for Instagram (Profile + Posts concurrently)
    # ---------------------------------------------------------
    if platform == "instagram":
        profile_config = ACTOR_MAPPING.get("instagram_profile", {})
        post_config = ACTOR_MAPPING.get("instagram", {})

        profile_actor_id = profile_config.get("actor_id", "apify/instagram-profile-scraper")
        post_actor_id = post_config.get("actor_id", "apify/instagram-scraper")

        profile_input = profile_config["build_input"](target_url) if "build_input" in profile_config else {"usernames": [target_url]}
        post_input = post_config["build_input"](target_url)

        logger.info(f"Starting Instagram dual scraping (profile: '{profile_actor_id}', posts: '{post_actor_id}') for URL '{target_url}'")

        profile_task = _run_apify_actor(client, profile_actor_id, profile_input)
        post_task = _run_apify_actor(client, post_actor_id, post_input)

        profile_res, post_res = await asyncio.gather(profile_task, post_task, return_exceptions=True)

        combined_data: List[Dict[str, Any]] = []
        error_messages: List[str] = []

        if isinstance(profile_res, list) and profile_res:
            normalized_profile = [
                normalize_instagram_profile_item(p) for p in profile_res if isinstance(p, dict)
            ]
            if normalized_profile:
                combined_data.append({"type": "profile", "items": normalized_profile})
        elif isinstance(profile_res, Exception):
            logger.warning(f"Instagram profile scraper failed for '{target_url}': {profile_res}")
            error_messages.append(f"Profile scraper error: {str(profile_res)}")
        else:
            logger.warning(f"Instagram profile scraper returned 0 items for '{target_url}'")
            error_messages.append("Profile scraper returned empty dataset")

        if isinstance(post_res, list) and post_res:
            normalized_posts = [
                normalize_instagram_post_item(p) for p in post_res if isinstance(p, dict)
            ]
            if normalized_posts:
                combined_data.append({"type": "posts", "items": normalized_posts})
        elif isinstance(post_res, Exception):
            logger.warning(f"Instagram posts scraper failed for '{target_url}': {post_res}")
            error_messages.append(f"Posts scraper error: {str(post_res)}")
        else:
            logger.warning(f"Instagram posts scraper returned 0 items for '{target_url}'")
            error_messages.append("Posts scraper returned empty dataset")

        if combined_data:
            logger.info(f"Instagram dual scraping completed for '{target_url}'. Retrieved {len(combined_data)} data source(s).")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="instagram",
                success=True,
                data=combined_data,
                error="; ".join(error_messages) if error_messages else None,
            )
        else:
            logger.error(f"Instagram dual scraping failed for '{target_url}'. Errors: {error_messages}")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="instagram",
                success=False,
                data=[],
                error="; ".join(error_messages) if error_messages else "Instagram profile and posts scraping failed",
            )

    # ---------------------------------------------------------
    # 6. Execution for Facebook (Page + Posts concurrently)
    # ---------------------------------------------------------
    if platform == "facebook":
        page_config = ACTOR_MAPPING.get("facebook_page", {})
        post_config = ACTOR_MAPPING.get("facebook", {})

        page_actor_id = page_config.get("actor_id", "apify/facebook-pages-scraper")
        post_actor_id = post_config.get("actor_id", "apify/facebook-posts-scraper")

        page_input = (
            page_config["build_input"](target_url)
            if "build_input" in page_config
            else {"startUrls": [{"url": target_url}]}
        )
        post_input = post_config["build_input"](target_url)

        logger.info(
            f"Starting Facebook dual scraping (page: '{page_actor_id}', posts: '{post_actor_id}') for URL '{target_url}'"
        )

        page_task = _run_apify_actor(client, page_actor_id, page_input)
        post_task = _run_apify_actor(client, post_actor_id, post_input)

        page_res, post_res = await asyncio.gather(page_task, post_task, return_exceptions=True)

        combined_fb_data: List[Dict[str, Any]] = []
        fb_error_messages: List[str] = []

        if isinstance(page_res, list) and page_res:
            normalized_page = [
                normalize_facebook_page_item(p) for p in page_res if isinstance(p, dict)
            ]
            if normalized_page:
                combined_fb_data.append({"type": "page", "items": normalized_page})
        elif isinstance(page_res, Exception):
            logger.warning(f"Facebook page scraper failed for '{target_url}': {page_res}")
            fb_error_messages.append(f"Page scraper error: {str(page_res)}")
        else:
            logger.warning(f"Facebook page scraper returned 0 items for '{target_url}'")
            fb_error_messages.append("Page scraper returned empty dataset")

        if isinstance(post_res, list) and post_res:
            normalized_fb_posts = [
                normalize_facebook_post_item(p) for p in post_res if isinstance(p, dict)
            ]
            if normalized_fb_posts:
                combined_fb_data.append({"type": "posts", "items": normalized_fb_posts})
        elif isinstance(post_res, Exception):
            logger.warning(f"Facebook posts scraper failed for '{target_url}': {post_res}")
            fb_error_messages.append(f"Posts scraper error: {str(post_res)}")
        else:
            logger.warning(f"Facebook posts scraper returned 0 items for '{target_url}'")
            fb_error_messages.append("Posts scraper returned empty dataset")

        if combined_fb_data:
            logger.info(
                f"Facebook dual scraping completed for '{target_url}'. Retrieved {len(combined_fb_data)} data source(s)."
            )
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="facebook",
                success=True,
                data=combined_fb_data,
                error="; ".join(fb_error_messages) if fb_error_messages else None,
            )
        else:
            logger.error(f"Facebook dual scraping failed for '{target_url}'. Errors: {fb_error_messages}")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="facebook",
                success=False,
                data=[],
                error="; ".join(fb_error_messages) if fb_error_messages else "Facebook page and posts scraping failed",
            )

    # ---------------------------------------------------------
    # 6. Execution for other platforms (Facebook, LinkedIn, etc.)
    # ---------------------------------------------------------
    actor_info = ACTOR_MAPPING[platform]
    actor_id = actor_info["actor_id"]
    run_input = actor_info["build_input"](target_url)

    logger.info(f"Starting Apify actor '{actor_id}' for platform '{platform}' on URL '{target_url}'")

    try:
        items = await _run_apify_actor(client, actor_id, run_input)

        if not items:
            logger.warning(f"Apify actor '{actor_id}' finished but returned 0 items for URL '{target_url}'.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=platform,
                success=False,
                data=[],
                error="Apify Actor execution returned empty dataset",
            )

        logger.info(f"Successfully scraped {len(items)} items from Apify actor '{actor_id}' for URL '{target_url}'")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=platform,
            success=True,
            data=items,
            error=None,
        )

    except Exception as e:
        logger.error(f"Apify execution failed for URL '{target_url}' on actor '{actor_id}': {e}")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=platform,
            success=False,
            data=None,
            error=f"Apify execution error: {str(e)}",
        )


async def scrape_parsed_urls(
    parsed_urls: List[ParsedURL],
    on_url_complete: Optional[Callable[[ParsedURL, Any], None]] = None,
) -> List[ApifyScrapeResult]:
    """
    Scrape multiple ParsedURL objects concurrently.

    asyncio.gather() allows independent Apify and Firecrawl scraping tasks
    to run concurrently instead of waiting for each URL one by one.

    The returned results preserve the same order as parsed_urls.
    """

    # ---------------------------------------------------------
    # 1. Nothing to scrape
    # ---------------------------------------------------------
    if not parsed_urls:
        return []

    valid_urls = [u for u in parsed_urls if u.is_valid]
    has_website = any(u.platform == "website" for u in valid_urls)
    has_social = any(u.platform != "website" for u in valid_urls)

    if has_website and has_social:
        emit_progress(
            "Reviewing your online presence...",
            "Analyzing your website and social media profiles in parallel.",
        )
    elif has_website:
        emit_progress(
            "Reviewing your website...",
            "Analyzing your website content with Firecrawl.",
        )
    elif has_social:
        emit_progress(
            "Reviewing your social media presence...",
            "Gathering insights from your social media profiles.",
        )

    # Track pending platforms to report accurate status as parallel operations finish
    pending_platforms: List[str] = [u.platform for u in valid_urls]

    async def _wrapped_scrape(parsed_url: ParsedURL):
        try:
            res = await scrape_single_parsed_url(parsed_url)
        except Exception as exc:
            res = exc

        if on_url_complete:
            try:
                on_url_complete(parsed_url, res)
            except Exception:
                pass

        try:
            if parsed_url.platform in pending_platforms:
                pending_platforms.remove(parsed_url.platform)

            rem_web = any(p == "website" for p in pending_platforms)
            rem_soc = any(p != "website" for p in pending_platforms)

            if rem_soc and not rem_web and has_website:
                emit_progress(
                    "Reviewing your social media presence...",
                    "Website analysis complete. Continuing social media review.",
                )
            elif rem_web and not rem_soc and has_social:
                emit_progress(
                    "Reviewing your website...",
                    "Social media analysis complete. Continuing website review.",
                )
        except Exception:
            pass

        return res

    # ---------------------------------------------------------
    # 2. Run all URL scraping tasks concurrently
    # ---------------------------------------------------------
    results_or_exceptions = await asyncio.gather(
        *(_wrapped_scrape(parsed_url) for parsed_url in parsed_urls),
        return_exceptions=True,
    )

    # ---------------------------------------------------------
    # 3. Normalize results
    # ---------------------------------------------------------
    clean_results: List[ApifyScrapeResult] = []

    for parsed_url, result in zip(
        parsed_urls,
        results_or_exceptions,
    ):

        # Unexpected exception outside the normal
        # error handling inside scrape_single_parsed_url()
        if isinstance(result, Exception):
            logger.error(
                f"Unhandled exception during scraping for URL "
                f"'{parsed_url.original_url}': {result}"
            )

            clean_results.append(
                ApifyScrapeResult(
                    original_url=parsed_url.original_url,
                    normalized_url=parsed_url.normalized_url,
                    platform=parsed_url.platform,
                    success=False,
                    data=None,
                    error=(
                        "Unhandled exception during async scrape: "
                        f"{str(result)}"
                    ),
                )
            )

        elif isinstance(result, ApifyScrapeResult):
            clean_results.append(result)

    return clean_results