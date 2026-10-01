import logging
from typing import Any, Dict, List, Optional
from apify_client import ApifyClientAsync

from backend.app.core.config import settings
from backend.app.models.parsed_url import ParsedURL
from backend.app.models.scraped_result import ApifyScrapeResult

logger = logging.getLogger(__name__)

ACTOR_MAPPING: Dict[str, Dict[str, Any]] = {
    "website": {
        "actor_id": "apify/website-content-crawler",
        "build_input": lambda url: {"startUrls": [{"url": url}], "maxCrawlPages": 3},
    },
    "instagram": {
        "actor_id": "apify/instagram-scraper",
        "build_input": lambda url: {"directUrls": [url], "resultsLimit": 5},
    },
    "facebook": {
        "actor_id": "apify/facebook-posts-scraper",
        "build_input": lambda url: {"startUrls": [{"url": url}], "maxPosts": 5},
    },
    "linkedin": {
        "actor_id": "apify/linkedin-post-scraper",
        "build_input": lambda url: {"urls": [url], "deepScrape": False},
    },
}


async def scrape_single_parsed_url(parsed_url: ParsedURL) -> ApifyScrapeResult:
    """
    Scrapes a single ParsedURL using the corresponding Apify Actor based on platform.
    Handles invalid URLs, unconfigured API keys, network/API errors, and empty datasets gracefully.
    """
    if not parsed_url.is_valid or not parsed_url.normalized_url:
        logger.warning(f"Skipping Apify scraping for invalid URL: '{parsed_url.original_url}'")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=parsed_url.error or "Invalid URL string skipped"
        )

    platform = parsed_url.platform
    if platform not in ACTOR_MAPPING:
        logger.warning(f"Skipping Apify scraping for unsupported platform '{platform}' (URL: '{parsed_url.original_url}')")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=f"Platform '{platform}' is not supported by Apify scraping service"
        )

    token = settings.apify_api_key or settings.apify_api_token
    if not token:
        logger.error("Apify API token is not configured in application settings.")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error="Apify API token is missing or not configured"
        )

    actor_info = ACTOR_MAPPING[platform]
    actor_id = actor_info["actor_id"]
    target_url = parsed_url.normalized_url
    run_input = actor_info["build_input"](target_url)

    logger.info(f"Starting Apify actor '{actor_id}' for platform '{platform}' on URL '{target_url}'")

    try:
        client = ApifyClientAsync(token=token)
        run = await client.actor(actor_id).call(run_input=run_input)

        if not run:
            logger.error(f"Apify actor '{actor_id}' returned None response.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=None,
                error="Apify Actor returned empty run response"
            )

        dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
        if not dataset_id:
            logger.error(f"Apify actor '{actor_id}' run missing default dataset ID.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=None,
                error="Apify Actor run response missing default dataset ID"
            )

        dataset_client = client.dataset(dataset_id)
        items_page = await dataset_client.list_items()
        items = items_page.items if hasattr(items_page, "items") else (items_page.get("items") if isinstance(items_page, dict) else [])

        if not items:
            logger.warning(f"Apify actor '{actor_id}' finished but default dataset '{dataset_id}' returned 0 items.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=[],
                error="Apify Actor execution returned empty dataset"
            )

        logger.info(f"Successfully scraped {len(items)} items from Apify actor '{actor_id}' for URL '{target_url}'")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=True,
            data=items,
            error=None
        )

    except Exception as e:
        logger.error(f"Apify execution failed for URL '{target_url}' on actor '{actor_id}': {e}")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=f"Apify execution error: {str(e)}"
        )


async def scrape_parsed_urls(parsed_urls: List[ParsedURL]) -> List[ApifyScrapeResult]:
    """
    Processes multiple ParsedURL objects sequentially or concurrently and returns a list of ApifyScrapeResult objects.
    Ensures failure of one URL does not crash or block processing for other URLs.
    """
    if not parsed_urls:
        return []

    results: List[ApifyScrapeResult] = []
    for parsed_url in parsed_urls:
        res = await scrape_single_parsed_url(parsed_url)
        results.append(res)

    return results
