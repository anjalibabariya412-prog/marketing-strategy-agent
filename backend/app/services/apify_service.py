import asyncio
import logging
from typing import Any, Dict, List

from apify_client import ApifyClientAsync

from backend.app.core.config import settings
from backend.app.models.parsed_url import ParsedURL
from backend.app.models.scraped_result import ApifyScrapeResult
from backend.app.services.firecrawl_service import scrape_website_url


logger = logging.getLogger(__name__)


ACTOR_MAPPING: Dict[str, Dict[str, Any]] = {
    "instagram": {
        "actor_id": "apify/instagram-scraper",
        "build_input": lambda url: {
            "directUrls": [url],
            "resultsLimit": 5,
        },
    },
    "facebook": {
        "actor_id": "apify/facebook-posts-scraper",
        "build_input": lambda url: {
            "startUrls": [{"url": url}],
            "maxPosts": 5,
        },
    },
    "linkedin": {
        "actor_id": "apify/linkedin-post-scraper",
        "build_input": lambda url: {
            "urls": [url],
            "deepScrape": False,
        },
    },
}


async def scrape_single_parsed_url(
    parsed_url: ParsedURL,
) -> ApifyScrapeResult:
    """
    Scrape a single ParsedURL using the corresponding provider:
    - Firecrawl for website platform
    - Apify Actors for social platforms (instagram, facebook, linkedin)

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
            f"Skipping scraping for invalid URL: "
            f"'{parsed_url.original_url}'"
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
    # 2. Check supported platform
    # ---------------------------------------------------------
    platform = parsed_url.platform

    if platform not in ACTOR_MAPPING:
        logger.warning(
            f"Skipping Apify scraping for unsupported platform "
            f"'{platform}' (URL: '{parsed_url.original_url}')"
        )

        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=(
                f"Platform '{platform}' is not supported "
                f"by Apify scraping service"
            ),
        )

    # ---------------------------------------------------------
    # 3. Get Apify API token
    # ---------------------------------------------------------
    token = settings.apify_api_key or settings.apify_api_token

    if not token:
        logger.error(
            "Apify API token is not configured in application settings."
        )

        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error="Apify API token is missing or not configured",
        )

    # ---------------------------------------------------------
    # 4. Get Actor configuration
    # ---------------------------------------------------------
    actor_info = ACTOR_MAPPING[platform]

    actor_id = actor_info["actor_id"]
    target_url = parsed_url.normalized_url
    run_input = actor_info["build_input"](target_url)

    logger.info(
        f"Starting Apify actor '{actor_id}' "
        f"for platform '{platform}' "
        f"on URL '{target_url}'"
    )

    try:
        # -----------------------------------------------------
        # 5. Create Apify client
        # -----------------------------------------------------
        client = ApifyClientAsync(token=token)

        # -----------------------------------------------------
        # 6. Start Actor and wait for completion
        # -----------------------------------------------------
        run = await client.actor(actor_id).call(
            run_input=run_input
        )

        if not run:
            logger.error(
                f"Apify actor '{actor_id}' returned None response."
            )

            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=None,
                error="Apify Actor returned empty run response",
            )

        # -----------------------------------------------------
        # 7. Get dataset ID
        #
        # IMPORTANT:
        # `run` is an Apify Run object, not a dictionary.
        # Therefore, do NOT use run.get(...)
        # -----------------------------------------------------
        if isinstance(run, dict):
            dataset_id = run.get("default_dataset_id") or run.get("defaultDatasetId")
        else:
            dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None) or (run.get("default_dataset_id") if hasattr(run, "get") else None) or (run.get("defaultDatasetId") if hasattr(run, "get") else None)

        if not dataset_id:
            logger.error(
                f"Apify actor '{actor_id}' run missing "
                f"default dataset ID."
            )

            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=None,
                error=(
                    "Apify Actor run response missing "
                    "default dataset ID"
                ),
            )

        logger.info(
            f"Apify actor '{actor_id}' completed successfully. "
            f"Dataset ID: '{dataset_id}'"
        )

        # -----------------------------------------------------
        # 8. Fetch dataset items
        # -----------------------------------------------------
        dataset_client = client.dataset(dataset_id)

        items_page = await dataset_client.list_items()

        # Apify SDK normally returns an object with `.items`.
        # Keep dictionary fallback for compatibility.
        if hasattr(items_page, "items"):
            items = items_page.items
        elif isinstance(items_page, dict):
            items = items_page.get("items", [])
        else:
            items = []

        # -----------------------------------------------------
        # 9. Handle empty dataset
        # -----------------------------------------------------
        if not items:
            logger.warning(
                f"Apify actor '{actor_id}' finished but "
                f"default dataset '{dataset_id}' returned 0 items."
            )

            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform=parsed_url.platform,
                success=False,
                data=[],
                error="Apify Actor execution returned empty dataset",
            )

        # -----------------------------------------------------
        # 10. Successful result
        # -----------------------------------------------------
        logger.info(
            f"Successfully scraped {len(items)} items "
            f"from Apify actor '{actor_id}' "
            f"for URL '{target_url}'"
        )

        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=True,
            data=items,
            error=None,
        )

    except Exception as e:
        # -----------------------------------------------------
        # 11. Graceful Apify error handling
        # -----------------------------------------------------
        logger.error(
            f"Apify execution failed for URL '{target_url}' "
            f"on actor '{actor_id}': {e}"
        )

        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform=parsed_url.platform,
            success=False,
            data=None,
            error=f"Apify execution error: {str(e)}",
        )


async def scrape_parsed_urls(
    parsed_urls: List[ParsedURL],
) -> List[ApifyScrapeResult]:
    """
    Scrape multiple ParsedURL objects concurrently.

    asyncio.gather() allows independent Apify scraping tasks
    to run concurrently instead of waiting for each URL
    one by one.

    The returned results preserve the same order as parsed_urls.
    """

    # ---------------------------------------------------------
    # 1. Nothing to scrape
    # ---------------------------------------------------------
    if not parsed_urls:
        return []

    # ---------------------------------------------------------
    # 2. Run all URL scraping tasks concurrently
    # ---------------------------------------------------------
    results_or_exceptions = await asyncio.gather(
        *(
            scrape_single_parsed_url(parsed_url)
            for parsed_url in parsed_urls
        ),
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