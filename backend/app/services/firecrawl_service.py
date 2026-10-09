import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from firecrawl import Firecrawl

from backend.app.core.config import settings
from backend.app.models.parsed_url import ParsedURL
from backend.app.models.scraped_result import ApifyScrapeResult

logger = logging.getLogger(__name__)

# Default crawl page limit
DEFAULT_CRAWL_LIMIT = 2

# Tracking query parameters to strip
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "ref", "source", "_ga", "mc_cid", "mc_eid"
}

# Asset file extensions to exclude
EXCLUDED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".css", ".js",
    ".pdf", ".zip", ".xml", ".json", ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".avi", ".mov"
)

# Irrelevant URL path keywords (login, cart, legal, tech admin, etc.)
EXCLUDED_PATH_PATTERNS = re.compile(
    r'/(?:login|logout|signin|signup|register|auth|account|profile|my-account|password|reset|forgot|'
    r'cart|checkout|basket|bag|payment|order-status|order-confirmation|billing|'
    r'privacy|privacy-policy|terms|terms-of-service|terms-and-conditions|legal|cookie-policy|gdpr|disclaimer|'
    r'wp-admin|wp-includes|cdn-cgi|xmlrpc\.php|ssrf-check)(?:/|[\?#]|$)',
    re.IGNORECASE
)

# Keyword groups & weights for marketing relevance
KEYWORD_GROUPS = {
    "products_services": (
        ["product", "products", "service", "services", "solution", "solutions", "feature", "features", "offering", "offerings", "platform", "tool", "tools", "capabilities"],
        10.0
    ),
    "pricing": (
        ["price", "pricing", "plan", "plans", "cost", "tier", "tiers", "subscription", "package", "packages", "rate", "rates"],
        9.0
    ),
    "company_about": (
        ["about", "company", "story", "team", "mission", "vision", "who-we-are", "our-story", "leadership", "culture", "overview"],
        8.0
    ),
    "social_proof_use_cases": (
        ["case-study", "case-studies", "testimonial", "testimonials", "portfolio", "customer", "customers", "client", "clients", "use-case", "use-cases", "industry", "industries", "review", "reviews", "result", "results", "work"],
        8.0
    ),
    "conversion_contact": (
        ["contact", "contact-us", "faq", "faqs", "help", "get-started", "demo", "request-demo", "book-call"],
        5.0
    ),
}

# Negative keyword penalty for low-signal secondary content
NEGATIVE_KEYWORDS = (
    "blog", "article", "news", "press", "media", "tag", "category",
    "archive", "author", "page/", "feed", "comments", "event", "webinar", "podcast"
)


def normalize_and_clean_url(url_str: str) -> Optional[str]:
    """
    Normalizes a URL string:
    - Strips whitespace & trailing fragments
    - Validates scheme (http/https) and hostname
    - Filters out static asset extensions
    - Strips tracking query parameters while preserving benign parameters
    - Normalizes trailing slashes
    """
    if not url_str or not isinstance(url_str, str):
        return None
    cleaned = url_str.strip()
    if not cleaned:
        return None

    try:
        parsed = urlparse(cleaned)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https") or not parsed.hostname:
            return None

        hostname = parsed.hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]

        netloc = hostname
        if parsed.port and not (
            (scheme == "http" and parsed.port == 80) or
            (scheme == "https" and parsed.port == 443)
        ):
            netloc = f"{hostname}:{parsed.port}"

        path = parsed.path or "/"
        if any(path.lower().endswith(ext) for ext in EXCLUDED_EXTENSIONS):
            return None

        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        filtered_query = [(k, v) for k, v in query_pairs if k.lower() not in TRACKING_PARAMS]
        query = urlencode(filtered_query) if filtered_query else ""

        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return None


def get_domain_from_url(url_str: str) -> Optional[str]:
    """Extracts clean domain name (without www)."""
    try:
        parsed = urlparse(url_str.strip())
        if not parsed.hostname:
            return None
        host = parsed.hostname.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


def is_url_excluded(url_str: str) -> bool:
    """Checks if URL path contains excluded patterns (login, cart, legal, tech admin, etc.)."""
    return bool(EXCLUDED_PATH_PATTERNS.search(url_str))


def calculate_url_relevance_score(
    url_str: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    is_homepage: bool = False
) -> float:
    """
    Calculates a deterministic marketing relevance score for a URL candidate.
    - Homepage receives highest priority boost (+100)
    - Signals from URL path, title, and description are evaluated against marketing keyword groups
    - Low-signal secondary content (blog, tags, archives) receives a penalty
    """
    if is_homepage:
        return 100.0

    score = 1.0
    text_corpus = f"{url_str} {title or ''} {description or ''}".lower()

    for _, (keywords, weight) in KEYWORD_GROUPS.items():
        for kw in keywords:
            if kw in text_corpus:
                score += weight

    for neg_kw in NEGATIVE_KEYWORDS:
        if neg_kw in url_str.lower():
            score -= 3.0

    parsed = urlparse(url_str)
    path_segments = [s for s in parsed.path.split("/") if s]
    if len(path_segments) > 3:
        score -= 0.5 * (len(path_segments) - 3)

    return score


def select_top_urls_for_crawling(
    target_url: str,
    discovered_candidates: List[Dict[str, Any]],
    limit: int = DEFAULT_CRAWL_LIMIT,
) -> List[str]:
    """
    Filters discovered candidates to the target domain, removes excluded URLs,
    scores candidates by marketing relevance, and selects the top `limit` URLs.
    Ensures target_url (homepage) is included when valid.
    """
    clean_target = normalize_and_clean_url(target_url) or target_url
    target_domain = get_domain_from_url(clean_target)

    seen_urls = set()
    scored_candidates = []

    for candidate in discovered_candidates:
        raw_url = candidate.get("url") if isinstance(candidate, dict) else str(candidate)
        norm_url = normalize_and_clean_url(raw_url)
        if not norm_url or norm_url in seen_urls:
            continue

        cand_domain = get_domain_from_url(norm_url)
        if target_domain and cand_domain != target_domain:
            continue

        if is_url_excluded(norm_url):
            continue

        seen_urls.add(norm_url)

        title = candidate.get("title") if isinstance(candidate, dict) else ""
        desc = candidate.get("description") if isinstance(candidate, dict) else ""
        parsed = urlparse(norm_url)
        is_hp = (parsed.path in ("", "/"))

        score = calculate_url_relevance_score(norm_url, title=title, description=desc, is_homepage=is_hp)
        scored_candidates.append((norm_url, score, is_hp))

    if clean_target not in seen_urls and not is_url_excluded(clean_target):
        parsed = urlparse(clean_target)
        is_hp = (parsed.path in ("", "/"))
        score = calculate_url_relevance_score(clean_target, is_homepage=is_hp)
        scored_candidates.append((clean_target, score, is_hp))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    selected = [url for url, score, is_hp in scored_candidates[:limit]]

    # Ensure homepage is present in selected list if available
    if clean_target not in selected and selected:
        selected[-1] = clean_target

    return selected or [clean_target]


async def scrape_website_url(
    parsed_url: ParsedURL,
    limit: int = DEFAULT_CRAWL_LIMIT,
) -> ApifyScrapeResult:
    """
    Scrape website URLs using Firecrawl Python SDK:
    1. Uses Map API to discover domain URLs and metadata
    2. Filters irrelevant/external/tracking URLs in Python
    3. Ranks remaining URLs using deterministic marketing relevance scoring
    4. Batch scrapes only the top selected URLs (up to limit)

    Handles:
    - Invalid URLs
    - Missing Firecrawl API key
    - Map/Batch API execution errors
    - Empty crawl results
    """

    # 1. Validate parsed URL
    if not parsed_url.is_valid or not parsed_url.normalized_url:
        logger.warning(
            f"Skipping Firecrawl scraping for invalid website URL: '{parsed_url.original_url}'"
        )
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform="website",
            success=False,
            data=None,
            error=parsed_url.error or "Invalid URL string skipped",
        )

    # 2. Check Firecrawl API key configuration
    api_key = settings.firecrawl_api_key
    if not api_key:
        logger.error("Firecrawl API key is missing or not configured in settings.")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform="website",
            success=False,
            data=None,
            error="Firecrawl API key is missing or not configured",
        )

    target_url = parsed_url.normalized_url
    logger.info(f"Starting Firecrawl discovery and scraping for website URL '{target_url}' (limit={limit})")

    # 3. Synchronous discovery & batch scraping operation wrapped in thread pool
    def _run_firecrawl_process() -> Any:
        app = Firecrawl(api_key=api_key)

        # Step 3a: Discover website URLs via Map API
        discovered_links: List[Dict[str, Any]] = []
        try:
            logger.info(f"Mapping website URLs via Firecrawl Map API for '{target_url}'")
            map_data = app.map(target_url)
            links = getattr(map_data, "links", None) or getattr(map_data, "data", None) or []
            for link_item in links:
                if hasattr(link_item, "model_dump"):
                    discovered_links.append(link_item.model_dump())
                elif hasattr(link_item, "url"):
                    discovered_links.append({
                        "url": getattr(link_item, "url"),
                        "title": getattr(link_item, "title", None),
                        "description": getattr(link_item, "description", None),
                    })
                elif isinstance(link_item, dict) and "url" in link_item:
                    discovered_links.append(link_item)
                elif isinstance(link_item, str):
                    discovered_links.append({"url": link_item})
        except Exception as map_err:
            logger.warning(f"Firecrawl Map API call failed for '{target_url}': {map_err}. Falling back to target URL.")

        # Step 3b: Filter, rank, and select top relevant URLs in Python
        selected_urls = select_top_urls_for_crawling(
            target_url=target_url,
            discovered_candidates=discovered_links,
            limit=limit,
        )
        logger.info(f"Selected {len(selected_urls)} relevant URL(s) for scraping: {selected_urls}")

        # Step 3c: Batch scrape selected URLs
        try:
            return app.batch_scrape(selected_urls, formats=["markdown"])
        except Exception as batch_err:
            logger.warning(f"Firecrawl batch_scrape failed for URLs {selected_urls}: {batch_err}. Falling back to crawl.")
            return app.crawl(
                url=target_url,
                limit=limit,
                formats=["markdown"],
                deduplicate_similar_urls=True,
            )

    try:
        job_result = await asyncio.to_thread(_run_firecrawl_process)

        if not job_result or not hasattr(job_result, "data") or job_result.data is None:
            logger.warning(f"Firecrawl returned no data for URL '{target_url}'.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="website",
                success=False,
                data=[],
                error="Firecrawl execution returned no data",
            )

        # 4. Normalize returned documents into structured dataset items
        scraped_pages: List[Dict[str, Any]] = []

        for doc in job_result.data:
            if hasattr(doc, "model_dump"):
                doc_dict = doc.model_dump()
            elif isinstance(doc, dict):
                doc_dict = doc
            else:
                doc_dict = {}

            metadata = doc_dict.get("metadata") or {}
            if hasattr(metadata, "model_dump"):
                metadata = metadata.model_dump()
            elif not isinstance(metadata, dict):
                metadata = {}

            page_url = metadata.get("url") or metadata.get("source_url") or doc_dict.get("url") or target_url
            page_title = metadata.get("title") or doc_dict.get("title") or ""
            page_desc = metadata.get("description") or doc_dict.get("description") or ""
            markdown_content = doc_dict.get("markdown") or doc_dict.get("text") or ""

            page_item = {
                "url": page_url,
                "title": page_title,
                "text": markdown_content,
                "markdown": markdown_content,
                "description": page_desc,
                "metadata": metadata,
            }
            scraped_pages.append(page_item)

        if not scraped_pages:
            logger.warning(f"Firecrawl returned empty page list for URL '{target_url}'.")
            return ApifyScrapeResult(
                original_url=parsed_url.original_url,
                normalized_url=parsed_url.normalized_url,
                platform="website",
                success=False,
                data=[],
                error="Firecrawl execution returned 0 scraped pages",
            )

        logger.info(
            f"Successfully scraped {len(scraped_pages)} page(s) via Firecrawl for website '{target_url}'"
        )

        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform="website",
            success=True,
            data=scraped_pages,
            error=None,
        )

    except Exception as e:
        logger.error(f"Firecrawl execution failed for URL '{target_url}': {e}")
        return ApifyScrapeResult(
            original_url=parsed_url.original_url,
            normalized_url=parsed_url.normalized_url,
            platform="website",
            success=False,
            data=None,
            error=f"Firecrawl execution error: {str(e)}",
        )
