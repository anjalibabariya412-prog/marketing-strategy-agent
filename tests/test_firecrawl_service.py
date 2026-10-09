import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.parsed_url import ParsedURL
from backend.app.models.scraped_result import ApifyScrapeResult
from backend.app.services.apify_service import scrape_single_parsed_url, scrape_parsed_urls
from backend.app.services.firecrawl_service import (
    scrape_website_url,
    normalize_and_clean_url,
    is_url_excluded,
    calculate_url_relevance_score,
    select_top_urls_for_crawling,
)
from backend.app.services.online_presence_processor import _format_scraped_data_for_prompt


def run_async(coro):
    return asyncio.run(coro)


def test_firecrawl_filtering_and_scoring():
    print("=== Testing Firecrawl URL Filtering, Relevance Scoring & Selection ===")

    # 1. Test URL Normalization & Cleaning
    url_tracking = "https://example.com/products/pricing?utm_source=twitter&utm_medium=cpc&gclid=123"
    norm_tracking = normalize_and_clean_url(url_tracking)
    assert norm_tracking == "https://example.com/products/pricing"

    # Static asset exclusion
    assert normalize_and_clean_url("https://example.com/images/hero.png") is None
    assert normalize_and_clean_url("https://example.com/docs/manual.pdf") is None

    print("✓ Test Unit 1 Passed: URL normalization, tracking parameter stripping, and asset filtering")

    # 2. Test URL Exclusion Patterns
    assert is_url_excluded("https://example.com/login") is True
    assert is_url_excluded("https://example.com/cart") is True
    assert is_url_excluded("https://example.com/privacy-policy") is True
    assert is_url_excluded("https://example.com/checkout") is True
    assert is_url_excluded("https://example.com/services/marketing") is False
    assert is_url_excluded("https://example.com/about-us") is False

    print("✓ Test Unit 2 Passed: Irrelevant URL path exclusion (auth, cart, legal)")

    # 3. Test Marketing Relevance Scoring
    score_hp = calculate_url_relevance_score("https://example.com/", is_homepage=True)
    score_products = calculate_url_relevance_score("https://example.com/services/pricing-plans", title="Services & Pricing")
    score_about = calculate_url_relevance_score("https://example.com/about-us", title="About Our Company")
    score_blog = calculate_url_relevance_score("https://example.com/blog/2023/10/article", title="Blog Article")

    assert score_hp == 100.0
    assert score_products > score_about
    assert score_about > score_blog
    print("✓ Test Unit 3 Passed: Marketing relevance scoring (products > about > blog)")

    # 4. Test Selection & Domain Scoping
    candidates = [
        {"url": "https://example.com/", "title": "Home"},
        {"url": "https://example.com/login", "title": "Login"},
        {"url": "https://example.com/services/digital-marketing", "title": "Our Services"},
        {"url": "https://example.com/pricing", "title": "Plans & Pricing"},
        {"url": "https://example.com/about", "title": "About Us"},
        {"url": "https://example.com/blog/post-1", "title": "Blog Post"},
        {"url": "https://external-domain.com/page", "title": "External Page"},
    ]

    selected = select_top_urls_for_crawling("https://example.com", candidates, limit=5)
    assert "https://example.com" in selected or "https://example.com/" in selected
    assert "https://example.com/services/digital-marketing" in selected
    assert "https://example.com/pricing" in selected
    assert "https://example.com/login" not in selected
    assert "https://external-domain.com/page" not in selected
    assert len(selected) <= 5

    print("✓ Test Unit 4 Passed: Top URL selection, domain scoping, and homepage preservation")


def test_firecrawl_and_routing():
    print("=== Testing Firecrawl Website Scraping & Social Media Apify Routing ===")

    website_url = ParsedURL(
        original_url="https://example.com",
        normalized_url="https://example.com",
        platform="website",
        is_valid=True,
    )
    instagram_url = ParsedURL(
        original_url="https://instagram.com/brand",
        normalized_url="https://instagram.com/brand",
        platform="instagram",
        is_valid=True,
    )
    facebook_url = ParsedURL(
        original_url="https://facebook.com/brand",
        normalized_url="https://facebook.com/brand",
        platform="facebook",
        is_valid=True,
    )
    linkedin_url = ParsedURL(
        original_url="https://linkedin.com/company/brand",
        normalized_url="https://linkedin.com/company/brand",
        platform="linkedin",
        is_valid=True,
    )

    with patch("backend.app.services.apify_service.settings") as mock_apify_settings, \
         patch("backend.app.services.firecrawl_service.settings") as mock_firecrawl_settings:

        mock_apify_settings.apify_api_key = "mock_apify_token_xyz"
        mock_apify_settings.apify_api_token = "mock_apify_token_xyz"
        mock_firecrawl_settings.firecrawl_api_key = "mock_firecrawl_key_123"

        # -------------------------------------------------------------
        # Test 1: Website URL is routed to Firecrawl Map & Batch Scrape
        # -------------------------------------------------------------
        with patch("backend.app.services.firecrawl_service.Firecrawl") as mock_firecrawl_cls, \
             patch("backend.app.services.apify_service.ApifyClientAsync") as mock_apify_cls:

            mock_firecrawl_inst = MagicMock()
            mock_firecrawl_cls.return_value = mock_firecrawl_inst

            mock_map_data = MagicMock()
            link1 = MagicMock()
            link1.model_dump.return_value = {"url": "https://example.com/", "title": "Home"}
            link2 = MagicMock()
            link2.model_dump.return_value = {"url": "https://example.com/pricing", "title": "Pricing"}
            mock_map_data.links = [link1, link2]
            mock_firecrawl_inst.map.return_value = mock_map_data

            doc1 = MagicMock()
            doc1.model_dump.return_value = {
                "markdown": "# Welcome to Example",
                "metadata": {"url": "https://example.com/", "title": "Home", "description": "Home page"}
            }
            doc2 = MagicMock()
            doc2.model_dump.return_value = {
                "markdown": "# Pricing Plans",
                "metadata": {"url": "https://example.com/pricing", "title": "Pricing", "description": "Pricing"}
            }
            mock_batch_job = MagicMock()
            mock_batch_job.data = [doc1, doc2]
            mock_firecrawl_inst.batch_scrape.return_value = mock_batch_job

            res = run_async(scrape_single_parsed_url(website_url))

            mock_firecrawl_cls.assert_called_with(api_key="mock_firecrawl_key_123")
            mock_firecrawl_inst.map.assert_called_once_with("https://example.com")
            mock_firecrawl_inst.batch_scrape.assert_called_once_with(
                ["https://example.com/", "https://example.com/pricing"],
                formats=["markdown"],
            )
            mock_apify_cls.assert_not_called()
            assert res.success is True
            assert res.platform == "website"
            assert len(res.data) == 2
            print("✓ Test 1 Passed: Website URL routed to Map API and Batch Scrape (Apify Client NOT called)")

        # -------------------------------------------------------------
        # Test 2: Instagram URLs continue to use Apify
        # -------------------------------------------------------------
        with patch("backend.app.services.firecrawl_service.Firecrawl") as mock_firecrawl_cls, \
             patch("backend.app.services.apify_service.ApifyClientAsync") as mock_apify_cls:

            mock_apify_inst = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_apify_cls.return_value = mock_apify_inst
            mock_apify_inst.actor.return_value = mock_actor
            mock_apify_inst.dataset.return_value = mock_dataset
            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_insta"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"caption": "Post 1"}]))

            res = run_async(scrape_single_parsed_url(instagram_url))

            mock_apify_inst.actor.assert_called_with("apify/instagram-scraper")
            mock_firecrawl_cls.assert_not_called()
            assert res.success is True
            assert res.platform == "instagram"
            print("✓ Test 2 Passed: Instagram URL uses Apify Actor and Firecrawl was NOT called")

        # -------------------------------------------------------------
        # Test 3: Facebook URLs continue to use Apify
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_apify_cls:
            mock_apify_inst = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_apify_cls.return_value = mock_apify_inst
            mock_apify_inst.actor.return_value = mock_actor
            mock_apify_inst.dataset.return_value = mock_dataset
            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_fb"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"post_text": "FB Post"}]))

            res = run_async(scrape_single_parsed_url(facebook_url))

            mock_apify_inst.actor.assert_called_with("apify/facebook-posts-scraper")
            assert res.success is True
            assert res.platform == "facebook"
            print("✓ Test 3 Passed: Facebook URL uses Apify Actor")

        # -------------------------------------------------------------
        # Test 4: LinkedIn URLs continue to use Apify
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_apify_cls:
            mock_apify_inst = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_apify_cls.return_value = mock_apify_inst
            mock_apify_inst.actor.return_value = mock_actor
            mock_apify_inst.dataset.return_value = mock_dataset
            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_li"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"text": "LI Post"}]))

            res = run_async(scrape_single_parsed_url(linkedin_url))

            mock_apify_inst.actor.assert_called_with("apify/linkedin-post-scraper")
            assert res.success is True
            assert res.platform == "linkedin"
            print("✓ Test 4 Passed: LinkedIn URL uses Apify Actor")

        # -------------------------------------------------------------
        # Test 5: Website Firecrawl results match expected downstream contract
        # -------------------------------------------------------------
        with patch("backend.app.services.firecrawl_service.Firecrawl") as mock_firecrawl_cls:
            mock_firecrawl_inst = MagicMock()
            mock_firecrawl_cls.return_value = mock_firecrawl_inst
            mock_firecrawl_inst.map.return_value = MagicMock(links=[{"url": "https://example.com/products", "title": "Products"}])

            doc = MagicMock()
            doc.model_dump.return_value = {
                "markdown": "# Main Product Page",
                "metadata": {"url": "https://example.com/products", "title": "Products", "description": "Our product line"}
            }
            mock_firecrawl_inst.batch_scrape.return_value = MagicMock(data=[doc])

            res = run_async(scrape_website_url(website_url))
            assert isinstance(res, ApifyScrapeResult)
            assert res.original_url == "https://example.com"
            assert res.normalized_url == "https://example.com"
            assert res.platform == "website"
            assert res.success is True
            assert isinstance(res.data, list)
            item = res.data[0]
            assert item["url"] == "https://example.com/products"
            assert item["title"] == "Products"
            assert item["text"] == "# Main Product Page"
            assert item["markdown"] == "# Main Product Page"
            assert item["description"] == "Our product line"

            formatted_prompt = _format_scraped_data_for_prompt([res])
            assert "--- PLATFORM: WEBSITE ---" in formatted_prompt
            assert "https://example.com/products" in formatted_prompt
            print("✓ Test 5 Passed: Firecrawl result matches downstream contract and formats cleanly for LLM prompt")

        # -------------------------------------------------------------
        # Test 6: Missing Firecrawl API key handled gracefully
        # -------------------------------------------------------------
        mock_firecrawl_settings.firecrawl_api_key = None
        res_nokey = run_async(scrape_website_url(website_url))
        assert res_nokey.success is False
        assert "missing or not configured" in res_nokey.error
        print("✓ Test 6 Passed: Missing Firecrawl API key handled gracefully")

        mock_firecrawl_settings.firecrawl_api_key = "mock_firecrawl_key_123"

        # -------------------------------------------------------------
        # Test 7: Firecrawl API failure or empty results handled gracefully
        # -------------------------------------------------------------
        with patch("backend.app.services.firecrawl_service.Firecrawl") as mock_firecrawl_cls:
            mock_firecrawl_inst = MagicMock()
            mock_firecrawl_cls.return_value = mock_firecrawl_inst

            # Map failure -> falls back to target URL -> Batch scrape failure -> falls back to crawl -> raises exception
            mock_firecrawl_inst.map.side_effect = RuntimeError("Map error")
            mock_firecrawl_inst.batch_scrape.side_effect = RuntimeError("Batch error")
            mock_firecrawl_inst.crawl.side_effect = RuntimeError("Firecrawl service timeout")

            res_err = run_async(scrape_website_url(website_url))
            assert res_err.success is False
            assert "Firecrawl execution error: Firecrawl service timeout" in res_err.error
            print("✓ Test 7 Passed: Firecrawl API failures handled gracefully")

        # -------------------------------------------------------------
        # Test 8: Results preserve input URL ordering when multiple URLs are processed
        # -------------------------------------------------------------
        with patch("backend.app.services.firecrawl_service.Firecrawl") as mock_firecrawl_cls, \
             patch("backend.app.services.apify_service.ApifyClientAsync") as mock_apify_cls:

            mock_firecrawl_inst = MagicMock()
            mock_firecrawl_cls.return_value = mock_firecrawl_inst
            mock_firecrawl_inst.map.return_value = MagicMock(links=[{"url": "https://example.com"}])
            mock_firecrawl_inst.batch_scrape.return_value = MagicMock(data=[
                MagicMock(model_dump=lambda: {"markdown": "Web", "metadata": {"url": "https://example.com"}})
            ])

            mock_apify_inst = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()
            mock_apify_cls.return_value = mock_apify_inst
            mock_apify_inst.actor.return_value = mock_actor
            mock_apify_inst.dataset.return_value = mock_dataset
            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_mix"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"item": "social"}]))

            urls = [website_url, instagram_url, linkedin_url, facebook_url]
            results = run_async(scrape_parsed_urls(urls))

            assert len(results) == 4
            assert results[0].platform == "website"
            assert results[1].platform == "instagram"
            assert results[2].platform == "linkedin"
            assert results[3].platform == "facebook"
            print("✓ Test 8 Passed: Results strictly preserve input URL ordering")

    print("\n=== All Firecrawl Filtering, Scoring & Routing Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_firecrawl_filtering_and_scoring()
    test_firecrawl_and_routing()
