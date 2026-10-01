import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.parsed_url import ParsedURL
from backend.app.services.apify_service import scrape_single_parsed_url, scrape_parsed_urls


def run_async(coro):
    return asyncio.run(coro)


def test_apify_service():
    print("=== Testing Task 4: Apify Service Integration & Actor Routing ===")

    # Setup parsed URLs for each platform
    website_url = ParsedURL(original_url="https://example.com", normalized_url="https://example.com", platform="website", is_valid=True)
    instagram_url = ParsedURL(original_url="https://instagram.com/brand", normalized_url="https://instagram.com/brand", platform="instagram", is_valid=True)
    facebook_url = ParsedURL(original_url="https://facebook.com/brand", normalized_url="https://facebook.com/brand", platform="facebook", is_valid=True)
    linkedin_url = ParsedURL(original_url="https://linkedin.com/company/brand", normalized_url="https://linkedin.com/company/brand", platform="linkedin", is_valid=True)
    invalid_url = ParsedURL(original_url="invalid_string", normalized_url=None, platform="unknown", is_valid=False, error="Invalid URL")

    with patch("backend.app.services.apify_service.settings") as mock_settings:
        mock_settings.apify_api_key = "mock_apify_token_xyz"
        mock_settings.apify_api_token = "mock_apify_token_xyz"

        # -------------------------------------------------------------
        # Test 1: Website URL -> Actor: apify/website-content-crawler
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_web_123"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"text": "Sample website content"}]))

            res = run_async(scrape_single_parsed_url(website_url))

            mock_client_instance.actor.assert_called_with("apify/website-content-crawler")
            mock_actor.call.assert_called_with(run_input={"startUrls": [{"url": "https://example.com"}], "maxCrawlPages": 3})
            assert res.success is True
            assert res.platform == "website"
            assert res.data == [{"text": "Sample website content"}]
            print("✓ Test 1 Passed: Website URL routed to 'apify/website-content-crawler'")

        # -------------------------------------------------------------
        # Test 2: Instagram URL -> Actor: apify/instagram-scraper
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_insta_123"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"caption": "Post 1"}]))

            res = run_async(scrape_single_parsed_url(instagram_url))

            mock_client_instance.actor.assert_called_with("apify/instagram-scraper")
            mock_actor.call.assert_called_with(run_input={"directUrls": ["https://instagram.com/brand"], "resultsLimit": 5})
            assert res.success is True
            assert res.platform == "instagram"
            print("✓ Test 2 Passed: Instagram URL routed to 'apify/instagram-scraper'")

        # -------------------------------------------------------------
        # Test 3: Facebook URL -> Actor: apify/facebook-posts-scraper
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_fb_123"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"post_text": "FB Post 1"}]))

            res = run_async(scrape_single_parsed_url(facebook_url))

            mock_client_instance.actor.assert_called_with("apify/facebook-posts-scraper")
            mock_actor.call.assert_called_with(run_input={"startUrls": [{"url": "https://facebook.com/brand"}], "maxPosts": 5})
            assert res.success is True
            assert res.platform == "facebook"
            print("✓ Test 3 Passed: Facebook URL routed to 'apify/facebook-posts-scraper'")

        # -------------------------------------------------------------
        # Test 4: LinkedIn URL -> Actor: apify/linkedin-post-scraper
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_li_123"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"text": "LinkedIn Post 1"}]))

            res = run_async(scrape_single_parsed_url(linkedin_url))

            mock_client_instance.actor.assert_called_with("apify/linkedin-post-scraper")
            mock_actor.call.assert_called_with(run_input={"urls": ["https://linkedin.com/company/brand"], "deepScrape": False})
            assert res.success is True
            assert res.platform == "linkedin"
            print("✓ Test 4 Passed: LinkedIn URL routed to 'apify/linkedin-post-scraper'")

        # -------------------------------------------------------------
        # Test 5: Invalid URL is skipped without invoking Apify client
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            res_invalid = run_async(scrape_single_parsed_url(invalid_url))
            mock_client_cls.assert_not_called()
            assert res_invalid.success is False
            assert res_invalid.error is not None
            print("✓ Test 5 Passed: Invalid URL skipped without invoking Apify Client")

        # -------------------------------------------------------------
        # Test 6: Empty Apify dataset result handled gracefully
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_empty"})
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[]))

            res_empty = run_async(scrape_single_parsed_url(website_url))
            assert res_empty.success is False
            assert res_empty.data == []
            assert "empty dataset" in res_empty.error
            print("✓ Test 6 Passed: Empty Apify dataset handled gracefully")

        # -------------------------------------------------------------
        # Test 7: Batch request with mixed URLs and failure resilience
        # -------------------------------------------------------------
        with patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_actor = MagicMock()
            mock_dataset = MagicMock()

            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.actor.return_value = mock_actor
            mock_client_instance.dataset.return_value = mock_dataset

            # First call succeeds, second fails with exception, third is invalid (skipped)
            mock_actor.call = AsyncMock(side_effect=[
                {"defaultDatasetId": "ds_1"},
                RuntimeError("Apify connection timeout"),
            ])
            mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[{"item": "val"}]))

            urls = [website_url, instagram_url, invalid_url]
            results = run_async(scrape_parsed_urls(urls))

            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert "Apify connection timeout" in results[1].error
            assert results[2].success is False
            print("✓ Test 7 Passed: Batch processing handles individual failures without crashing whole request")

    print("\n=== All Task 4 Apify Service Unit Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_apify_service()
