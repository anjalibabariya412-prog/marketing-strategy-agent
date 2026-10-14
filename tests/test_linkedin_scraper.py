import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.app.models.parsed_url import ParsedURL
from backend.app.services.apify_service import (
    ACTOR_MAPPING,
    scrape_single_parsed_url,
)


def run_async(coro):
    return asyncio.run(coro)


def test_linkedin_actor_mapping_configuration():
    """Verify that ACTOR_MAPPING defines the correct actor ID and input builder for LinkedIn."""
    assert "linkedin" in ACTOR_MAPPING
    config = ACTOR_MAPPING["linkedin"]
    assert config["actor_id"] == "automation-lab/linkedin-company-posts-scraper"

    test_url = "https://www.linkedin.com/company/zoho/"
    built_input = config["build_input"](test_url)
    assert built_input == {
        "companyUrls": [test_url],
        "maxPostsPerCompany": 5,
    }


def test_linkedin_scraping_success_routes_to_correct_actor():
    """Verify successful LinkedIn scraping calls the correct actor with proper input and returns items."""
    linkedin_url = ParsedURL(
        original_url="https://www.linkedin.com/company/zoho/",
        normalized_url="https://www.linkedin.com/company/zoho/",
        platform="linkedin",
        is_valid=True,
    )

    mock_items = [
        {
            "companyName": "Zoho",
            "text": "Post 1 about enterprise software innovation.",
            "postUrl": "https://www.linkedin.com/posts/zoho_activity-1",
            "datePublished": "2026-10-13T07:00:33.904Z",
        },
        {
            "companyName": "Zoho",
            "text": "Post 2 about customer engagement.",
            "postUrl": "https://www.linkedin.com/posts/zoho_activity-2",
            "datePublished": "2026-10-12T14:20:00.000Z",
        },
    ]

    with patch("backend.app.services.apify_service.settings") as mock_settings, \
         patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:

        mock_settings.apify_api_token = "mock_token"
        mock_settings.apify_api_key = "mock_token"

        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_dataset = MagicMock()

        mock_client_cls.return_value = mock_client
        mock_client.actor.return_value = mock_actor
        mock_client.dataset.return_value = mock_dataset

        mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_zoho_123"})
        mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=mock_items))

        result = run_async(scrape_single_parsed_url(linkedin_url))

        mock_client.actor.assert_called_with("automation-lab/linkedin-company-posts-scraper")
        mock_actor.call.assert_called_with(
            run_input={
                "companyUrls": ["https://www.linkedin.com/company/zoho/"],
                "maxPostsPerCompany": 5,
            }
        )
        assert result.success is True
        assert result.platform == "linkedin"
        assert result.data == mock_items
        assert len(result.data) == 2
        assert result.error is None


def test_linkedin_scraping_empty_dataset_returns_handled_error():
    """Verify that an empty dataset returned by Apify is handled with an appropriate error."""
    linkedin_url = ParsedURL(
        original_url="https://www.linkedin.com/company/zoho/",
        normalized_url="https://www.linkedin.com/company/zoho/",
        platform="linkedin",
        is_valid=True,
    )

    with patch("backend.app.services.apify_service.settings") as mock_settings, \
         patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:

        mock_settings.apify_api_token = "mock_token"
        mock_settings.apify_api_key = "mock_token"

        mock_client = MagicMock()
        mock_actor = MagicMock()
        mock_dataset = MagicMock()

        mock_client_cls.return_value = mock_client
        mock_client.actor.return_value = mock_actor
        mock_client.dataset.return_value = mock_dataset

        mock_actor.call = AsyncMock(return_value={"defaultDatasetId": "ds_empty"})
        mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[]))

        result = run_async(scrape_single_parsed_url(linkedin_url))

        assert result.success is False
        assert result.platform == "linkedin"
        assert result.data == []
        assert "empty dataset" in result.error.lower()


def test_linkedin_scraping_actor_exception_handled():
    """Verify that exceptions (e.g. auth failure or invalid actor) do not crash the service."""
    linkedin_url = ParsedURL(
        original_url="https://www.linkedin.com/company/zoho/",
        normalized_url="https://www.linkedin.com/company/zoho/",
        platform="linkedin",
        is_valid=True,
    )

    with patch("backend.app.services.apify_service.settings") as mock_settings, \
         patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:

        mock_settings.apify_api_token = "mock_token"
        mock_settings.apify_api_key = "mock_token"

        mock_client = MagicMock()
        mock_actor = MagicMock()

        mock_client_cls.return_value = mock_client
        mock_client.actor.return_value = mock_actor
        mock_actor.call = AsyncMock(side_effect=RuntimeError("Actor authorization failed (401)"))

        result = run_async(scrape_single_parsed_url(linkedin_url))

        assert result.success is False
        assert result.platform == "linkedin"
        assert result.data is None
        assert "Actor authorization failed" in result.error


def test_linkedin_scraping_missing_token_handled():
    """Verify that missing Apify API token returns a clear error without calling Apify."""
    linkedin_url = ParsedURL(
        original_url="https://www.linkedin.com/company/zoho/",
        normalized_url="https://www.linkedin.com/company/zoho/",
        platform="linkedin",
        is_valid=True,
    )

    with patch("backend.app.services.apify_service.settings") as mock_settings, \
         patch("backend.app.services.apify_service.ApifyClientAsync") as mock_client_cls:

        mock_settings.apify_api_token = None
        mock_settings.apify_api_key = None

        result = run_async(scrape_single_parsed_url(linkedin_url))

        mock_client_cls.assert_not_called()
        assert result.success is False
        assert result.platform == "linkedin"
        assert "Apify API token is missing" in result.error
