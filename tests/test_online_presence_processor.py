import json
import os
import sys
from unittest.mock import patch

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.models.business_context import BusinessContext
from backend.app.models.online_presence import OnlinePresenceContext, OnlineSourceSummary
from backend.app.models.scraped_result import ApifyScrapeResult
from backend.app.services.online_presence_processor import process_online_presence


def test_online_presence_processor():
    print("=== Testing Online Presence Processor ===")

    ctx = BusinessContext(
        company_name="Sweet Crumbs Bakery",
        product_or_service="artisanal cupcakes and wedding cakes",
        marketing_goal="increase local wedding cake orders",
        target_audience="engaged couples in Seattle"
    )

    web_scrape = ApifyScrapeResult(
        original_url="https://sweetcrumbs.com",
        normalized_url="https://sweetcrumbs.com",
        platform="website",
        success=True,
        data=[{"title": "Custom Wedding Cakes", "text": "We craft premium custom wedding cakes in Seattle."}],
        error=None
    )

    insta_scrape = ApifyScrapeResult(
        original_url="https://instagram.com/sweetcrumbs",
        normalized_url="https://instagram.com/sweetcrumbs",
        platform="instagram",
        success=True,
        data=[{"caption": "Beautiful 3-tier wedding cake delivered today! #seattlewedding"}],
        error=None
    )

    failed_insta_scrape = ApifyScrapeResult(
        original_url="https://instagram.com/sweetcrumbs",
        normalized_url="https://instagram.com/sweetcrumbs",
        platform="instagram",
        success=False,
        data=None,
        error="Apify actor timeout"
    )

    # -------------------------------------------------------------
    # Test 1: Single source (Website only)
    # -------------------------------------------------------------
    mock_llm_json_1 = json.dumps({
        "overall_summary": "Sweet Crumbs Bakery is a Seattle-based artisanal bakery specializing in custom wedding cakes.",
        "website": {
            "summary": "Website highlights custom wedding cake design services.",
            "relevant_products_or_services": ["custom wedding cakes", "cupcakes"],
            "marketing_content": ["Wedding cake portfolio"],
            "positioning_or_messaging": ["Premium custom wedding cakes in Seattle"],
            "other_strategy_relevant_information": ["Offers consultation bookings"]
        },
        "instagram": None,
        "facebook": None,
        "linkedin": None
    })

    with patch("backend.app.services.online_presence_processor.get_llm_response", return_value=mock_llm_json_1):
        res1 = process_online_presence([web_scrape], ctx)
        assert res1 is not None
        assert isinstance(res1, OnlinePresenceContext)
        assert res1.website is not None
        assert res1.website.summary == "Website highlights custom wedding cake design services."
        assert "custom wedding cakes" in res1.website.relevant_products_or_services
        assert res1.instagram is None
        assert res1.facebook is None
        print("✓ Test 1 Passed: Processed single website source successfully")

    # -------------------------------------------------------------
    # Test 2: Multiple sources (Website + Instagram)
    # -------------------------------------------------------------
    mock_llm_json_2 = json.dumps({
        "overall_summary": "Sweet Crumbs Bakery maintains a strong web portfolio and active Instagram showcase.",
        "website": {
            "summary": "Website highlights custom wedding cake design services.",
            "relevant_products_or_services": ["custom wedding cakes"],
            "marketing_content": ["Portfolio"],
            "positioning_or_messaging": ["Premium cakes"],
            "other_strategy_relevant_information": []
        },
        "instagram": {
            "summary": "Instagram showcases recent 3-tier wedding cake deliveries in Seattle.",
            "relevant_products_or_services": ["3-tier wedding cakes"],
            "marketing_content": ["Seattle wedding hashtags"],
            "positioning_or_messaging": [],
            "other_strategy_relevant_information": []
        },
        "facebook": None,
        "linkedin": None
    })

    with patch("backend.app.services.online_presence_processor.get_llm_response", return_value=mock_llm_json_2):
        res2 = process_online_presence([web_scrape, insta_scrape], ctx)
        assert res2 is not None
        assert res2.website is not None
        assert res2.instagram is not None
        assert res2.instagram.summary == "Instagram showcases recent 3-tier wedding cake deliveries in Seattle."
        print("✓ Test 2 Passed: Processed multiple sources (Website + Instagram) successfully")

    # -------------------------------------------------------------
    # Test 3: Partial failure (Website succeeded, Instagram failed)
    # -------------------------------------------------------------
    with patch("backend.app.services.online_presence_processor.get_llm_response", return_value=mock_llm_json_1):
        res3 = process_online_presence([web_scrape, failed_insta_scrape], ctx)
        assert res3 is not None
        assert res3.website is not None
        assert res3.instagram is None
        print("✓ Test 3 Passed: Partial source failure handled gracefully (Website preserved, Instagram None)")

    # -------------------------------------------------------------
    # Test 4: All sources failed
    # -------------------------------------------------------------
    res4 = process_online_presence([failed_insta_scrape], ctx)
    assert res4 is None
    print("✓ Test 4 Passed: All failed sources returns None gracefully")

    # -------------------------------------------------------------
    # Test 5: Empty input list
    # -------------------------------------------------------------
    res5 = process_online_presence([], ctx)
    assert res5 is None
    print("✓ Test 5 Passed: Empty input list returns None gracefully")

    # -------------------------------------------------------------
    # Test 6: Verify 12 Instagram target fields formatting & exclusions
    # -------------------------------------------------------------
    from backend.app.services.online_presence_processor import _format_instagram_item, _format_scraped_data_for_prompt

    sample_insta_item = {
        "caption": "Fresh baked cookies sale! #cookie #bakery @influencer",
        "hashtags": ["cookie", "bakery"],
        "mentions": ["influencer"],
        "timestamp": "2026-10-05T12:00:00Z",
        "likesCount": 450,
        "ownerUsername": "cookiemanindia",
        "ownerFullName": "Cookie Man India",
        "type": "Video",
        "videoViewCount": 1200,
        "paidPartnership": False,
        "taggedUsers": [{"username": "partner_brand"}],
        "url": "https://instagram.com/p/C12345/",
        # Excluded comment fields
        "commentsCount": 45,
        "latestComments": ["Great cookies!"],
        "isCommentsDisabled": False,
        # Excluded technical clutter
        "displayUrl": "https://instagram.com/image.jpg",
        "dimensions": {"height": 1080, "width": 1080}
    }

    formatted = _format_instagram_item(sample_insta_item)
    assert "Author: Cookie Man India (@cookiemanindia)" in formatted
    assert "Type: Video" in formatted
    assert "Caption: Fresh baked cookies sale!" in formatted
    assert "Hashtags: cookie, bakery" in formatted
    assert "Mentions: influencer" in formatted
    assert "Likes: 450" in formatted
    assert "Video Views: 1200" in formatted
    assert "Paid Partnership: False" in formatted
    assert "Tagged Users: partner_brand" in formatted
    assert "URL: https://instagram.com/p/C12345/" in formatted
    assert "latestComments" not in formatted
    assert "displayUrl" not in formatted
    assert "commentsCount" not in formatted

    insta_full_scrape = ApifyScrapeResult(
        original_url="https://instagram.com/cookiemanindia",
        normalized_url="https://instagram.com/cookiemanindia",
        platform="instagram",
        success=True,
        data=[sample_insta_item],
        error=None
    )
    prompt_text = _format_scraped_data_for_prompt([insta_full_scrape])
    assert "--- PLATFORM: INSTAGRAM ---" in prompt_text
    assert "Author: Cookie Man India" in prompt_text
    print("✓ Test 6 Passed: Instagram 12-field extraction & exclusions verified")

    print("\n=== All Online Presence Processor Unit Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_online_presence_processor()
