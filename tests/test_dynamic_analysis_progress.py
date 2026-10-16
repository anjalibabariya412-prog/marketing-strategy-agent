import json
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.agent.parsed_url import ParsedURL
from backend.app.schemas.agent.scraped_result import ApifyScrapeResult
from backend.app.db.session import SessionLocal
from backend.app.models.uploaded_document import UploadedDocument
from backend.app.services.apify_service import scrape_parsed_urls
from backend.app.core.progress_notifier import set_progress_notifier, reset_progress_notifier

client = TestClient(app)


def parse_sse_events(response_text: str):
    """Parses raw text/event-stream into a list of {event, data} dicts."""
    events = []
    blocks = response_text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        event_type = None
        data_str = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str = line.replace("data:", "").strip()
        if event_type or data_str:
            parsed_data = None
            if data_str:
                try:
                    parsed_data = json.loads(data_str)
                except Exception:
                    parsed_data = data_str
            events.append({"event": event_type, "data": parsed_data})
    return events


def test_no_optional_inputs_emits_business_and_strategy_status():
    """Scenario 5: No optional inputs provided."""
    payload = {
        "company_name": "Acme Local Bakery",
        "product_or_service": "Artisan sourdough bread",
        "marketing_goal": "Increase store walk-ins",
        "target_audience": "Local neighborhood residents",
        "budget_resources": {"amount": 500, "currency": "USD"},
        "current_marketing_channels": "Word of mouth",
    }

    res = client.post(
        "/start",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    events = parse_sse_events(res.text)
    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}: {events}"

    progress_events = [e for e in events if e["event"] == "progress"]
    complete_events = [e for e in events if e["event"] == "complete"]

    assert len(progress_events) >= 1
    assert len(complete_events) == 1

    messages = [e["data"]["message"] for e in progress_events]
    all_text = " ".join(messages)

    # Must contain business analysis message
    assert any("business" in m.lower() for m in messages)
    # Must NOT contain website, social, or document messages
    assert "website" not in all_text.lower()
    assert "social" not in all_text.lower()
    assert "uploaded document" not in all_text.lower()

    # Complete event must have thread_id and valid status
    assert "thread_id" in complete_events[0]["data"]
    assert complete_events[0]["data"]["status"] in ("waiting_for_reply", "completed")


def test_website_only_input_emits_website_progress():
    """Scenario 1: Website-only input."""
    payload = {
        "company_name": "TechFlow Software",
        "product_or_service": "Workflow automation SaaS",
        "marketing_goal": "Generate B2B demo signups",
        "target_audience": "Operations managers",
        "budget_resources": {"amount": 1500, "currency": "USD"},
        "current_marketing_channels": "Google Search",
        "website_social_links": "https://techflow-example.io",
    }

    # Mock scraping to keep test fast and deterministic
    mock_result = [
        ApifyScrapeResult(
            original_url="https://techflow-example.io",
            normalized_url="https://techflow-example.io",
            platform="website",
            success=True,
            data=[{"url": "https://techflow-example.io", "markdown": "Workflow automation tools"}],
        )
    ]

    with patch("backend.app.api.conversation.scrape_parsed_urls", side_effect=scrape_parsed_urls), \
         patch("backend.app.services.apify_service.scrape_single_parsed_url", return_value=mock_result[0]):
        res = client.post(
            "/start",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    assert res.status_code == 200
    events = parse_sse_events(res.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    messages = [e["data"]["message"] for e in progress_events]
    all_text = " ".join(messages)

    # Must contain website review message
    assert any("website" in m.lower() for m in messages)
    # Must NOT contain social or document messages
    assert "social" not in all_text.lower()
    assert "uploaded document" not in all_text.lower()


def test_social_only_input_emits_social_progress():
    """Scenario 2: Social-media-only input."""
    payload = {
        "company_name": "Glow Cosmetics",
        "product_or_service": "Cruelty-free skincare line",
        "marketing_goal": "Build brand awareness",
        "target_audience": "Gen Z beauty enthusiasts",
        "budget_resources": {"amount": 2500, "currency": "USD"},
        "current_marketing_channels": "Influencer gifting",
        "website_social_links": "https://instagram.com/glowcosmetics",
    }

    mock_result = [
        ApifyScrapeResult(
            original_url="https://instagram.com/glowcosmetics",
            normalized_url="https://instagram.com/glowcosmetics",
            platform="instagram",
            success=True,
            data=[{"caption": "Glow with natural ingredients!"}],
        )
    ]

    with patch("backend.app.api.conversation.scrape_parsed_urls", side_effect=scrape_parsed_urls), \
         patch("backend.app.services.apify_service.scrape_single_parsed_url", return_value=mock_result[0]):
        res = client.post(
            "/start",
            json=payload,
            headers={"Accept": "text/event-stream"},
        )

    assert res.status_code == 200
    events = parse_sse_events(res.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    messages = [e["data"]["message"] for e in progress_events]
    all_text = " ".join(messages)

    # Must contain social review message
    assert any("social" in m.lower() for m in messages)
    # Must NOT contain website or document messages
    assert "website" not in all_text.lower()
    assert "uploaded document" not in all_text.lower()


def test_pdf_only_input_emits_document_progress():
    """Scenario 3: PDF-only input."""
    doc_id = uuid.uuid4()
    with SessionLocal() as db:
        test_doc = UploadedDocument(
            id=doc_id,
            filename="annual_marketing_review.pdf",
            file_size_bytes=2048,
            page_count=1,
            is_marketing_relevant=True,
            document_type="marketing_plan",
            relevance_reason="Previous marketing strategy",
            executive_summary="Targeted 15% increase in conversions via email marketing.",
            raw_extracted_text="Targeted 15% increase in conversions via email marketing.",
        )
        db.add(test_doc)
        db.commit()

    payload = {
        "company_name": "Summit Financial Advisory",
        "product_or_service": "Retirement planning services",
        "marketing_goal": "Acquire 50 new consultation clients",
        "target_audience": "Professionals aged 45-60",
        "budget_resources": {"amount": 5000, "currency": "USD"},
        "current_marketing_channels": "Local seminars",
        "uploaded_document_id": str(doc_id),
    }

    res = client.post(
        "/start",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert res.status_code == 200

    events = parse_sse_events(res.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    messages = [e["data"]["message"] for e in progress_events]
    all_text = " ".join(messages)

    # Must contain document message
    assert any("document" in m.lower() for m in messages)
    # Must NOT contain website or social messages
    assert "website" not in all_text.lower()
    assert "social" not in all_text.lower()


def test_parallel_operations_completing_and_transitioning():
    """Scenario 4 & 6: Website and social media run in parallel and update progress when one finishes."""
    emitted = []

    def tracking_notifier(msg, detail=None):
        emitted.append((msg, detail))

    token = set_progress_notifier(tracking_notifier)

    parsed_urls = [
        ParsedURL(
            original_url="https://cleanseas.org",
            normalized_url="https://cleanseas.org",
            platform="website",
            is_valid=True,
        ),
        ParsedURL(
            original_url="https://instagram.com/cleanseas",
            normalized_url="https://instagram.com/cleanseas",
            platform="instagram",
            is_valid=True,
        ),
    ]

    import asyncio

    async def run_parallel_scrape_sim():
        async def mock_scrape_single(parsed_url: ParsedURL):
            if parsed_url.platform == "website":
                await asyncio.sleep(0.01)  # website finishes first
                return ApifyScrapeResult(
                    original_url=parsed_url.original_url,
                    normalized_url=parsed_url.normalized_url,
                    platform="website",
                    success=True,
                    data=[],
                )
            else:
                await asyncio.sleep(0.05)  # social finishes second
                return ApifyScrapeResult(
                    original_url=parsed_url.original_url,
                    normalized_url=parsed_url.normalized_url,
                    platform="instagram",
                    success=True,
                    data=[],
                )

        with patch("backend.app.services.apify_service.scrape_single_parsed_url", side_effect=mock_scrape_single):
            return await scrape_parsed_urls(parsed_urls)

    results = asyncio.run(run_parallel_scrape_sim())
    reset_progress_notifier(token)

    assert len(results) == 2
    messages = [msg for msg, _ in emitted]

    # Initial parallel message
    assert "Reviewing your online presence..." in messages
    # Transition message when website finishes and social is still active
    assert "Reviewing your social media presence..." in messages


def test_scraping_failure_preserves_fallback():
    """Scenario 7: One optional scraping operation fails while another succeeds."""
    parsed_urls = [
        ParsedURL(
            original_url="https://cleanseas.org",
            normalized_url="https://cleanseas.org",
            platform="website",
            is_valid=True,
        ),
        ParsedURL(
            original_url="https://instagram.com/cleanseas",
            normalized_url="https://instagram.com/cleanseas",
            platform="instagram",
            is_valid=True,
        ),
    ]

    import asyncio

    async def run_failure_scrape():
        async def mock_scrape_single(parsed_url: ParsedURL):
            if parsed_url.platform == "website":
                return ApifyScrapeResult(
                    original_url=parsed_url.original_url,
                    normalized_url=parsed_url.normalized_url,
                    platform="website",
                    success=False,
                    error="Firecrawl timeout error",
                )
            else:
                return ApifyScrapeResult(
                    original_url=parsed_url.original_url,
                    normalized_url=parsed_url.normalized_url,
                    platform="instagram",
                    success=True,
                    data=[{"followers": 5000}],
                )

        with patch("backend.app.services.apify_service.scrape_single_parsed_url", side_effect=mock_scrape_single):
            return await scrape_parsed_urls(parsed_urls)

    results = asyncio.run(run_failure_scrape())
    assert len(results) == 2
    web_res = next(r for r in results if r.platform == "website")
    soc_res = next(r for r in results if r.platform == "instagram")

    assert web_res.success is False
    assert "Firecrawl timeout error" in web_res.error
    assert soc_res.success is True


def test_full_workflow_failure_emits_error_event():
    """Scenario 8: Workflow error emits SSE error event and terminates cleanly."""
    payload = {
        "company_name": "Test Fail Co",
        "product_or_service": "Testing failure path",
        "marketing_goal": "Check error event",
        "target_audience": "Testers",
        "budget_resources": {"amount": 100, "currency": "USD"},
        "current_marketing_channels": "None",
        "uploaded_document_id": str(uuid.uuid4()),  # Nonexistent doc ID -> raises HTTPException 400
    }

    res = client.post(
        "/start",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert res.status_code == 200

    events = parse_sse_events(res.text)
    error_events = [e for e in events if e["event"] == "error"]

    assert len(error_events) >= 1, f"Expected error event, got: {events}"
    assert "not found" in str(error_events[0]["data"]).lower()


def test_backward_compatibility_json_response():
    """Verify standard POST /start without Accept: text/event-stream returns JSON StartResponse."""
    payload = {
        "company_name": "Normal JSON Co",
        "product_or_service": "Consulting",
        "marketing_goal": "Get clients",
        "target_audience": "Local businesses",
        "budget_resources": {"amount": 1000, "currency": "USD"},
        "current_marketing_channels": "Email",
    }

    res = client.post("/start", json=payload)
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]

    data = res.json()
    assert "thread_id" in data
    assert "status" in data
