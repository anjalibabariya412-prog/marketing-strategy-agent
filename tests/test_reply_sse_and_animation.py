import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app

client = TestClient(app)


def test_reply_supports_sse_streaming_and_json():
    # 1. Start a quick conversation
    start_payload = {
        "company_name": "Test Animation Brand",
        "product_or_service": "Eco-friendly sneakers",
        "marketing_goal": "Sell 500 pairs in 30 days",
        "target_audience": "Urban athletes aged 20-35",
        "budget_resources": {"amount": 5000, "currency": "USD"},
        "website_social_links": "https://testbrand.example.com",
    }
    res_start = client.post("/start", json=start_payload)
    assert res_start.status_code == 200
    start_data = res_start.json()
    thread_id = start_data["thread_id"]
    assert thread_id is not None
    assert start_data["status"] == "waiting_for_reply"

    # 2. Reply to first question with Accept: text/event-stream
    reply_payload = {
        "thread_id": thread_id,
        "message": "We sell direct-to-consumer online and through running clubs in New York.",
    }
    res_stream = client.post(
        "/reply",
        json=reply_payload,
        headers={"Accept": "text/event-stream, application/json"},
    )
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers.get("content-type", "")

    # Parse stream lines
    events = []
    for line in res_stream.iter_lines():
        if line:
            events.append(line)

    # Verify at least one complete or progress event is streamed
    has_event = any("event:" in line or "data:" in line for line in events)
    assert has_event, "Expected SSE stream lines in response"

    # Check that data lines contain valid JSON
    data_lines = [line for line in events if line.startswith("data: ")]
    assert len(data_lines) > 0
    parsed_payloads = [json.loads(line[6:]) for line in data_lines]
    
    # Verify the last payload is the completion data
    last_payload = parsed_payloads[-1]
    assert "thread_id" in last_payload or "status" in last_payload or "error" in last_payload
