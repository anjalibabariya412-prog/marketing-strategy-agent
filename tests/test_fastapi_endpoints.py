import os
import sys

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_fastapi_endpoints():
    print("=== Testing Task 8.4: FastAPI Integration Endpoints ===")

    # -------------------------------------------------------------
    # Test 1: POST /start (Start new conversation)
    # -------------------------------------------------------------
    print("\n--- Test 1: POST /start ---")
    start_payload = {
        "company_name": "CleanSeas Ocean Cleanup",
        "product_or_service": "ocean plastic cleanup drives",
        "marketing_goal": "recruit 500 volunteer cleanup captains",
        "target_audience": "college students aged 18-30",
        "budget_resources": "$2,000 promotional budget"
    }
    res_start = client.post("/start", json=start_payload)
    assert res_start.status_code == 200, f"Expected 200, got {res_start.status_code}: {res_start.text}"

    start_data = res_start.json()
    thread_id = start_data.get("thread_id")
    status = start_data.get("status")
    question = start_data.get("question")
    req_id = start_data.get("requirement_id")

    assert thread_id is not None, "Expected thread_id in response."
    assert status == "waiting_for_reply", f"Expected status 'waiting_for_reply', got '{status}'"
    assert question is not None, "Expected question text when graph interrupts."
    assert req_id is not None, "Expected requirement_id in response."

    print(f"✓ POST /start successful:")
    print(f"  thread_id: '{thread_id}'")
    print(f"  status: '{status}'")
    print(f"  requirement_id: '{req_id}'")
    print(f"  question: \"{question}\"")

    # -------------------------------------------------------------
    # Test 5: GET /strategy Prematurely (Before completion)
    # -------------------------------------------------------------
    print("\n--- Test 5: GET /strategy (Premature Request) ---")
    res_premature = client.get(f"/strategy?thread_id={thread_id}")
    assert res_premature.status_code == 400, f"Expected 400 for premature strategy, got {res_premature.status_code}"
    print(f"✓ Premature GET /strategy returned 400 BAD REQUEST: {res_premature.json()['detail']}")

    # -------------------------------------------------------------
    # Test 4: POST /reply & GET /strategy Invalid Thread ID
    # -------------------------------------------------------------
    print("\n--- Test 4: Invalid Thread ID (404 Handling) ---")
    res_fake_reply = client.post("/reply", json={"thread_id": "nonexistent-thread-xyz", "message": "hello"})
    assert res_fake_reply.status_code == 404, f"Expected 404 for invalid thread, got {res_fake_reply.status_code}"
    print(f"✓ Invalid POST /reply returned 404 NOT FOUND: {res_fake_reply.json()['detail']}")

    res_fake_strat = client.get("/strategy?thread_id=nonexistent-thread-xyz")
    assert res_fake_strat.status_code == 404, f"Expected 404 for invalid thread, got {res_fake_strat.status_code}"
    print(f"✓ Invalid GET /strategy returned 404 NOT FOUND: {res_fake_strat.json()['detail']}")

    # -------------------------------------------------------------
    # Test 2: POST /reply (Submit answer & continue conversation)
    # -------------------------------------------------------------
    print("\n--- Test 2: POST /reply (Resuming conversation) ---")
    reply_payload = {
        "thread_id": thread_id,
        "message": "$2,000 promotional budget, 2 full-time staff members, and student club networks across 4 local campuses."
    }
    res_reply = client.post("/reply", json=reply_payload)
    assert res_reply.status_code == 200, f"Expected 200, got {res_reply.status_code}: {res_reply.text}"

    reply_data = res_reply.json()
    reply_status = reply_data.get("status")
    next_q = reply_data.get("question")
    next_req = reply_data.get("requirement_id")

    print(f"✓ POST /reply successful:")
    print(f"  thread_id: '{reply_data.get('thread_id')}'")
    print(f"  status: '{reply_status}'")
    if reply_status == "waiting_for_reply":
        print(f"  next_requirement_id: '{next_req}'")
        print(f"  next_question: \"{next_q}\"")

    # -------------------------------------------------------------
    # Test 3: Complete Q&A turns until strategy is generated, then GET /strategy
    # -------------------------------------------------------------
    print("\n--- Test 3: Complete Conversation Flow to Strategy Generation ---")
    current_status = reply_status
    max_turns = 10
    turns = 0

    while current_status == "waiting_for_reply" and turns < max_turns:
        turns += 1
        print(f"\nSubmitting turn {turns} answer...")
        turn_reply = client.post("/reply", json={
            "thread_id": thread_id,
            "message": f"Sample response for requirement iteration {turns} providing comprehensive details."
        })
        assert turn_reply.status_code == 200, f"Expected 200, got {turn_reply.status_code}: {turn_reply.text}"
        turn_data = turn_reply.json()
        current_status = turn_data.get("status")
        print(f"  Turn {turns} status -> '{current_status}'")

    print(f"\nConversation reached status: '{current_status}'")

    if current_status == "completed":
        print("\nFetching final strategy via GET /strategy...")
        res_strategy = client.get(f"/strategy?thread_id={thread_id}")
        assert res_strategy.status_code == 200, f"Expected 200, got {res_strategy.status_code}: {res_strategy.text}"

        strat_data = res_strategy.json()
        assert strat_data.get("status") == "completed"
        strategy_obj = strat_data.get("strategy")
        assert strategy_obj is not None, "Expected strategy object in response."

        print(f"✓ GET /strategy returned 200 OK:")
        print(f"  business_overview: \"{strategy_obj.get('business_overview')[:80] if strategy_obj.get('business_overview') else None}...\"")
        print(f"  target_audience_insights: \"{strategy_obj.get('target_audience_insights')[:80] if strategy_obj.get('target_audience_insights') else None}...\"")
        print(f"  kpis: \"{strategy_obj.get('kpis')[:80] if strategy_obj.get('kpis') else None}...\"")

    print("\n=== All FastAPI Integration Tests Passed Successfully! ===")


if __name__ == "__main__":
    test_fastapi_endpoints()
