import os
import sys
from unittest.mock import patch, MagicMock

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.agent_state import MarketingAgentState
from backend.app.models.information_requirement import RequirementStatus
from backend.app.models.business_context import BusinessContext
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.agent.graph import graph, ask_node

client = TestClient(app)


def test_1_start_and_reply_resumes_immediately_and_resolves_target_market_location():
    """
    TEST 1:
    - Start a conversation.
    - Generate target_market_location.
    - Interrupt.
    - Resume with: "Ahmedabad and Gandhinagar"
    - Verify the answer is processed immediately.
    - Verify target_market_location becomes resolved.
    - Verify the next requirement/question is generated.
    - Verify the same target_market_location question is NOT generated again.
    """
    print("\n--- TEST 1: /start and /reply resume flow ---")

    start_payload = {
        "company_name": "Organic Greens Ahmedabad",
        "product_or_service": "organic farm-to-table meal kits",
        "marketing_goal": "acquire 200 monthly subscribers",
        "target_audience": "health conscious urban households"
    }

    res_start = client.post("/start", json=start_payload)
    assert res_start.status_code == 200, f"Expected 200, got {res_start.status_code}: {res_start.text}"

    start_data = res_start.json()
    thread_id = start_data.get("thread_id")
    status = start_data.get("status")
    question1 = start_data.get("question")
    req_id1 = start_data.get("requirement_id")

    assert status == "waiting_for_reply"
    assert req_id1 == "target_market_location", f"Expected initial requirement 'target_market_location', got '{req_id1}'"
    print(f"✓ /start generated requirement: '{req_id1}', question: \"{question1}\"")

    # Resume with user answer
    reply_payload = {
        "thread_id": thread_id,
        "message": "Ahmedabad and Gandhinagar"
    }

    res_reply = client.post("/reply", json=reply_payload)
    assert res_reply.status_code == 200, f"Expected 200, got {res_reply.status_code}: {res_reply.text}"

    reply_data = res_reply.json()
    reply_status = reply_data.get("status")
    question2 = reply_data.get("question")
    req_id2 = reply_data.get("requirement_id")

    # Inspect state from checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    assert snapshot and snapshot.values, "Expected state snapshot in checkpointer"

    current_state = MarketingAgentState.model_validate(snapshot.values)
    tml_req = current_state.get_requirement_by_id("target_market_location")
    assert tml_req is not None
    assert tml_req.status in (RequirementStatus.KNOWN, RequirementStatus.UNAVAILABLE), \
        f"Expected target_market_location to be resolved, got status '{tml_req.status}'"

    # Verify qa_history recorded the turn
    asked_req_ids = [turn.requirement_id for turn in current_state.qa_history]
    assert "target_market_location" in asked_req_ids, "Expected target_market_location in qa_history"

    # Verify same target_market_location question is NOT generated again
    assert req_id2 != "target_market_location", f"Requirement 'target_market_location' was repeated in next turn!"

    print(f"✓ /reply processed answer immediately for '{req_id1}'")
    print(f"✓ target_market_location status: '{tml_req.status.value}', value: '{tml_req.value}'")
    print(f"✓ Next requirement generated: '{req_id2}', question: \"{question2}\"")


def test_2_second_reply_processes_active_requirement_first():
    """
    TEST 2:
    - Resume with a second answer.
    - Verify the currently active requirement is processed first.
    - Verify the next question is generated only after processing the answer.
    """
    print("\n--- TEST 2: Second /reply processes active requirement first ---")

    start_payload = {
        "company_name": "TechFlow SaaS",
        "product_or_service": "automated invoice processing software",
        "marketing_goal": "get 50 B2B client demos",
        "target_audience": "finance managers in mid-market companies"
    }

    res_start = client.post("/start", json=start_payload)
    start_data = res_start.json()
    thread_id = start_data.get("thread_id")
    req_id1 = start_data.get("requirement_id")

    # Reply 1
    res_reply1 = client.post("/reply", json={
        "thread_id": thread_id,
        "message": "We currently use LinkedIn ads and direct cold email outreach to target finance managers in Mumbai and Bangalore metro areas."
    })
    reply1_data = res_reply1.json()
    req_id2 = reply1_data.get("requirement_id")
    assert req_id2 is not None and req_id2 != req_id1

    # Reply 2
    res_reply2 = client.post("/reply", json={
        "thread_id": thread_id,
        "message": "Finance managers want to cut invoice processing time from days to minutes and eliminate human error."
    })
    reply2_data = res_reply2.json()

    # Inspect state snapshot
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    current_state = MarketingAgentState.model_validate(snapshot.values)

    # Verify both req_id1 and req_id2 are resolved
    req1_obj = current_state.get_requirement_by_id(req_id1)
    req2_obj = current_state.get_requirement_by_id(req_id2)

    assert req1_obj and req1_obj.status != RequirementStatus.UNKNOWN, f"Expected {req_id1} to be resolved"
    assert req2_obj and req2_obj.status != RequirementStatus.UNKNOWN, f"Expected {req_id2} to be resolved"

    # Verify qa_history order
    qa_ids = [turn.requirement_id for turn in current_state.qa_history]
    assert qa_ids[:2] == [req_id1, req_id2], f"Expected QA history order [{req_id1}, {req_id2}], got {qa_ids}"

    print(f"✓ Second /reply processed active requirement '{req_id2}' first before generating turn 3 question.")


def test_3_ask_node_uses_existing_pending_question_without_calling_prepare():
    """
    TEST 3:
    - Verify that when a valid pending_question already exists, ask_node does NOT call prepare_next_question().
    """
    print("\n--- TEST 3: ask_node reuses valid pending_question without calling prepare_next_question() ---")

    state = MarketingAgentState(thread_id="test-thread-pending-3")
    load_requirements_into_state(state)

    state.pending_requirement_id = "customer_needs_buying_behavior"
    state.pending_question = "What do your customers look for when choosing your service?"

    with patch("backend.app.agent.graph.prepare_next_question") as mock_prep:
        with patch("backend.app.agent.graph.interrupt") as mock_interrupt:
            mock_interrupt.return_value = None  # Simulates pause on interrupt
            
            ask_node(state)

            # Assert prepare_next_question was NOT called
            mock_prep.assert_not_called()

            # Assert pending question was promoted to current_question
            assert state.current_question == "What do your customers look for when choosing your service?"
            assert state.active_requirement_id == "customer_needs_buying_behavior"
            assert state.pending_question is None
            assert state.pending_requirement_id is None

            # Assert interrupt was called with the promoted question
            mock_interrupt.assert_called_once_with({
                "question": "What do your customers look for when choosing your service?",
                "requirement_id": "customer_needs_buying_behavior"
            })

    print("✓ ask_node used pending question directly and did NOT call prepare_next_question()")


def test_4_initial_start_flow_calls_prepare_next_question_when_no_pending_question():
    """
    TEST 4:
    - Verify that the normal initial /start flow still calls prepare_next_question() when there is no pending question.
    """
    print("\n--- TEST 4: ask_node calls prepare_next_question() when no pending question exists ---")

    state = MarketingAgentState(thread_id="test-thread-initial-4")
    load_requirements_into_state(state)

    assert state.pending_question is None
    assert state.current_question is None

    with patch("backend.app.agent.graph.prepare_next_question") as mock_prep:
        def fake_prepare(st):
            st.current_question = "Which location(s) do you want to target?"
            st.active_requirement_id = "target_market_location"
            return st.current_question

        mock_prep.side_effect = fake_prepare

        with patch("backend.app.agent.graph.interrupt") as mock_interrupt:
            mock_interrupt.return_value = None

            ask_node(state)

            # Assert prepare_next_question WAS called once
            mock_prep.assert_called_once_with(state)

            # Assert interrupt was called with the prepared question
            mock_interrupt.assert_called_once_with({
                "question": "Which location(s) do you want to target?",
                "requirement_id": "target_market_location"
            })

    print("✓ ask_node called prepare_next_question() when no pending question existed")


if __name__ == "__main__":
    test_1_start_and_reply_resumes_immediately_and_resolves_target_market_location()
    test_2_second_reply_processes_active_requirement_first()
    test_3_ask_node_uses_existing_pending_question_without_calling_prepare()
    test_4_initial_start_flow_calls_prepare_next_question_when_no_pending_question()
    print("\n=== ALL 4 GRAPH /REPLY FLOW TESTS PASSED SUCCESSFULLY! ===")
