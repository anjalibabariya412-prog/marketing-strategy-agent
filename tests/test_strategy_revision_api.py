import json
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app
from backend.app.db.session import SessionLocal
from backend.app.models.strategy_request import StrategyRequest
from backend.app.models.business import Business
from backend.app.models.conversation import Conversation
from backend.app.models.strategy import Strategy
from backend.app.models.revision_message import RevisionMessage
from backend.app.services.llm_service import LLMServiceError

client = TestClient(app)


SAMPLE_STRATEGY_JSON = {
    "business_overview": [
        "EcoClean is a green commercial cleaning service targeting urban corporate offices in Mumbai."
    ],
    "target_audience_insights": [
        "Facilities managers and corporate administrative heads prioritizing sustainability."
    ],
    "competitive_positioning": [
        "Positioned against standard chemical cleaning services by offering certified non-toxic solutions."
    ],
    "value_proposition": [
        "Certified eco-friendly office cleaning that reduces toxic exposure and supports corporate ESG goals."
    ],
    "marketing_channels_and_tactics": [
        "LinkedIn direct outreach to facilities heads and local SEO for Mumbai corporate hubs."
    ],
    "customer_acquisition_approach": [
        "Offer a free 1-week pilot green cleaning trial to prospective corporate clients."
    ],
    "budget_considerations": [
        {
            "label": "LinkedIn InMail & Outreach",
            "amount": "₹30,000",
            "percentage": "50%",
            "description": "Targeted direct outreach campaigns to corporate facilities heads."
        },
        {
            "label": "Local SEO & Content",
            "amount": "₹30,000",
            "percentage": "50%",
            "description": "Search optimization and sustainability case studies."
        }
    ],
    "kpis": [
        {
            "metric": "Corporate Pilot Trials",
            "target": "10 pilots per month",
            "description": "Signed trial agreements from commercial prospects."
        }
    ],
    "action_plan": [
        "Week 1-2: Audit current target account list in Mumbai.",
        "Week 3-4: Launch LinkedIn outreach and pilot offer."
    ]
}


@pytest.fixture
def mock_strategy_session():
    """
    Creates a temporary StrategyRequest, Business, Conversation, and Strategy in the database.
    Cleans them up after the test.
    """
    with SessionLocal() as db:
        strat_req = StrategyRequest(
            status="completed",
            current_stage="completed",
        )
        db.add(strat_req)
        db.flush()

        biz = Business(
            strategy_request_id=strat_req.id,
            company_name="EcoClean Facility Solutions",
            product_or_service="Green commercial cleaning",
            marketing_goal="Acquire 10 corporate office clients in Mumbai",
            target_audience="Corporate facility heads in Mumbai",
            budget_amount=60000,
            budget_currency="INR",
            current_marketing_channels="Direct sales, Word of mouth",
            website_social_links="https://ecoclean.in",
        )
        db.add(biz)

        conv = Conversation(
            strategy_request_id=strat_req.id,
            status="completed",
            is_sufficient=True,
        )
        db.add(conv)
        db.flush()

        strat = Strategy(
            conversation_id=conv.id,
            version=1,
            strategy_json=SAMPLE_STRATEGY_JSON,
        )
        db.add(strat)
        db.commit()

        strat_req_id = strat_req.id
        conv_id = conv.id

    yield {
        "strategy_request_id": str(strat_req_id),
        "thread_id": str(conv_id),
    }

    # Cleanup
    with SessionLocal() as db:
        req = db.query(StrategyRequest).filter(StrategyRequest.id == strat_req_id).first()
        if req:
            db.delete(req)
            db.commit()


# ==============================================================================
# Phase 3 Tests: Intelligent Strategy Revision Workflow
# ==============================================================================

def test_clear_request_needs_no_clarification(mock_strategy_session):
    """
    Test 1: Clear request with enough info advances directly to proposal_ready
    without asking an unnecessary follow-up question.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "I've structured a proposal to reduce your budget to ₹30,000 with 60% allocated to Instagram and 40% to SEO. Does this look good to you?",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Reduce marketing budget to ₹30,000 and reallocate to Instagram and SEO",
        "affected_sections": ["budget_considerations", "marketing_channels_and_tactics"],
        "proposed_changes": [
            {
                "section_id": "budget_considerations",
                "section_name": "Budget Considerations & Allocation",
                "proposed_change": "Reduce total allocation to ₹30,000 (₹18,000 Instagram, ₹12,000 SEO)",
                "unchanged_aspects": "Payment schedule remains unchanged"
            },
            {
                "section_id": "marketing_channels_and_tactics",
                "section_name": "Marketing Channels & Tactics",
                "proposed_change": "Shift focus to Instagram Ads and organic SEO",
                "unchanged_aspects": "Content themes remain sustainability focused"
            }
        ],
        "unchanged_facts": ["Core product is green cleaning", "Target market is Mumbai offices"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": ["Assumed 60-40 split based on user input"]
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Reduce my marketing budget to ₹30,000 and allocate 60% to Instagram and 40% to SEO."}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        state = data["revision_state"]
        assert state is not None
        assert state["status"] == "proposal_ready"
        assert state["is_ready_for_confirmation"] is True
        assert len(state["missing_information"]) == 0
        assert "budget_considerations" in state["affected_sections"]
        assert "marketing_channels_and_tactics" in state["affected_sections"]


def test_request_requiring_follow_up_question(mock_strategy_session):
    """
    Test 2: An incomplete or ambiguous request triggers a clarifying question.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "I'd be happy to adjust your target audience. Which specific demographic or customer persona would you like to shift focus toward?",
        "intent_type": "new_request",
        "status": "clarifying",
        "normalized_intent": "Update target audience profile",
        "affected_sections": ["target_audience_insights"],
        "proposed_changes": [],
        "unchanged_facts": ["Product is green cleaning", "Operating in Mumbai"],
        "missing_information": ["New target persona or customer segment"],
        "current_clarification_question": "Which specific demographic or customer persona would you like to shift focus toward?",
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Change my target audience."}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        assert state["status"] == "clarifying"
        assert state["is_ready_for_confirmation"] is False
        assert state["current_clarification_question"] is not None
        assert len(state["missing_information"]) > 0


def test_multiturn_clarification_workflow(mock_strategy_session):
    """
    Test 3: Clarifying question in turn 1 followed by clarification answer in turn 2
    leads to proposal_ready in turn 2 while remembering original intent.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    turn1_json = json.dumps({
        "reply": "Which specific social media platform would you like to add?",
        "intent_type": "new_request",
        "status": "clarifying",
        "normalized_intent": "Add a new social media marketing channel",
        "affected_sections": ["marketing_channels_and_tactics"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": ["Platform name"],
        "current_clarification_question": "Which specific social media platform would you like to add?",
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    turn2_json = json.dumps({
        "reply": "I have created a revision proposal to incorporate TikTok short-form video campaigns into Section 5. Please confirm if you would like me to apply this.",
        "intent_type": "clarification_answer",
        "status": "proposal_ready",
        "normalized_intent": "Add TikTok short-form video campaigns",
        "affected_sections": ["marketing_channels_and_tactics", "action_plan"],
        "proposed_changes": [
            {
                "section_id": "marketing_channels_and_tactics",
                "section_name": "Marketing Channels & Tactics",
                "proposed_change": "Introduce TikTok short-form educational videos on green cleaning",
                "unchanged_aspects": "LinkedIn remains primary B2B channel"
            }
        ],
        "unchanged_facts": ["Target audience is commercial offices"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": ["Assumed organic video creation without external production agency"]
    })

    with patch("backend.app.services.revision_service.get_llm_response") as mock_llm:
        mock_llm.side_effect = [turn1_json, turn2_json]

        # Turn 1
        res1 = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Add a new social media channel."}
        )
        assert res1.json()["revision_state"]["status"] == "clarifying"

        # Turn 2
        res2 = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "TikTok for viral short videos."}
        )
        state2 = res2.json()["revision_state"]
        assert state2["status"] == "proposal_ready"
        assert state2["is_ready_for_confirmation"] is True
        assert "marketing_channels_and_tactics" in state2["affected_sections"]


def test_user_correction_updates_proposal(mock_strategy_session):
    """
    Test 4: User correction updates proposed changes rather than treating as confirmation.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "I've updated the proposal to reflect ₹35,000 instead of ₹30,000 for your total budget. Does this look accurate now?",
        "intent_type": "correction",
        "status": "proposal_ready",
        "normalized_intent": "Adjust revised budget to ₹35,000",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [
            {
                "section_id": "budget_considerations",
                "section_name": "Budget Considerations & Allocation",
                "proposed_change": "Set total budget to ₹35,000",
                "unchanged_aspects": "Channel distribution remains 50-50"
            }
        ],
        "unchanged_facts": ["Product is green commercial cleaning"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "correction",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "No, make the budget ₹35,000 instead of ₹30,000."}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        assert state["status"] == "proposal_ready"
        assert state["confirmation_status"] == "correction"
        assert state["is_ready_for_confirmation"] is True


def test_explicit_confirmation_and_rejection(mock_strategy_session):
    """
    Test 5: Explicit confirmation sets status to confirmed, and explicit rejection sets status to rejected.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    confirm_json = json.dumps({
        "reply": "Your revision proposal has been confirmed and saved for application.",
        "intent_type": "confirmation",
        "status": "confirmed",
        "normalized_intent": "Confirmed revision proposal",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "confirmed",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=confirm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Yes, please confirm and save this revision proposal."}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        assert state["status"] == "confirmed"
        assert state["confirmation_status"] == "confirmed"

    reject_json = json.dumps({
        "reply": "Understood, this revision proposal has been cancelled. Your strategy remains untouched.",
        "intent_type": "rejection",
        "status": "rejected",
        "normalized_intent": "Cancelled revision proposal",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "rejected",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=reject_json):
        res_reject = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Please cancel this revision proposal."}
        )
        assert res_reject.status_code == 200
        state_reject = res_reject.json()["revision_state"]
        assert state_reject["status"] == "rejected"
        assert state_reject["confirmation_status"] == "rejected"


def test_ambiguous_positive_response_does_not_auto_confirm(mock_strategy_session):
    """
    Test 6: Casual positive remarks with questions (e.g. "Cool, but what about X?")
    must NOT automatically confirm the proposal.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    inquiry_json = json.dumps({
        "reply": "Regarding pricing, the green certified materials are included in the revised budget. Would you like to confirm the proposal or adjust the pricing model?",
        "intent_type": "inquiry",
        "status": "proposal_ready",
        "normalized_intent": "Inquiry about pricing details within proposal",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=inquiry_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Cool, but what about pricing?"}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        # Must still be pending, not confirmed!
        assert state["confirmation_status"] == "pending"
        assert state["status"] == "proposal_ready"


def test_persistence_and_restoration_of_revision_state(mock_strategy_session):
    """
    Test 7: Revision state is persisted durably and restored identically across GET requests.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "Proposed changes to Section 9: Timeline adjusted to 8 weeks.",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Extend implementation timeline to 8 weeks",
        "affected_sections": ["action_plan"],
        "proposed_changes": [
            {
                "section_id": "action_plan",
                "section_name": "Action Plan & Timeline",
                "proposed_change": "Extend timeline to 8 weeks",
                "unchanged_aspects": "Milestone deliverables remain same"
            }
        ],
        "unchanged_facts": ["Budget remains unchanged"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Extend the timeline to 8 weeks."}
        )

    # 1. Fetch via GET /revisions/state
    res_state = client.get(f"/api/strategy-requests/{strat_req_id}/revisions/state")
    assert res_state.status_code == 200
    state_data = res_state.json()["revision_state"]
    assert state_data["status"] == "proposal_ready"
    assert "action_plan" in state_data["affected_sections"]

    # 2. Fetch via GET /revisions/messages
    res_hist = client.get(f"/api/strategy-requests/{strat_req_id}/revisions/messages")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert hist_data["revision_state"]["status"] == "proposal_ready"
    # Ensure internal revision_state records are NOT included in the user-visible message list
    for msg in hist_data["messages"]:
        assert msg["sender"] in ("user", "assistant")


def test_isolation_between_different_strategy_requests(mock_strategy_session):
    """
    Test 8: Two separate strategy requests maintain isolated revision states and histories.
    """
    strat_req_1 = mock_strategy_session["strategy_request_id"]

    with SessionLocal() as db:
        sr2 = StrategyRequest(status="completed", current_stage="completed")
        db.add(sr2)
        db.flush()
        c2 = Conversation(strategy_request_id=sr2.id, status="completed", is_sufficient=True)
        db.add(c2)
        db.flush()
        s2 = Strategy(conversation_id=c2.id, version=1, strategy_json=SAMPLE_STRATEGY_JSON)
        db.add(s2)
        db.commit()
        strat_req_2 = str(sr2.id)

    try:
        reply_1 = json.dumps({
            "reply": "Session 1 proposal.",
            "intent_type": "new_request",
            "status": "proposal_ready",
            "normalized_intent": "Session 1 intent",
            "affected_sections": ["budget_considerations"],
            "proposed_changes": [],
            "unchanged_facts": [],
            "missing_information": [],
            "current_clarification_question": None,
            "is_ready_for_confirmation": True,
            "confirmation_status": "pending",
            "assumptions_and_constraints": []
        })

        with patch("backend.app.services.revision_service.get_llm_response", return_value=reply_1):
            client.post(
                f"/api/strategy-requests/{strat_req_1}/revisions/messages",
                json={"message": "Message for Session 1"}
            )

        # Session 2 state should still be empty/default
        res2 = client.get(f"/api/strategy-requests/{strat_req_2}/revisions/state")
        assert res2.status_code == 200
        state2 = res2.json()["revision_state"]
        assert state2["normalized_intent"] is None
        assert state2["affected_sections"] == []

        # Session 1 state should reflect session 1 intent
        res1 = client.get(f"/api/strategy-requests/{strat_req_1}/revisions/state")
        assert res1.json()["revision_state"]["normalized_intent"] == "Session 1 intent"
    finally:
        with SessionLocal() as db:
            req2 = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_2)).first()
            if req2:
                db.delete(req2)
                db.commit()


def test_invalid_structured_llm_output_handling(mock_strategy_session):
    """
    Test 9: Invalid JSON from LLM does not crash the server and falls back gracefully.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    non_json_output = "I recommend reallocating your marketing budget to organic channels."

    with patch("backend.app.services.revision_service.get_llm_response", return_value=non_json_output):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What do you suggest for budget?"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["reply"] == non_json_output
        assert data["revision_state"] is not None


def test_saved_strategy_is_not_modified_in_phase_3(mock_strategy_session):
    """
    Test 10: Crucial constraint: Strategy.strategy_json in database remains untouched
    before, during, and after confirmation in Phase 3.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Verify strategy in DB before
    with SessionLocal() as db:
        conv = db.query(Conversation).filter(Conversation.strategy_request_id == uuid.UUID(strat_req_id)).first()
        initial_strategy = db.query(Strategy).filter(Strategy.conversation_id == conv.id).first()
        initial_json = dict(initial_strategy.strategy_json)

    # Post a message that confirms a revision
    confirm_json = json.dumps({
        "reply": "Confirmed and recorded for future application.",
        "intent_type": "confirmation",
        "status": "confirmed",
        "normalized_intent": "Confirmed budget cut",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "confirmed",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=confirm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Yes, confirm."}
        )
        assert res.status_code == 200

    # Verify strategy in DB after: MUST BE UNCHANGED
    with SessionLocal() as db:
        conv_after = db.query(Conversation).filter(Conversation.strategy_request_id == uuid.UUID(strat_req_id)).first()
        strategy_after = db.query(Strategy).filter(Strategy.conversation_id == conv_after.id).first()
        assert strategy_after.strategy_json == initial_json
        # Also ensure no new strategy versions were inserted
        all_strats = db.query(Strategy).filter(Strategy.conversation_id == conv_after.id).all()
        assert len(all_strats) == 1


# ==============================================================================
# Phase 2 Regression Tests
# ==============================================================================

def test_regression_resolution_by_thread_id(mock_strategy_session):
    """
    Regression Test: Conversation thread_id still resolves session correctly.
    """
    thread_id = mock_strategy_session["thread_id"]
    reply_json = json.dumps({
        "reply": "Action plan adjustment proposal.",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Make weekly plan more practical",
        "affected_sections": ["action_plan"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=reply_json):
        res = client.post(
            f"/api/strategy-requests/{thread_id}/revisions/messages",
            json={"message": "Make the weekly plan more practical."},
        )
        assert res.status_code == 200
        assert res.json()["reply"] == "Action plan adjustment proposal."


def test_regression_missing_or_invalid_strategy_request():
    """
    Regression Test: Invalid UUIDs and empty messages handled properly.
    """
    res_invalid = client.post(
        "/api/strategy-requests/not-a-valid-uuid/revisions/messages",
        json={"message": "Revise my strategy"},
    )
    assert res_invalid.status_code == 404

    random_uuid = str(uuid.uuid4())
    res_nonexistent = client.post(
        f"/api/strategy-requests/{random_uuid}/revisions/messages",
        json={"message": "Revise my strategy"},
    )
    assert res_nonexistent.status_code == 404

    res_empty = client.post(
        f"/api/strategy-requests/{random_uuid}/revisions/messages",
        json={"message": "   "},
    )
    assert res_empty.status_code == 422


def test_regression_llm_failure_handling(mock_strategy_session):
    """
    Regression Test: LLM failure handled gracefully without leaking keys.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    with patch(
        "backend.app.services.revision_service.get_llm_response",
        side_effect=LLMServiceError("Connection error to Groq/OpenAI"),
    ):
        response = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Change my pricing strategy."},
        )
        assert response.status_code == 500
        detail = response.json().get("detail", "")
        assert "temporarily unavailable" in detail.lower()
        assert "sk-" not in detail


def test_regression_no_strategy_generated_yet():
    """
    Regression Test: Calling endpoint before strategy generation returns 400.
    """
    with SessionLocal() as db:
        incomplete_req = StrategyRequest(status="in_progress")
        db.add(incomplete_req)
        db.commit()
        incomplete_id = str(incomplete_req.id)

    try:
        res = client.post(
            f"/api/strategy-requests/{incomplete_id}/revisions/messages",
            json={"message": "Revise my strategy."},
        )
        assert res.status_code == 400
        assert "Strategy has not been generated yet" in res.json()["detail"]
    finally:
        with SessionLocal() as db:
            r = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(incomplete_id)).first()
            if r:
                db.delete(r)
                db.commit()


def test_multi_sentence_budget_clarification_paragraph_formatting(mock_strategy_session):
    """
    Test that a multi-sentence budget clarification response:
    1. Formats into distinct paragraphs separated by blank lines (\n\n).
    2. Separates introductory explanation, context, and follow-up question.
    3. Contains no bullet points or numbered lists.
    4. Preserves paragraph breaks in API response and database persistence.
    5. Leaves revision workflow and intent classification intact.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    clarification_reply = (
        "I can certainly help you adjust the budget allocation to give more focus to paid advertising.\n\n"
        "Your current plan allocates ₹60,000 evenly between direct LinkedIn outreach and local SEO. "
        "Increasing paid search or social advertising will require either reducing the allocation for local content "
        "or expanding your total monthly marketing investment to support the additional ad spend.\n\n"
        "Would you like to maintain your current ₹60,000 total budget and rebalance between channels, "
        "or would you like to set a higher overall budget to fund dedicated ad campaigns?"
    )

    llm_payload = json.dumps({
        "reply": clarification_reply,
        "intent_type": "new_request",
        "status": "clarifying",
        "normalized_intent": "Increase budget allocation for paid advertising",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [
            {
                "section_id": "budget_considerations",
                "section_name": "Budget Considerations & Allocation",
                "proposed_change": "Rebalance budget to prioritize paid advertising",
                "unchanged_aspects": "Overall business objectives remain unchanged"
            }
        ],
        "unchanged_facts": ["Current strategy has ₹60,000 budget"],
        "missing_information": ["Target budget amount or reallocation proportion"],
        "current_clarification_question": (
            "Would you like to maintain your current ₹60,000 total budget and rebalance between channels, "
            "or would you like to set a higher overall budget to fund dedicated ad campaigns?"
        ),
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": ["Ensure marketing spend aligns with corporate goals"]
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=llm_payload) as mock_llm:
        response = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "I want to put more of our budget into paid advertising."},
        )
        assert response.status_code == 200
        data = response.json()

        # 1. Verify prompt includes paragraph formatting and no-bullet-point guidelines
        called_args, called_kwargs = mock_llm.call_args
        prompt = called_kwargs.get("system_prompt", "")
        assert "REPLY FORMATTING AND PARAGRAPH STRUCTURE GUIDELINES" in prompt
        assert "distinct, readable paragraphs" in prompt
        assert "Do NOT use bullet points" in prompt

        # 2. Verify reply is returned with paragraph breaks preserved
        returned_reply = data["reply"]
        paragraphs = [p.strip() for p in returned_reply.split("\n\n") if p.strip()]
        assert len(paragraphs) == 3, f"Expected 3 distinct paragraphs, got {len(paragraphs)}: {paragraphs}"

        # 3. Verify introductory explanation, context, and follow-up question in separate paragraphs
        assert "adjust the budget allocation" in paragraphs[0]
        assert "current plan allocates ₹60,000" in paragraphs[1]
        assert "Would you like to maintain" in paragraphs[2]

        # 4. Verify no bullet points or numbered lists in conversational response
        for p in paragraphs:
            assert not p.startswith("*"), f"Paragraph should not start with bullet: {p}"
            assert not p.startswith("-"), f"Paragraph should not start with dash: {p}"
            assert not p.startswith("•"), f"Paragraph should not start with bullet: {p}"
            assert not p.startswith("1.") and not p.startswith("2."), f"Paragraph should not start with number: {p}"
            assert "•" not in p
            assert "\n*" not in p
            assert "\n-" not in p

        # 5. Verify database persistence preserves exact paragraph breaks
        with SessionLocal() as db:
            saved_msg = (
                db.query(RevisionMessage)
                .filter(
                    RevisionMessage.strategy_request_id == uuid.UUID(strat_req_id),
                    RevisionMessage.sender == "assistant",
                )
                .order_by(RevisionMessage.id.desc())
                .first()
            )
            assert saved_msg is not None
            assert saved_msg.content == clarification_reply
            assert "\n\n" in saved_msg.content

        # 6. Verify revision state and workflow remain intact
        assert data["revision_state"]["status"] == "clarifying"
        assert data["revision_state"]["is_ready_for_confirmation"] is False
        assert data["revision_state"]["affected_sections"] == ["budget_considerations"]

