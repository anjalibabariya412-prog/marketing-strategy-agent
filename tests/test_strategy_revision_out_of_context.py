import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app
from backend.app.db.session import SessionLocal
from backend.app.db.orm.strategy_request import StrategyRequest
from backend.app.db.orm.business import Business
from backend.app.db.orm.conversation import Conversation
from backend.app.db.orm.strategy import Strategy
from backend.app.schemas.revision import ProposedSectionChange, RevisionState
from backend.app.services.revision_service import (
    format_out_of_context_response,
    save_revision_state,
)

client = TestClient(app)

SAMPLE_STRATEGY_JSON = {
    "business_overview": ["EcoClean is a green commercial cleaning service in Mumbai."],
    "target_audience_insights": ["Corporate facility managers."],
    "competitive_positioning": ["Non-toxic eco-friendly positioning."],
    "value_proposition": ["Green office cleaning supporting ESG."],
    "marketing_channels_and_tactics": ["LinkedIn outreach and local SEO."],
    "customer_acquisition_approach": ["Free 1-week pilot green cleaning."],
    "budget_considerations": [
        {"label": "LinkedIn Outreach", "amount": "₹30,000", "percentage": "50%", "description": "Outreach"}
    ],
    "kpis": [{"metric": "Pilots", "target": "10 pilots/mo", "description": "Signed trials"}],
    "action_plan": ["Week 1: Audit accounts", "Week 2: Launch campaigns"]
}


@pytest.fixture
def mock_strategy_session():
    """
    Creates a temporary StrategyRequest, Business, Conversation, and Strategy in PostgreSQL.
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

        strat_req_id = str(strat_req.id)
        conv_id = str(conv.id)

    yield {
        "strategy_request_id": strat_req_id,
        "conversation_id": conv_id,
    }

    with SessionLocal() as db:
        from backend.app.db.orm.revision_message import RevisionMessage
        import uuid
        req_uuid = uuid.UUID(strat_req_id)
        db.query(RevisionMessage).filter(RevisionMessage.strategy_request_id == req_uuid).delete()
        c_uuid = uuid.UUID(conv_id)
        db.query(Strategy).filter(Strategy.conversation_id == c_uuid).delete()
        db.query(Conversation).filter(Conversation.id == c_uuid).delete()
        db.query(Business).filter(Business.strategy_request_id == req_uuid).delete()
        db.query(StrategyRequest).filter(StrategyRequest.id == req_uuid).delete()
        db.commit()


# ---------------------------------------------------------------------------
# Unit tests for format_out_of_context_response
# ---------------------------------------------------------------------------

def test_format_out_of_context_response_with_business_name():
    """Verify business name is inserted correctly into exact two-paragraph template."""
    res = format_out_of_context_response("Plum Goodness")
    expected = (
        "I'm not able to help with that — I'm here specifically to help you refine your Plum Goodness marketing plan.\n\n"
        "Is there anything in the plan you'd like to change or discuss?"
    )
    assert res == expected
    paragraphs = res.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0] == "I'm not able to help with that — I'm here specifically to help you refine your Plum Goodness marketing plan."
    assert paragraphs[1] == "Is there anything in the plan you'd like to change or discuss?"
    assert '"' not in res


def test_format_out_of_context_response_missing_business_name():
    """Verify missing/empty/N/A business name uses 'your marketing plan' without inventing a name."""
    for missing_val in [None, "", "   ", "N/A", "n/a", "None", "NULL"]:
        res = format_out_of_context_response(missing_val)
        expected = (
            "I'm not able to help with that — I'm here specifically to help you refine your marketing plan.\n\n"
            "Is there anything in the plan you'd like to change or discuss?"
        )
        assert res == expected
        paragraphs = res.split("\n\n")
        assert len(paragraphs) == 2
        assert paragraphs[0] == "I'm not able to help with that — I'm here specifically to help you refine your marketing plan."
        assert paragraphs[1] == "Is there anything in the plan you'd like to change or discuss?"


# ---------------------------------------------------------------------------
# API / Integration tests for revision chatbot out-of-context handling
# ---------------------------------------------------------------------------

def test_unrelated_weather_question_receives_exact_two_paragraph_response(mock_strategy_session):
    """
    1. An unrelated weather question receives the exact two-paragraph response with business name.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "I'm not able to help with that.",
        "intent_type": "out_of_context",
        "status": "in_progress",
        "normalized_intent": "Unrelated weather inquiry",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What is the weather like in Tokyo right now?"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        reply = data["reply"]

        expected_reply = (
            "I'm not able to help with that — I'm here specifically to help you refine your EcoClean Facility Solutions marketing plan.\n\n"
            "Is there anything in the plan you'd like to change or discuss?"
        )
        assert reply == expected_reply

        # Check two separate paragraphs with blank line
        parts = reply.split("\n\n")
        assert len(parts) == 2
        assert parts[0] == "I'm not able to help with that — I'm here specifically to help you refine your EcoClean Facility Solutions marketing plan."
        assert parts[1] == "Is there anything in the plan you'd like to change or discuss?"
        assert '"' not in reply


def test_missing_business_name_handled_without_inventing_name(mock_strategy_session):
    """
    3. Missing business name in database is handled without inventing a name.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Set business company_name to "N/A" to simulate missing/unavailable name
    with SessionLocal() as db:
        import uuid
        biz = db.query(Business).filter(Business.strategy_request_id == uuid.UUID(strat_req_id)).first()
        biz.company_name = "N/A"
        db.commit()

    mock_llm_json = json.dumps({
        "reply": "I cannot answer that.",
        "intent_type": "out_of_context",
        "status": "in_progress",
        "normalized_intent": "Unrelated question",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What's the capital of France?"}
        )
        assert res.status_code == 200
        reply = res.json()["reply"]

        expected_reply = (
            "I'm not able to help with that — I'm here specifically to help you refine your marketing plan.\n\n"
            "Is there anything in the plan you'd like to change or discuss?"
        )
        assert reply == expected_reply
        assert "your marketing plan" in reply


def test_none_business_object_uses_your_marketing_plan(mock_strategy_session):
    """Verify that when no business profile exists at all, 'your marketing plan' is used."""
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Delete business record completely
    with SessionLocal() as db:
        import uuid
        db.query(Business).filter(Business.strategy_request_id == uuid.UUID(strat_req_id)).delete()
        db.commit()

    mock_llm_json = json.dumps({
        "reply": "I cannot answer that.",
        "intent_type": "out_of_context",
        "status": "in_progress",
        "normalized_intent": "Unrelated question",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "How do I bake sourdough bread?"}
        )
        assert res.status_code == 200
        reply = res.json()["reply"]
        expected_reply = (
            "I'm not able to help with that — I'm here specifically to help you refine your marketing plan.\n\n"
            "Is there anything in the plan you'd like to change or discuss?"
        )
        assert reply == expected_reply


def test_valid_marketing_question_does_not_receive_out_of_context_response(mock_strategy_session):
    """
    4. A valid marketing-related question does not receive the out-of-context response.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "We chose LinkedIn as the primary channel because your business targets corporate facility heads who are most effectively reached via direct B2B messaging.",
        "intent_type": "inquiry",
        "status": "in_progress",
        "normalized_intent": "Inquiry about channel selection rationale",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": ["Product is green cleaning", "Target is corporate offices"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Why did you choose LinkedIn as the primary marketing channel?"}
        )
        assert res.status_code == 200
        reply = res.json()["reply"]
        assert "LinkedIn" in reply
        assert "I'm not able to help with that" not in reply


def test_valid_strategy_modification_continues_revision_workflow(mock_strategy_session):
    """
    5. A valid strategy modification request continues through the revision workflow.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    mock_llm_json = json.dumps({
        "reply": "Here is the proposed budget revision to ₹40,000. Please confirm if you'd like me to apply this proposal.",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Reduce budget to ₹40,000",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [
            {
                "section_id": "budget_considerations",
                "section_name": "Budget Considerations & Allocation",
                "proposed_change": "Reduce total budget to ₹40,000",
                "unchanged_aspects": "Channel allocation ratios"
            }
        ],
        "unchanged_facts": ["EcoClean in Mumbai"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Reduce my marketing budget to ₹40,000."}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        assert state["status"] == "proposal_ready"
        assert state["is_ready_for_confirmation"] is True
        assert "budget_considerations" in state["affected_sections"]


def test_pending_proposal_preserved_when_unrelated_message_arrives_and_can_be_confirmed_or_rejected(mock_strategy_session):
    """
    6. Pending proposal confirmation and rejection still work:
       - Pending proposal is NOT modified or reset when an unrelated message arrives.
       - The proposal can still be confirmed or rejected subsequently.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Step 1: Pre-populate a proposal_ready state in the database
    with SessionLocal() as db:
        import uuid
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        proposal_state = RevisionState(
            status="proposal_ready",
            is_ready_for_confirmation=True,
            confirmation_status="pending",
            normalized_intent="Update marketing channels to include Instagram",
            affected_sections=["marketing_channels_and_tactics"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="marketing_channels_and_tactics",
                    section_name="Marketing Channels & Tactics",
                    proposed_change="Add Instagram reels showcasing green cleaning",
                    unchanged_aspects="LinkedIn outreach remains active"
                )
            ],
            unchanged_facts=["Business is EcoClean", "Mumbai market"],
            missing_information=[],
            current_clarification_question=None,
            assumptions_and_constraints=[]
        )
        save_revision_state(db, strat_req, proposal_state)
        db.commit()

    # Step 2: User asks an unrelated question ("What's the weather like?")
    mock_weather_llm = json.dumps({
        "reply": "I don't know the weather.",
        "intent_type": "out_of_context",
        "status": "proposal_ready",
        "normalized_intent": "Update marketing channels to include Instagram",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_weather_llm):
        res1 = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What's the weather like in Mumbai today?"}
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert "I'm not able to help with that — I'm here specifically to help you refine your EcoClean Facility Solutions marketing plan." in data1["reply"]

        # Crucial: Verify pending proposal was preserved!
        state1 = data1["revision_state"]
        assert state1["status"] == "proposal_ready"
        assert state1["is_ready_for_confirmation"] is True
        assert state1["confirmation_status"] == "pending"
        assert state1["affected_sections"] == ["marketing_channels_and_tactics"]
        assert len(state1["proposed_changes"]) == 1
        assert state1["proposed_changes"][0]["section_id"] == "marketing_channels_and_tactics"

    # Step 3: User now confirms the proposal ("Yes, please confirm and apply this")
    mock_confirm_llm = json.dumps({
        "reply": "Great! Your proposal to add Instagram reels has been confirmed.",
        "intent_type": "confirmation",
        "status": "confirmed",
        "normalized_intent": "Update marketing channels to include Instagram",
        "affected_sections": ["marketing_channels_and_tactics"],
        "proposed_changes": [
            {
                "section_id": "marketing_channels_and_tactics",
                "section_name": "Marketing Channels & Tactics",
                "proposed_change": "Add Instagram reels showcasing green cleaning",
                "unchanged_aspects": "LinkedIn outreach remains active"
            }
        ],
        "unchanged_facts": ["Business is EcoClean", "Mumbai market"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "confirmed",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_confirm_llm):
        res2 = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Yes, please confirm and apply this proposal."}
        )
        assert res2.status_code == 200
        state2 = res2.json()["revision_state"]
        assert state2["status"] == "confirmed"
        assert state2["confirmation_status"] == "confirmed"
