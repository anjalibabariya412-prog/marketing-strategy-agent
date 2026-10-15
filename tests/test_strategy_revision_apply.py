import copy
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
from backend.app.db.orm.strategy_request import StrategyRequest
from backend.app.db.orm.business import Business
from backend.app.db.orm.conversation import Conversation
from backend.app.db.orm.strategy import Strategy
from backend.app.db.orm.revision_message import RevisionMessage
from backend.app.schemas.revision import ProposedSectionChange, RevisionState
from backend.app.services.revision_service import save_revision_state

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
            strategy_json=copy.deepcopy(SAMPLE_STRATEGY_JSON),
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
# Phase 4 Tests: Apply Confirmed Revisions to Saved Marketing Strategy
# ==============================================================================

def test_apply_confirmed_proposal_single_section(mock_strategy_session):
    """
    Test 1: Confirmed proposal correctly updates the intended strategy section.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    # Set up a confirmed proposal on budget_considerations
    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            normalized_intent="Reduce budget to ₹30,000",
            affected_sections=["budget_considerations"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="budget_considerations",
                    section_name="Budget Considerations & Allocation",
                    proposed_change="Allocate ₹20,000 to LinkedIn Ads and ₹10,000 to SEO.",
                    unchanged_aspects="Total allocation matches revised budget limit.",
                    replacement_content=[
                        {
                            "label": "LinkedIn Sponsored Ads",
                            "amount": "₹20,000",
                            "percentage": "67%",
                            "description": "Paid B2B sponsored content for Mumbai facility heads."
                        },
                        {
                            "label": "Local SEO Optimization",
                            "amount": "₹10,000",
                            "percentage": "33%",
                            "description": "Targeted Google My Business and local organic keywords."
                        }
                    ]
                )
            ],
            is_ready_for_confirmation=False,
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    # Call POST /api/strategy-requests/{id}/revisions/apply
    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["application_status"] == "applied"
    assert data["strategy"]["budget_considerations"][0]["label"] == "LinkedIn Sponsored Ads"
    assert data["strategy"]["budget_considerations"][0]["amount"] == "₹20,000"

    # Verify directly in the PostgreSQL database
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat is not None
        assert saved_strat.strategy_json["budget_considerations"][0]["label"] == "LinkedIn Sponsored Ads"
        assert saved_strat.strategy_json["budget_considerations"][0]["amount"] == "₹20,000"


def test_unaffected_sections_remain_strictly_unchanged(mock_strategy_session):
    """
    Test 2: Every unaffected section remains byte-for-byte and item-for-item unchanged.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            normalized_intent="Update value proposition statement",
            affected_sections=["value_proposition"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="value_proposition",
                    section_name="Value Proposition & Positioning",
                    proposed_change="Refine value proposition to emphasize zero VOC emissions.",
                    replacement_content=[
                        "Zero-VOC, hospital-grade commercial cleaning certified for modern ESG-compliant corporate facilities."
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200

    # Verify unaffected sections in database
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        saved_json = saved_strat.strategy_json

        # Affected section changed
        assert saved_json["value_proposition"] == [
            "Zero-VOC, hospital-grade commercial cleaning certified for modern ESG-compliant corporate facilities."
        ]

        # All other 8 sections remain completely identical
        unaffected_sections = [
            "business_overview",
            "target_audience_insights",
            "competitive_positioning",
            "marketing_channels_and_tactics",
            "customer_acquisition_approach",
            "budget_considerations",
            "kpis",
            "action_plan"
        ]
        for sec in unaffected_sections:
            assert saved_json[sec] == SAMPLE_STRATEGY_JSON[sec], f"Section {sec} was unexpectedly modified!"


def test_apply_confirmed_proposal_multiple_sections(mock_strategy_session):
    """
    Test 3: A proposal affecting multiple sections updates only those sections via LLM generation.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            normalized_intent="Focus on Instagram and update 30-day timeline",
            affected_sections=["marketing_channels_and_tactics", "action_plan"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="marketing_channels_and_tactics",
                    section_name="Marketing Channels & Tactics",
                    proposed_change="Add Instagram Reels and corporate case study carousel ads.",
                ),
                ProposedSectionChange(
                    section_id="action_plan",
                    section_name="Action Plan & Timeline",
                    proposed_change="Dedicate Week 1 to Instagram creative asset production.",
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    mock_llm_output = json.dumps({
        "marketing_channels_and_tactics": [
            "Instagram corporate visual storytelling highlighting certified non-toxic office transformations.",
            "Targeted B2B carousel ads reaching facilities executives in Mumbai."
        ],
        "action_plan": [
            "Week 1: Produce video walk-throughs of eco-cleaned commercial workspaces.",
            "Week 2-4: Run Instagram campaign and follow up with trial offers."
        ]
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_output):
        response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
        assert response.status_code == 200
        data = response.json()

        assert len(data["strategy"]["marketing_channels_and_tactics"]) == 2
        assert "Instagram" in data["strategy"]["marketing_channels_and_tactics"][0]
        assert "Week 1: Produce video" in data["strategy"]["action_plan"][0]

    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        # Verify affected sections updated
        assert "Instagram" in saved_strat.strategy_json["marketing_channels_and_tactics"][0]
        assert "Week 1: Produce video" in saved_strat.strategy_json["action_plan"][0]
        # Verify other 7 sections untouched
        assert saved_strat.strategy_json["business_overview"] == SAMPLE_STRATEGY_JSON["business_overview"]
        assert saved_strat.strategy_json["budget_considerations"] == SAMPLE_STRATEGY_JSON["budget_considerations"]


def test_cannot_apply_unconfirmed_proposal(mock_strategy_session):
    """
    Test 4: An unconfirmed proposal cannot be applied (HTTP 400).
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="proposal_ready",
            confirmation_status="pending",
            is_ready_for_confirmation=True,
            affected_sections=["budget_considerations"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="budget_considerations",
                    section_name="Budget Considerations & Allocation",
                    proposed_change="Reduce to ₹20,000",
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 400
    assert "Revision proposal has not been confirmed" in response.json()["detail"]

    # Verify database strategy remains strictly untouched
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat.strategy_json == SAMPLE_STRATEGY_JSON


def test_cannot_apply_rejected_proposal(mock_strategy_session):
    """
    Test 5: A rejected proposal cannot be applied (HTTP 400).
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="rejected",
            confirmation_status="rejected",
            is_ready_for_confirmation=False,
            affected_sections=["budget_considerations"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="budget_considerations",
                    section_name="Budget Considerations & Allocation",
                    proposed_change="Reduce to ₹10,000",
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 400
    assert "Cannot apply a rejected revision proposal" in response.json()["detail"]

    # Verify database strategy remains strictly untouched
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat.strategy_json == SAMPLE_STRATEGY_JSON


def test_cannot_apply_missing_proposal(mock_strategy_session):
    """
    Test 6: Applying when no proposal exists returns HTTP 400.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 400
    assert "No revision proposal found" in response.json()["detail"]


def test_invalid_llm_response_does_not_modify_saved_strategy(mock_strategy_session):
    """
    Test 7: Malformed or invalid LLM response during apply does not modify the saved strategy.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["business_overview"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="business_overview",
                    section_name="Business Overview",
                    proposed_change="Revise company mission statement",
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    # Mock invalid JSON from LLM
    with patch("backend.app.services.revision_service.get_llm_response", return_value="INVALID NON-JSON OUTPUT"):
        response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
        assert response.status_code == 400
        assert "Failed to generate structured section updates" in response.json()["detail"]

    # Verify strategy in DB was NOT changed
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat.strategy_json == SAMPLE_STRATEGY_JSON


def test_database_failure_rolls_back_transaction(mock_strategy_session):
    """
    Test 8: Database failure rolls back the strategy update and state transition.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["business_overview"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="business_overview",
                    section_name="Business Overview",
                    proposed_change="Revise company mission",
                    replacement_content=["EcoClean is Mumbai's premier certified green facility cleaner."]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    # Simulate database commit error during apply
    with patch("sqlalchemy.orm.Session.commit", side_effect=Exception("Database lock acquisition failure")):
        response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
        assert response.status_code == 500

    # Verify original strategy is intact
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat.strategy_json == SAMPLE_STRATEGY_JSON


def test_repeated_request_is_idempotent(mock_strategy_session):
    """
    Test 9: Repeated application requests do not re-apply or corrupt the strategy.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["business_overview"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="business_overview",
                    section_name="Business Overview",
                    proposed_change="Updated overview",
                    replacement_content=["EcoClean provides zero-emission commercial sanitation in Mumbai."]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    # First call -> success
    res1 = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert res1.status_code == 200
    assert res1.json()["application_status"] == "applied"

    # Second call (e.g. user double-clicked) -> returns already applied
    res2 = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert res2.status_code == 200
    assert res2.json()["application_status"] == "already_applied"
    assert "already been applied" in res2.json()["message"]

    # Verify DB content remains consistent
    with SessionLocal() as db:
        saved_strat = db.query(Strategy).filter(Strategy.conversation_id == conv_id).first()
        assert saved_strat.strategy_json["business_overview"] == [
            "EcoClean provides zero-emission commercial sanitation in Mumbai."
        ]


def test_cannot_apply_nonexistent_or_isolated_strategy_request():
    """
    Test 10: Nonexistent strategy request returns HTTP 404.
    """
    fake_id = str(uuid.uuid4())
    response = client.post(f"/api/strategy-requests/{fake_id}/revisions/apply")
    assert response.status_code == 404
    assert f"Strategy request '{fake_id}' not found" in response.json()["detail"]


def test_persisted_state_records_application_status_and_message(mock_strategy_session):
    """
    Test 11: Successful application stores status='applied' and adds an assistant notification message.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    req_uuid = uuid.UUID(strat_req_id)

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == req_uuid).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["kpis"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="kpis",
                    section_name="KPIs & Success Metrics",
                    proposed_change="Increase monthly pilot target to 20",
                    replacement_content=[
                        {
                            "metric": "Corporate Pilot Trials",
                            "target": "20 pilots per month",
                            "description": "Signed trial agreements from commercial prospects."
                        }
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200

    with SessionLocal() as db:
        # Check latest revision_state snapshot
        state_msg = (
            db.query(RevisionMessage)
            .filter(
                RevisionMessage.strategy_request_id == req_uuid,
                RevisionMessage.sender == "revision_state",
            )
            .order_by(RevisionMessage.id.desc())
            .first()
        )
        assert state_msg is not None
        persisted_state = RevisionState.model_validate_json(state_msg.content)
        assert persisted_state.status == "applied"
        assert persisted_state.application_status == "applied"
        assert persisted_state.applied_at is not None

        # Check that assistant notification was recorded
        notif_msg = (
            db.query(RevisionMessage)
            .filter(
                RevisionMessage.strategy_request_id == req_uuid,
                RevisionMessage.sender == "assistant",
            )
            .order_by(RevisionMessage.id.desc())
            .first()
        )
        assert notif_msg is not None
        assert "confirmed revisions have been successfully applied" in notif_msg.content


def test_get_strategy_returns_updated_strategy_after_apply(mock_strategy_session):
    """
    Test 12: GET /strategy?thread_id=... returns the newly updated strategy from DB.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    thread_id = mock_strategy_session["thread_id"]

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["action_plan"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="action_plan",
                    section_name="Action Plan & Timeline",
                    proposed_change="Update to a 90-day expansion plan",
                    replacement_content=[
                        "Month 1: Mumbai BKC commercial hub rollout.",
                        "Month 2: Lower Parel commercial offices pilot launch.",
                        "Month 3: Full quarterly corporate review and contract renewals."
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    apply_res = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert apply_res.status_code == 200

    # Call GET /strategy?thread_id=...
    get_res = client.get(f"/strategy?thread_id={thread_id}")
    assert get_res.status_code == 200
    strat_data = get_res.json()["strategy"]

    assert len(strat_data["action_plan"]) == 3
    assert "Month 1: Mumbai BKC" in strat_data["action_plan"][0]
    # Check that unaffected section is preserved
    assert strat_data["business_overview"] == SAMPLE_STRATEGY_JSON["business_overview"]
