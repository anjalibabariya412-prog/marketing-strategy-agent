import copy
import json
import os
import sys
import uuid
from unittest.mock import patch

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
from backend.app.schemas.agent.revision_state import ProposedSectionChange, RevisionState
from backend.app.services.revision_service import save_revision_state

client = TestClient(app)

SAMPLE_STRATEGY_JSON = {
    "business_overview": [
        "EcoClean is a green commercial cleaning service targeting corporate offices in Mumbai."
    ],
    "target_audience_insights": [
        "Facilities directors and administrative heads prioritizing ESG compliance."
    ],
    "competitive_positioning": [
        "Differentiated from traditional chemical cleaners via certified non-toxic biodegradable products."
    ],
    "value_proposition": [
        "Certified eco-friendly office cleaning that safeguards employee health and lowers corporate carbon footprints."
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
            "description": "Direct B2B outreach campaigns to corporate facilities heads."
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
# Phase 5 Tests: Display Updated Marketing Strategy in the Frontend & Next Cycle
# ==============================================================================

def test_original_strategy_fetches_correctly(mock_strategy_session):
    """
    Scenario 1: GET /strategy returns the original 9 standard sections with correct structure.
    """
    thread_id = mock_strategy_session["thread_id"]

    response = client.get(f"/strategy?thread_id={thread_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    strategy = data["strategy"]
    for sec_key in SAMPLE_STRATEGY_JSON.keys():
        assert sec_key in strategy, f"Missing section '{sec_key}' in fetched strategy"
        assert strategy[sec_key] is not None


def test_apply_revision_returns_complete_updated_strategy_payload(mock_strategy_session):
    """
    Scenario 2: POST .../revisions/apply returns the complete updated strategy payload,
    allowing the frontend to refresh state immediately without an extra request.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["competitive_positioning"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="competitive_positioning",
                    section_name="Competitive Positioning",
                    proposed_change="Emphasize GreenGuard Platinum indoor air quality certification.",
                    replacement_content=[
                        "Positioned as Mumbai's only commercial cleaner with GreenGuard Platinum indoor air certification."
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["application_status"] == "applied"
    updated_strat = data["strategy"]
    assert "GreenGuard Platinum" in updated_strat["competitive_positioning"][0]

    # All 9 sections must be present in the returned strategy payload
    for sec_key in SAMPLE_STRATEGY_JSON.keys():
        assert sec_key in updated_strat
        assert updated_strat[sec_key] is not None


def test_single_section_revision_does_not_cause_other_sections_to_disappear(mock_strategy_session):
    """
    Scenario 3: Updating 1 section keeps the other 8 sections intact in the returned strategy.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["value_proposition"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="value_proposition",
                    section_name="Value Proposition & Positioning",
                    proposed_change="Update value proposition statement",
                    replacement_content=[
                        "Zero toxic residue, 100% transparent sustainability reporting for top-tier Indian enterprises."
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200
    updated_strat = response.json()["strategy"]

    assert updated_strat["value_proposition"] == [
        "Zero toxic residue, 100% transparent sustainability reporting for top-tier Indian enterprises."
    ]

    # Verify every unaffected section matches the original
    for sec in ["business_overview", "target_audience_insights", "competitive_positioning",
                "marketing_channels_and_tactics", "customer_acquisition_approach",
                "budget_considerations", "kpis", "action_plan"]:
        assert updated_strat[sec] == SAMPLE_STRATEGY_JSON[sec]


def test_multisection_revision_updates_all_intended_sections(mock_strategy_session):
    """
    Scenario 4: A revision affecting multiple sections updates all of them correctly.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["marketing_channels_and_tactics", "budget_considerations"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="marketing_channels_and_tactics",
                    section_name="Marketing Channels & Tactics",
                    proposed_change="Add Google Search Ads for commercial cleaning keywords",
                    replacement_content=[
                        "Google Search Ads targeting 'office cleaning service Mumbai' and LinkedIn B2B campaigns."
                    ]
                ),
                ProposedSectionChange(
                    section_id="budget_considerations",
                    section_name="Budget Considerations & Allocation",
                    proposed_change="Reallocate budget: ₹35,000 Google Ads, ₹25,000 LinkedIn",
                    replacement_content=[
                        {
                            "label": "Google Search Ads",
                            "amount": "₹35,000",
                            "percentage": "58%",
                            "description": "High intent commercial search clicks."
                        },
                        {
                            "label": "LinkedIn InMail",
                            "amount": "₹25,000",
                            "percentage": "42%",
                            "description": "Direct outreach to facility managers."
                        }
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    response = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert response.status_code == 200
    strat = response.json()["strategy"]

    assert "Google Search Ads" in strat["marketing_channels_and_tactics"][0]
    assert strat["budget_considerations"][0]["label"] == "Google Search Ads"
    assert strat["budget_considerations"][0]["amount"] == "₹35,000"


def test_unpersisted_proposal_does_not_modify_saved_strategy(mock_strategy_session):
    """
    Scenario 5: An unpersisted proposal does not appear as the saved strategy.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    thread_id = mock_strategy_session["thread_id"]

    # Post an unconfirmed proposal
    mock_llm_json = json.dumps({
        "reply": "I propose we change the budget to ₹20,000. Do you confirm?",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Reduce budget",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [
            {
                "section_id": "budget_considerations",
                "section_name": "Budget Considerations & Allocation",
                "proposed_change": "Reduce budget to ₹20,000",
                "unchanged_aspects": "None"
            }
        ],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_llm_json):
        msg_res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Please reduce the budget to ₹20,000"}
        )
        assert msg_res.status_code == 200

    # Ensure GET /strategy still returns the original, unchanged strategy
    get_res = client.get(f"/strategy?thread_id={thread_id}")
    assert get_res.status_code == 200
    assert get_res.json()["strategy"]["budget_considerations"][0]["amount"] == "₹30,000"


def test_application_error_preserves_previously_displayed_strategy(mock_strategy_session):
    """
    Scenario 6: Application error preserves the previously saved strategy in DB.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    thread_id = mock_strategy_session["thread_id"]

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
                    proposed_change="Corrupt the business overview"
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    with patch("backend.app.services.revision_service.get_llm_response", side_effect=Exception("LLM timeout")):
        apply_res = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
        assert apply_res.status_code == 500

    # Confirm strategy remains intact
    get_res = client.get(f"/strategy?thread_id={thread_id}")
    assert get_res.status_code == 200
    assert get_res.json()["strategy"]["business_overview"] == SAMPLE_STRATEGY_JSON["business_overview"]


def test_subsequent_revision_uses_latest_saved_strategy_as_baseline(mock_strategy_session):
    """
    Scenario 7 & 8: After Revision 1 is applied, Revision 2 uses the updated strategy as baseline.
    The revision chatbot remains fully functional and conversational.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]
    conv_id = uuid.UUID(mock_strategy_session["thread_id"])
    thread_id = mock_strategy_session["thread_id"]

    # 1. Apply Revision 1: Update target_audience_insights
    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        state = RevisionState(
            status="confirmed",
            confirmation_status="confirmed",
            affected_sections=["target_audience_insights"],
            proposed_changes=[
                ProposedSectionChange(
                    section_id="target_audience_insights",
                    section_name="Target Audience & Insights",
                    proposed_change="Target IT and fintech startups in Mumbai BKC",
                    replacement_content=[
                        "Fast-growing IT and fintech startups in Mumbai BKC requiring rapid eco-cleaning onboarding."
                    ]
                )
            ]
        )
        save_revision_state(db, strat_req, state)
        db.commit()

    res1 = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
    assert res1.status_code == 200
    assert res1.json()["application_status"] == "applied"
    assert "IT and fintech" in res1.json()["strategy"]["target_audience_insights"][0]

    # 2. Check that GET /strategy now returns Revision 1 content
    get_res1 = client.get(f"/strategy?thread_id={thread_id}")
    assert get_res1.status_code == 200
    assert "IT and fintech" in get_res1.json()["strategy"]["target_audience_insights"][0]

    # 3. User requests a SECOND revision: "Now let's also update the KPIs"
    # When process_revision_message runs, it should format strategy_summary using Revision 1 as baseline
    mock_rev2_proposal = json.dumps({
        "reply": "I've prepared a new proposal to update the KPIs for your fintech target audience. Do you confirm?",
        "intent_type": "new_request",
        "status": "proposal_ready",
        "normalized_intent": "Update KPIs for IT and fintech startup accounts",
        "affected_sections": ["kpis"],
        "proposed_changes": [
            {
                "section_id": "kpis",
                "section_name": "KPIs & Success Metrics",
                "proposed_change": "Increase target to 25 fintech corporate pilot trials",
                "unchanged_aspects": "None"
            }
        ],
        "unchanged_facts": ["Audience remains IT and fintech startups in BKC"],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    captured_system_prompt = {}

    def mock_get_llm(*args, **kwargs):
        captured_system_prompt["prompt"] = kwargs.get("system_prompt", "")
        return mock_rev2_proposal

    with patch("backend.app.services.revision_service.get_llm_response", side_effect=mock_get_llm):
        msg_res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Now let's update our KPIs for these fintech startups to 25 pilots"}
        )
        assert msg_res.status_code == 200
        msg_data = msg_res.json()
        assert msg_data["revision_state"]["status"] == "proposal_ready"
        assert msg_data["revision_state"]["affected_sections"] == ["kpis"]

    # Verify that the system prompt passed to the LLM contained Revision 1's content as baseline!
    assert "IT and fintech startups in Mumbai BKC" in captured_system_prompt["prompt"], \
        "The subsequent revision did not use the updated strategy as baseline!"

    # 4. User confirms Revision 2
    mock_rev2_confirm = json.dumps({
        "reply": "Great, Revision 2 is confirmed! Ready to apply.",
        "intent_type": "confirmation",
        "status": "confirmed",
        "normalized_intent": "Update KPIs to 25 fintech pilots",
        "affected_sections": ["kpis"],
        "proposed_changes": [
            {
                "section_id": "kpis",
                "section_name": "KPIs & Success Metrics",
                "proposed_change": "Target 25 fintech pilots",
                "unchanged_aspects": "None"
            }
        ],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "confirmed",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_rev2_confirm):
        confirm_res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Yes, please confirm and apply Revision 2"}
        )
        assert confirm_res.status_code == 200
        assert confirm_res.json()["revision_state"]["confirmation_status"] == "confirmed"

    # 5. Apply Revision 2
    mock_rev2_apply = json.dumps({
        "kpis": [
            {
                "metric": "Fintech Startup Pilot Trials",
                "target": "25 pilots per month",
                "description": "Signed trials from BKC fintech enterprises."
            }
        ]
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=mock_rev2_apply):
        apply2_res = client.post(f"/api/strategy-requests/{strat_req_id}/revisions/apply")
        assert apply2_res.status_code == 200
        strat2 = apply2_res.json()["strategy"]

        # BOTH Revision 1 and Revision 2 are present in the final strategy!
        assert "IT and fintech" in strat2["target_audience_insights"][0]
        assert strat2["kpis"][0]["target"] == "25 pilots per month"
        # And unaffected sections remain intact
        assert strat2["business_overview"] == SAMPLE_STRATEGY_JSON["business_overview"]

    # 6. Verify that GET /strategy now returns the compound result of both revisions
    get_res2 = client.get(f"/strategy?thread_id={thread_id}")
    assert get_res2.status_code == 200
    final_strat = get_res2.json()["strategy"]
    assert "IT and fintech" in final_strat["target_audience_insights"][0]
    assert final_strat["kpis"][0]["target"] == "25 pilots per month"
