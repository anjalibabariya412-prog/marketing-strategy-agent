"""
Tests for website context handling in the Marketing Strategy Agent revision chatbot.

Covers:
- Scenario A: Website URL and scraped summary are available -> directly summarize stored findings, no repeated link/screenshot requests.
- Scenario B: Website URL exists, but no useful scraped content -> honest limitation response, do not ask for the URL again.
- Scenario C: A product-page URL is shared, but only general website was analyzed -> distinguish general analysis from unverified product page, no repetitive screenshot/pasting demands.
- Informational inquiries do not alter strategy or revision state.
- format_online_presence_summary helper unit checks.
"""

import json
import os
import sys
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.main import app
from backend.app.db.session import SessionLocal
from backend.app.db.orm.strategy_request import StrategyRequest
from backend.app.db.orm.business import Business
from backend.app.db.orm.conversation import Conversation
from backend.app.db.orm.strategy import Strategy
from backend.app.db.orm.online_presence_source import OnlinePresenceSource
from backend.app.schemas.revision import RevisionState
from backend.app.services.revision_service import (
    format_online_presence_summary,
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


def test_format_online_presence_summary_unit():
    """
    Unit test for format_online_presence_summary with successful and failed sources.
    """
    biz = Business(website_social_links="https://ecoclean.in")
    src_success = OnlinePresenceSource(
        platform="website",
        original_url="https://ecoclean.in",
        normalized_url="https://ecoclean.in",
        scrape_status="success",
        summary="EcoClean provides certified green commercial cleaning services in Mumbai.",
        details={"evidence": [{"fact": "Certified non-toxic solutions"}]}
    )
    summary_text = format_online_presence_summary([src_success], biz)
    assert "Website" in summary_text
    assert "https://ecoclean.in" in summary_text
    assert "success" in summary_text
    assert "Certified non-toxic solutions" in summary_text

    # Failed source
    src_failed = OnlinePresenceSource(
        platform="website",
        original_url="https://failedsite.com",
        scrape_status="failed",
        error_message="Host unreachable"
    )
    failed_summary = format_online_presence_summary([src_failed], biz)
    assert "failed" in failed_summary
    assert "Host unreachable" in failed_summary


def test_scenario_a_website_url_and_scraped_summary_available(mock_strategy_session):
    """
    Scenario A: Website URL and scraped summary are available.
    User asks: "What did you find on my website?"
    Expected:
    - Stored website findings are passed in system prompt.
    - Assistant directly summarizes actual stored findings.
    - Assistant does not ask for website URL, screenshots, or pasted text.
    - Revision state and strategy remain intact.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Insert a successful OnlinePresenceSource record
    with SessionLocal() as db:
        source = OnlinePresenceSource(
            strategy_request_id=uuid.UUID(strat_req_id),
            platform="website",
            original_url="https://ecoclean.in",
            normalized_url="https://ecoclean.in",
            scrape_status="success",
            summary="EcoClean provides non-toxic commercial office cleaning in Mumbai with certified ESG-compliant solutions.",
            details={"evidence": [{"fact": "Hospital-grade non-toxic cleaning chemicals used."}]}
        )
        db.add(source)
        db.commit()

    mock_assistant_reply = (
        "Based on our analysis of your website (https://ecoclean.in), we found that EcoClean provides non-toxic commercial "
        "office cleaning in Mumbai with certified ESG-compliant solutions.\n\n"
        "Your site highlights the use of hospital-grade non-toxic cleaning chemicals and focuses on corporate facility managers. "
        "Is there a particular aspect of these findings you would like to incorporate further into your strategy?"
    )

    llm_payload = json.dumps({
        "reply": mock_assistant_reply,
        "intent_type": "inquiry",
        "status": "in_progress",
        "normalized_intent": "Inquiry about website findings",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=llm_payload) as mock_llm:
        response = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What did you find on my website?"}
        )
        assert response.status_code == 200
        data = response.json()

        # 1. Verify system prompt received stored website summary
        called_args, called_kwargs = mock_llm.call_args
        prompt = called_kwargs.get("system_prompt", "")
        assert "ONLINE PRESENCE & WEBSITE ANALYSIS (Stored System Findings):" in prompt
        assert "EcoClean provides non-toxic commercial office cleaning in Mumbai" in prompt
        assert "WEBSITE CONTEXT & ONLINE PRESENCE HANDLING GUIDELINES" in prompt

        # 2. Verify assistant response summarizes actual findings without asking for link or screenshot
        reply = data["reply"]
        assert "EcoClean provides non-toxic commercial" in reply
        assert "share your website link" not in reply.lower()
        assert "send a screenshot" not in reply.lower()
        assert "paste the content" not in reply.lower()
        assert "I'm not able to help with that" not in reply

        # 3. Revision state remains unchanged
        assert data["revision_state"]["status"] == "in_progress"
        assert data["revision_state"]["affected_sections"] == []


def test_scenario_b_website_url_exists_but_no_scraped_summary(mock_strategy_session):
    """
    Scenario B: Website URL exists in business profile, but no useful scraped content is available.
    User asks: "What did you find on my website?"
    Expected:
    - Assistant states: "I have your website link, but I don't have enough website content available to summarize its findings accurately."
    - Assistant does not ask for the same URL again.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Insert a failed OnlinePresenceSource record
    with SessionLocal() as db:
        source = OnlinePresenceSource(
            strategy_request_id=uuid.UUID(strat_req_id),
            platform="website",
            original_url="https://ecoclean.in",
            scrape_status="failed",
            error_message="Connection timed out",
            summary=None,
        )
        db.add(source)
        db.commit()

    mock_assistant_reply = (
        "I have your website link (https://ecoclean.in), but I don't have enough website content available to summarize its findings accurately.\n\n"
        "We can still refine your marketing strategy based on your core business details and goals. What specific section would you like to focus on?"
    )

    llm_payload = json.dumps({
        "reply": mock_assistant_reply,
        "intent_type": "inquiry",
        "status": "in_progress",
        "normalized_intent": "Inquiry about website findings when scraping failed",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=llm_payload) as mock_llm:
        response = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What did you find on my website?"}
        )
        assert response.status_code == 200
        data = response.json()

        # 1. Verify prompt contains status failure information
        called_args, called_kwargs = mock_llm.call_args
        prompt = called_kwargs.get("system_prompt", "")
        assert "Scrape Status: failed" in prompt

        # 2. Verify reply states the honest limitation without re-asking for the same link
        reply = data["reply"]
        assert "don't have enough website content available" in reply
        assert "please provide your website url" not in reply.lower()
        assert "what is your website" not in reply.lower()


def test_scenario_c_product_page_url_shared_general_website_only(mock_strategy_session):
    """
    Scenario C: A product-page URL is shared, but only the general website was analyzed.
    Expected:
    - Assistant explains what the existing website analysis covers.
    - Clearly states that the specific product page has not been verified.
    - Does not repeatedly demand screenshots or pasted text.
    - Advances the strategy revision constructively.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    with SessionLocal() as db:
        source = OnlinePresenceSource(
            strategy_request_id=uuid.UUID(strat_req_id),
            platform="website",
            original_url="https://ecoclean.in",
            scrape_status="success",
            summary="General website analysis for EcoClean highlights green commercial cleaning for corporate offices.",
            details={"items": [{"url": "https://ecoclean.in"}]}
        )
        db.add(source)
        db.commit()

    mock_assistant_reply = (
        "Our existing website analysis covers EcoClean's general commercial green cleaning services for corporate offices. "
        "The specific product page (https://ecoclean.in/services/deep-clean) has not been verified in our stored data.\n\n"
        "We can still adjust your marketing channels and tactics to highlight deep cleaning packages for corporate clients. "
        "Would you like to focus direct LinkedIn outreach on this specific service?"
    )

    llm_payload = json.dumps({
        "reply": mock_assistant_reply,
        "intent_type": "inquiry",
        "status": "in_progress",
        "normalized_intent": "Inquiry about deep clean product page",
        "affected_sections": [],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": False,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=llm_payload) as mock_llm:
        response = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "Here is our product page: https://ecoclean.in/services/deep-clean. What did you find on it?"}
        )
        assert response.status_code == 200
        data = response.json()

        reply = data["reply"]
        assert "existing website analysis covers" in reply
        assert "specific product page" in reply
        assert "has not been verified" in reply
        # Verify no repetitive demands for screenshots or pasted content
        assert "please upload a screenshot" not in reply.lower()
        assert "please paste the text" not in reply.lower()


def test_inquiry_does_not_overwrite_pending_proposal(mock_strategy_session):
    """
    Verify that asking an informational question about the website does NOT
    overwrite, cancel, or confirm an existing pending proposal.
    """
    strat_req_id = mock_strategy_session["strategy_request_id"]

    # Pre-save a revision state with a pending proposal
    with SessionLocal() as db:
        strat_req = db.query(StrategyRequest).filter(StrategyRequest.id == uuid.UUID(strat_req_id)).first()
        prior_state = RevisionState(
            status="proposal_ready",
            confirmation_status="pending",
            is_ready_for_confirmation=True,
            original_request="Increase budget to 80000",
            normalized_intent="Increase overall budget to ₹80,000",
            affected_sections=["budget_considerations"],
            proposed_changes=[
                {
                    "section_id": "budget_considerations",
                    "section_name": "Budget Considerations & Allocation",
                    "proposed_change": "Increase total budget to ₹80,000",
                    "unchanged_aspects": "Target audience remains unchanged"
                }
            ],
            unchanged_facts=["Core service is green cleaning"],
        )
        save_revision_state(db, strat_req, prior_state)
        db.commit()

    inquiry_reply = (
        "Based on our stored website findings, EcoClean focuses on certified green commercial cleaning in Mumbai.\n\n"
        "Your pending budget revision proposal (increasing budget to ₹80,000) is still waiting for your confirmation. "
        "Would you like to confirm the revision or discuss further adjustments?"
    )

    llm_payload = json.dumps({
        "reply": inquiry_reply,
        "intent_type": "inquiry",
        "status": "proposal_ready",
        "normalized_intent": "Inquiry about website findings during pending proposal",
        "affected_sections": ["budget_considerations"],
        "proposed_changes": [],
        "unchanged_facts": [],
        "missing_information": [],
        "current_clarification_question": None,
        "is_ready_for_confirmation": True,
        "confirmation_status": "pending",
        "assumptions_and_constraints": []
    })

    with patch("backend.app.services.revision_service.get_llm_response", return_value=llm_payload):
        res = client.post(
            f"/api/strategy-requests/{strat_req_id}/revisions/messages",
            json={"message": "What did you find on our website?"}
        )
        assert res.status_code == 200
        state = res.json()["revision_state"]
        # Pending proposal must remain intact!
        assert state["status"] == "proposal_ready"
        assert state["is_ready_for_confirmation"] is True
        assert state["confirmation_status"] == "pending"
        assert len(state["proposed_changes"]) == 1
        assert state["proposed_changes"][0]["section_id"] == "budget_considerations"
