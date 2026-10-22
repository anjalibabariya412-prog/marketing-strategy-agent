import uuid
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.information_requirement import InformationRequirement, RequirementStatus
from backend.app.agent.gap_analysis import analyze_relevance
from backend.app.agent.question_selection import (
    select_next_requirement,
    generate_question,
    generate_first_contextual_message,
)
from backend.app.agent.answer_processing import (
    process_answer_for_requirement,
    process_answer_and_plan_next,
    extract_competitors,
)
from backend.app.agent.strategy_generation import generate_strategy
from backend.app.services.tavily_research_service import (
    analyze_competitor_with_llm,
    discover_competitors_with_tavily,
    classify_competitor_response,
    conduct_competitor_research,
)
from backend.app.services.llm_service import _persist_llm_usage
from backend.app.models.conversation import Conversation
from backend.app.models.strategy_request import StrategyRequest
from backend.app.models.llm_usage import LLMUsage


# ---------------------------------------------------------------------------
# 1. Gap Analysis ID Propagation
# ---------------------------------------------------------------------------

def test_gap_analysis_passes_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        requirements=[
            InformationRequirement(id="target_audience", title="Target Audience", description="Audience details")
        ],
    )

    mock_json = json.dumps({"results": [{"id": "target_audience", "relevance": "relevant"}]})
    with patch("backend.app.agent.gap_analysis.get_llm_response", return_value=mock_json) as mock_llm:
        analyze_relevance(state)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "relevance_analysis"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id


# ---------------------------------------------------------------------------
# 2. Question Selection ID Propagation
# ---------------------------------------------------------------------------

def test_question_selection_passes_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    req = InformationRequirement(id="target_audience", title="Target Audience", description="Audience details")
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        requirements=[req],
    )

    # select_next_requirement
    mock_select = json.dumps({"selected_requirement_id": "target_audience", "question": "Who is your audience?"})
    with patch("backend.app.agent.question_selection.get_llm_response", return_value=mock_select) as mock_llm:
        select_next_requirement(state)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "question_selection"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # generate_question
    with patch("backend.app.agent.question_selection.get_llm_response", return_value="Who is your target audience?") as mock_llm:
        q = generate_question(state, req)
        assert q == "Who is your target audience?"
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "question_generation"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # generate_first_contextual_message
    mock_first = json.dumps({
        "context_summary": "Summary of business",
        "selected_requirement_id": "target_audience",
        "question": "Who is your primary customer?"
    })
    with patch("backend.app.agent.question_selection.get_llm_response", return_value=mock_first) as mock_llm:
        generate_first_contextual_message(state)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "first_message_generation"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id


# ---------------------------------------------------------------------------
# 3. Answer Processing & Incidental Competitor ID Propagation
# ---------------------------------------------------------------------------

def test_answer_processing_passes_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    req = InformationRequirement(id="target_audience", title="Target Audience", description="Audience details")
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        requirements=[req],
        active_requirement_id="target_audience",
        current_question="Who is your target audience?",
    )

    mock_ans = json.dumps({
        "active_requirement": {"status": "known", "value": "Small businesses"},
        "incidentally_known_requirements": [],
    })
    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_ans) as mock_llm:
        process_answer_for_requirement(state, "We target small businesses.")
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "answer_classification"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # merged_turn_processing
    state.active_requirement_id = "target_audience"
    state.current_question = "Who is your target audience?"
    mock_merged = json.dumps({
        "active_requirement": {"status": "known", "value": "Small businesses"},
        "incidentally_known_requirements": [],
        "selected_requirement_id": None,
        "question": None,
    })
    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_merged) as mock_llm:
        process_answer_and_plan_next(state, "We target small businesses.")
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "merged_turn_processing"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id


def test_incidental_competitor_extraction_propagates_both_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    req_audience = InformationRequirement(id="target_audience", title="Target Audience", description="Audience details")
    req_comp = InformationRequirement(id="competitive_landscape", title="Competitors", description="Competitor details")
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        requirements=[req_audience, req_comp],
        active_requirement_id="target_audience",
        current_question="Who is your target audience?",
    )

    # User answers audience question but incidentally names competitor
    mock_ans = json.dumps({
        "active_requirement": {"status": "known", "value": "DTC shoppers"},
        "incidentally_known_requirements": [
            {"requirement_id": "competitive_landscape", "value": "Acme Brand and Rival Co"}
        ],
    })
    mock_comps = json.dumps({"competitors": ["Acme Brand", "Rival Co"]})

    calls = []
    def fake_get_llm(**kwargs):
        calls.append(kwargs)
        if kwargs.get("operation") == "answer_classification":
            return mock_ans
        if kwargs.get("operation") == "competitor_extraction":
            return mock_comps
        return "{}"

    with patch("backend.app.agent.answer_processing.get_llm_response", side_effect=fake_get_llm):
        process_answer_for_requirement(state, "We target DTC shoppers, competing with Acme Brand and Rival Co")

    assert len(calls) >= 2
    for c in calls:
        assert c["conversation_id"] == conv_id
        assert c["strategy_request_id"] == strat_id
    op_names = {c["operation"] for c in calls}
    assert "answer_classification" in op_names
    assert "competitor_extraction" in op_names


# ---------------------------------------------------------------------------
# 4. Tavily Competitor Research ID Propagation
# ---------------------------------------------------------------------------

def test_tavily_competitor_research_propagates_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        business_context=BusinessContext(
            company_name="Apex Fitness",
            product_or_service="Smart kettlebells",
            competitors=["IronGrip"],
        ),
    )

    # classify_competitor_response
    mock_intent = json.dumps({"intent": "provided_specific", "competitor_names": ["IronGrip"]})
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_intent) as mock_llm:
        classify_competitor_response("We compete with IronGrip", conversation_id=conv_id, strategy_request_id=strat_id)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "competitor_intent_classification"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # discover_competitors_with_tavily
    mock_discover = json.dumps({"discovered_competitors": ["KettlePro"]})
    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "KettlePro", "url": "https://kettlepro.com", "content": "Kettlebells"}]), \
         patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_discover) as mock_llm:
        discover_competitors_with_tavily(state, conversation_id=conv_id, strategy_request_id=strat_id)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "competitor_discovery"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # analyze_competitor_with_llm
    mock_analysis = json.dumps({
        "competitor_name": "IronGrip",
        "official_website": "https://irongrip.com",
        "positioning": "Heavy duty gym gear",
        "is_relevant_competitor": True,
        "evidence_quality": "high",
        "source_urls": ["https://irongrip.com"]
    })
    with patch("backend.app.services.tavily_research_service.get_llm_response", return_value=mock_analysis) as mock_llm:
        analyze_competitor_with_llm(
            competitor_name="IronGrip",
            search_results=[{"title": "IronGrip", "url": "https://irongrip.com", "content": "Weights"}],
            business_context_summary="Fitness brand",
            conversation_id=conv_id,
            strategy_request_id=strat_id,
        )
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "competitor_analysis"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id

    # conduct_competitor_research orchestrator propagates to all child calls
    calls = []
    def fake_get_llm(**kwargs):
        calls.append(kwargs)
        if kwargs.get("operation") == "competitor_intent_classification":
            return mock_intent
        if kwargs.get("operation") == "competitor_analysis":
            return mock_analysis
        return "{}"

    with patch("backend.app.services.tavily_research_service.execute_tavily_search", return_value=[{"title": "IronGrip", "url": "https://irongrip.com", "content": "Weights"}]), \
         patch("backend.app.services.tavily_research_service.get_llm_response", side_effect=fake_get_llm):
        conduct_competitor_research(state, latest_user_answer="We compete with IronGrip")

    assert len(calls) >= 1
    for c in calls:
        assert c["conversation_id"] == conv_id
        assert c["strategy_request_id"] == strat_id


# ---------------------------------------------------------------------------
# 5. Strategy Generation ID Propagation
# ---------------------------------------------------------------------------

def test_strategy_generation_passes_ids():
    strat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    state = MarketingAgentState(
        thread_id=conv_id,
        strategy_request_id=strat_id,
        business_context=BusinessContext(company_name="Apex Fitness"),
    )
    mock_strategy = json.dumps({
        "executive_summary": "Executive summary text",
        "positioning_strategy": "Positioning text",
        "strategic_pillars": [],
        "budget_allocation": [],
        "channel_recommendations": [],
        "implementation_phases": [],
    })
    with patch("backend.app.agent.strategy_generation.get_llm_response", return_value=mock_strategy) as mock_llm:
        generate_strategy(state)
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "strategy_generation"
        assert kwargs["conversation_id"] == conv_id
        assert kwargs["strategy_request_id"] == strat_id


# ---------------------------------------------------------------------------
# 6. Usage Persistence Resolution & Safety
# ---------------------------------------------------------------------------

def test_persist_llm_usage_resolves_strategy_request_from_conversation():
    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = conv_id
    mock_conv.strategy_request_id = strat_id

    mock_db = MagicMock()
    # Query for Conversation returns mock_conv
    mock_db.query.return_value.filter.return_value.first.return_value = mock_conv

    usage_info = {
        "model": "gpt-6-luna",
        "prompt_tokens": 100,
        "completion_tokens": 20,
    }

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        # Only conversation_id is passed; strategy_request_id is None
        _persist_llm_usage(
            operation="relevance_analysis",
            model="gpt-6-luna",
            provider="openai",
            usage_info=usage_info,
            conversation_id=conv_id,
            strategy_request_id=None,
        )

    added_obj = mock_db.add.call_args[0][0]
    assert added_obj.conversation_id == conv_id
    assert added_obj.strategy_request_id == strat_id


def test_persist_llm_usage_resolves_conversation_from_strategy_request():
    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = conv_id
    mock_conv.strategy_request_id = strat_id

    mock_db = MagicMock()
    # Query for Conversation returns mock_conv
    mock_db.query.return_value.filter.return_value.first.return_value = mock_conv

    usage_info = {
        "model": "gpt-6-luna",
        "prompt_tokens": 100,
        "completion_tokens": 20,
    }

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        # Only strategy_request_id is passed; conversation_id is None
        _persist_llm_usage(
            operation="strategy_generation",
            model="gpt-6-luna",
            provider="openai",
            usage_info=usage_info,
            conversation_id=None,
            strategy_request_id=strat_id,
        )

    added_obj = mock_db.add.call_args[0][0]
    assert added_obj.strategy_request_id == strat_id
    assert added_obj.conversation_id == conv_id


def test_persist_llm_usage_standalone_call_keeps_ids_null():
    mock_db = MagicMock()
    usage_info = {"model": "gpt-6-luna", "prompt_tokens": 50, "completion_tokens": 10}

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        _persist_llm_usage(
            operation="pdf_insight_extraction",
            model="gpt-6-luna",
            provider="openai",
            usage_info=usage_info,
            conversation_id=None,
            strategy_request_id=None,
        )

    added_obj = mock_db.add.call_args[0][0]
    assert added_obj.strategy_request_id is None
    assert added_obj.conversation_id is None


def test_persist_llm_usage_invalid_uuid_handled_safely():
    mock_db = MagicMock()
    usage_info = {"model": "gpt-6-luna", "prompt_tokens": 50, "completion_tokens": 10}

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        # Passing invalid non-UUID strings
        _persist_llm_usage(
            operation="relevance_analysis",
            model="gpt-6-luna",
            provider="openai",
            usage_info=usage_info,
            conversation_id="non-uuid-string-123",
            strategy_request_id="malformed-uuid-456",
        )

    added_obj = mock_db.add.call_args[0][0]
    # Invalid UUIDs must not crash and should be stored as NULL
    assert added_obj.strategy_request_id is None
    assert added_obj.conversation_id is None


# ---------------------------------------------------------------------------
# 7. LangGraph Thread ID to Conversation and State Resolution
# ---------------------------------------------------------------------------

def test_state_resolves_strategy_request_id_from_db():
    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    mock_conv = MagicMock()
    mock_conv.strategy_request_id = strat_id

    state = MarketingAgentState(thread_id=str(conv_id))
    assert state.strategy_request_id is None

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_conv

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        resolved = state.resolve_strategy_request_id()
        assert resolved == str(strat_id)
        assert state.strategy_request_id == str(strat_id)


# ---------------------------------------------------------------------------
# 8. PDF Extraction Standalone vs Associated
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_pdf_extraction_standalone_call_passes_none():
    from backend.app.api.pdf_extraction import extract_pdf_text
    from fastapi import UploadFile, Request
    import io

    mock_req = MagicMock(spec=Request)
    mock_req.query_params = {}
    pdf_bytes = b"%PDF-1.4 sample content with enough text to exceed minimum length character count requirement easily."
    upload_file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))

    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "document_type": "Marketing Plan",
        "relevance_reason": "Contains strategy",
        "executive_summary": "Summary",
    })

    mock_db = MagicMock()
    mock_db.__enter__.return_value = mock_db
    sample_extracted_text = "Sample marketing strategy document content that easily exceeds fifty characters in length for valid testing."

    with patch("backend.app.api.pdf_extraction.extract_and_clean_pdf", return_value=(sample_extracted_text, 120, 1)), \
         patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal", return_value=mock_db):
        await extract_pdf_text(
            request=mock_req,
            file=upload_file,
            strategy_request_id=None,
            conversation_id=None,
        )
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "pdf_insight_extraction"
        assert kwargs["strategy_request_id"] is None
        assert kwargs["conversation_id"] is None


@pytest.mark.anyio
async def test_pdf_extraction_associated_call_passes_ids():
    from backend.app.api.pdf_extraction import extract_pdf_text
    from fastapi import UploadFile, Request
    import io

    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    mock_req = MagicMock(spec=Request)
    mock_req.query_params = {}
    pdf_bytes = b"%PDF-1.4 sample content with enough text to exceed minimum length character count requirement easily."
    upload_file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))

    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "document_type": "Marketing Plan",
        "relevance_reason": "Contains strategy",
        "executive_summary": "Summary",
    })

    mock_db = MagicMock()
    mock_db.__enter__.return_value = mock_db
    sample_extracted_text = "Sample marketing strategy document content that easily exceeds fifty characters in length for valid testing."

    with patch("backend.app.api.pdf_extraction.extract_and_clean_pdf", return_value=(sample_extracted_text, 120, 1)), \
         patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal", return_value=mock_db):
        await extract_pdf_text(
            request=mock_req,
            file=upload_file,
            strategy_request_id=str(strat_id),
            conversation_id=str(conv_id),
        )
        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args[1]
        assert kwargs["operation"] == "pdf_insight_extraction"
        assert kwargs["strategy_request_id"] == strat_id
        assert kwargs["conversation_id"] == conv_id

