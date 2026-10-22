"""
Comprehensive unit and integration test suite for Background Competitor Research.

Covers:
- Immediate background research triggering after competitor answer.
- Non-blocking execution (next question is available without waiting).
- Independent state snapshot (caller state not concurrently mutated).
- Prevention of duplicate research jobs.
- Controlled synchronization and result merging before strategy generation.
- Graceful handling of research failure and timeout without crashing.
- Verification of both merged and non-merged answer processing paths.
- Programmatic verification that no frontend files are modified.
"""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.config import settings
from backend.app.core.requirements_library import load_requirements_into_state
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.business_context import BusinessContext
from backend.app.schemas.agent.competitor_analysis import (
    CompetitorAnalysisItem,
    CompetitorResearchReport,
)
from backend.app.schemas.agent.information_requirement import RequirementStatus
from backend.app.agent.answer_processing import (
    process_answer_for_requirement,
    process_answer_and_plan_next,
)
from backend.app.agent.graph import competitor_research_node, generate_node, ask_node
from backend.app.services.competitor_research_manager import (
    CompetitorResearchManager,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    competitor_research_manager,
    is_competitor_related_turn,
    is_useful_report,
    maybe_trigger_background_competitor_research,
)


@pytest.fixture(autouse=True)
def clean_manager():
    """Ensure clean job registry before and after every test."""
    with competitor_research_manager._lock:
        competitor_research_manager._jobs.clear()
    yield
    with competitor_research_manager._lock:
        competitor_research_manager._jobs.clear()


def test_1_is_competitor_related_turn():
    """Verify competitor detection across requirement IDs, keywords, and state."""
    state = MarketingAgentState(thread_id="test-thread-1")
    load_requirements_into_state(state)

    # 1. Active requirement is competitive_landscape
    assert is_competitor_related_turn("competitive_landscape", "some answer", state) is True
    assert is_competitor_related_turn("competitors", "some answer", state) is True
    assert is_competitor_related_turn("competitor_analysis", "some answer", state) is True

    # 2. Unrelated requirement without keywords
    assert is_competitor_related_turn("pricing_offer_structure", "₹500 per month", state) is False

    # 3. Unrelated requirement WITH competitor keywords or competitor URLs in answer
    assert is_competitor_related_turn("pricing_offer_structure", "Cheaper than our competitor BrandX", state) is True
    assert is_competitor_related_turn("target_market_location", "Check https://competitor.com", state) is True

    # 4. State already has competitors listed but not researched
    state.business_context.competitors = ["BrandAlpha"]
    assert is_competitor_related_turn("sales_conversion_journey", "Online checkout", state) is True


def test_2_research_starts_immediately_and_is_non_blocking():
    """
    Verify that background research starts immediately on competitor answer
    and does NOT block the main thread.
    """
    state = MarketingAgentState(thread_id="test-thread-non-blocking")
    state.business_context = BusinessContext(
        company_name="Aura Coffee",
        product_or_service="Specialty arabica coffee beans",
    )
    load_requirements_into_state(state)

    comp_req = state.get_requirement_by_id("competitive_landscape")
    assert comp_req is not None
    state.active_requirement_id = "competitive_landscape"
    state.current_question = "Who are your primary competitors?"

    # Simulate slow external search (takes 0.4s)
    def slow_research(state=None, latest_user_answer=None, strategy_request_id=None, **kwargs):
        time.sleep(0.4)
        if state is not None:
            if not state.competitor_research:
                state.competitor_research = CompetitorResearchReport()
            state.competitor_research.competitors.append(
                CompetitorAnalysisItem(competitor_name="Blue Tokai")
            )

    with patch("backend.app.services.competitor_research_manager.conduct_competitor_research", side_effect=slow_research):
        start_time = time.time()
        job = maybe_trigger_background_competitor_research(
            state=state,
            user_answer="Blue Tokai and Sleepy Owl",
            active_req_id="competitive_landscape",
        )
        elapsed = time.time() - start_time

        # Must return almost instantly (well under the 0.4s sleep)
        assert elapsed < 0.15, f"Expected non-blocking execution (<0.15s), took {elapsed:.3f}s"
        assert job is not None
        assert state.competitor_research_status == STATUS_RUNNING

        # Verify job is tracking in manager
        assert competitor_research_manager.get_status(state.thread_id) in (STATUS_RUNNING, STATUS_COMPLETED)

        # Wait for the background worker to finish and verify completion
        if job.future:
            job.future.result(timeout=2.0)
        assert job.status == STATUS_COMPLETED
        assert len(job.result.competitors) == 1
        assert job.result.competitors[0].competitor_name == "Blue Tokai"


def test_3_independent_state_snapshot():
    """
    Verify that background research operates on an independent snapshot copy
    and does not mutate the caller's state concurrently.
    """
    state = MarketingAgentState(thread_id="test-thread-snapshot")
    state.business_context = BusinessContext(
        company_name="SnapBrand",
        product_or_service="Smart home widgets",
        competitors=["CompOne"],
    )
    load_requirements_into_state(state)

    def background_mutator(state=None, latest_user_answer=None, strategy_request_id=None, **kwargs):
        # Mutate the snapshot passed to worker
        if state is not None:
            state.business_context.company_name = "MUTATED_BY_BACKGROUND"
            if not state.competitor_research:
                state.competitor_research = CompetitorResearchReport()
            state.competitor_research.competitors.append(
                CompetitorAnalysisItem(competitor_name="CompOne")
            )

    with patch("backend.app.services.competitor_research_manager.conduct_competitor_research", side_effect=background_mutator):
        job = competitor_research_manager.start_background_research(state, candidate_answer="CompOne")
        assert job is not None

        # Wait for worker completion
        if job.future:
            job.future.result(timeout=2.0)

        # The caller's company name must NOT be mutated by the worker!
        assert state.business_context.company_name == "SnapBrand"
        assert state.competitor_research is None  # Not yet merged


def test_4_duplicate_job_prevention():
    """
    Verify that multiple identical trigger calls do not spawn duplicate background jobs.
    """
    state = MarketingAgentState(thread_id="test-thread-duplicate")
    state.business_context = BusinessContext(
        company_name="DupTest",
        product_or_service="Cloud storage",
        competitors=["Dropbox"],
    )

    with patch("backend.app.services.competitor_research_manager.conduct_competitor_research") as mock_research:
        job1 = competitor_research_manager.start_background_research(state, candidate_answer="Dropbox")
        assert job1 is not None

        # Immediate second trigger with same inputs
        job2 = competitor_research_manager.start_background_research(state, candidate_answer="Dropbox")

        assert job1 is job2, "Expected duplicate start to return existing job instance"


def test_5_synchronization_and_merging_before_strategy_generation():
    """
    Verify that generate_node synchronizes and merges background research
    into state.competitor_research and state.business_context.competitors.
    """
    state = MarketingAgentState(thread_id="test-thread-sync")
    state.business_context = BusinessContext(
        company_name="TeaHaven",
        product_or_service="Organic Darjeeling tea",
        marketing_goal="Increase DTC sales",
    )
    load_requirements_into_state(state)

    # Pre-populate background job as completed
    report = CompetitorResearchReport(
        competitors=[
            CompetitorAnalysisItem(
                competitor_name="Vahdam Teas",
                positioning="Premium Indian tea export",
                strengths=["Global brand equity"],
            )
        ],
        overall_summary="Researched 1 competitor: Vahdam Teas.",
        researched_names=["vahdam teas"],
    )

    with competitor_research_manager._lock:
        competitor_research_manager._jobs[state.thread_id] = MagicMock(
            status=STATUS_COMPLETED,
            result=report,
            discovered_competitors=["Vahdam Teas"],
            future=None,
        )

    mock_strategy = MagicMock()
    with patch("backend.app.agent.graph.generate_strategy", return_value=mock_strategy) as mock_gen:
        result_state = generate_node(state)

        assert result_state.competitor_research is not None
        assert len(result_state.competitor_research.competitors) == 1
        assert result_state.competitor_research.competitors[0].competitor_name == "Vahdam Teas"
        assert result_state.competitor_research_status == STATUS_COMPLETED
        assert "Vahdam Teas" in result_state.business_context.competitors

        # Verify strategy generation received the state with competitor research
        called_state = mock_gen.call_args[0][0]
        assert called_state.competitor_research is not None
        assert called_state.final_strategy == mock_strategy


def test_6_research_timeout_does_not_crash_workflow():
    """
    Verify that if background research times out, generate_node continues
    gracefully with available evidence and does not raise an exception.
    """
    state = MarketingAgentState(thread_id="test-thread-timeout")
    state.business_context = BusinessContext(
        company_name="TimeoutCo",
        product_or_service="Analytics platform",
    )
    load_requirements_into_state(state)

    def never_finishes(s, latest_user_answer=None, strategy_request_id=None):
        time.sleep(2.0)

    mock_strategy = MagicMock()
    with patch("backend.app.services.competitor_research_manager.conduct_competitor_research", side_effect=never_finishes):
        competitor_research_manager.start_background_research(state, candidate_answer="SlowComp")

        # Synchronize with a very short timeout (0.05s)
        with patch("backend.app.agent.graph.generate_strategy", return_value=mock_strategy), \
             patch("backend.app.core.config.settings.competitor_research_timeout", 0.05):
            res_state = generate_node(state)

            assert res_state.competitor_research_status == STATUS_FAILED
            assert res_state.final_strategy == mock_strategy


def test_7_research_failure_does_not_crash_workflow():
    """
    Verify that if background research throws an unexpected exception,
    generate_node continues without crashing.
    """
    state = MarketingAgentState(thread_id="test-thread-failure")
    state.business_context = BusinessContext(
        company_name="FailCo",
        product_or_service="Cybersecurity tool",
    )
    load_requirements_into_state(state)

    def crashing_research(s, latest_user_answer=None, strategy_request_id=None):
        raise RuntimeError("External network connection dropped")

    mock_strategy = MagicMock()
    with patch("backend.app.services.competitor_research_manager.conduct_competitor_research", side_effect=crashing_research):
        job = competitor_research_manager.start_background_research(state, candidate_answer="FailComp")
        if job and job.future:
            # Let worker complete
            try:
                job.future.result(timeout=1.0)
            except Exception:
                pass

        with patch("backend.app.agent.graph.generate_strategy", return_value=mock_strategy):
            res_state = generate_node(state)
            assert res_state.competitor_research_status == STATUS_FAILED
            assert res_state.final_strategy == mock_strategy


def test_8_merged_turn_answer_processing_triggers_background_research():
    """
    Verify that process_answer_and_plan_next triggers background research
    when competitor requirement is answered.
    """
    state = MarketingAgentState(thread_id="test-thread-merged-turn")
    state.business_context = BusinessContext(
        company_name="Urban Clean",
        product_or_service="Eco-friendly dry cleaning",
    )
    load_requirements_into_state(state)
    state.active_requirement_id = "competitive_landscape"
    state.current_question = "Who are your competitors?"

    mock_llm_json = """{
        "active_requirement": {
            "status": "known",
            "value": "CleanCo and Sparkle Cleaners"
        },
        "incidentally_known_requirements": [],
        "next": {
            "action": "ask_predefined",
            "selected_id": "pricing_offer_structure",
            "question": "What is your pricing model?",
            "create_dynamic": null
        }
    }"""

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json), \
         patch("backend.app.services.competitor_research_manager.competitor_research_manager.start_background_research") as mock_start:
        process_answer_and_plan_next(state, "CleanCo and Sparkle Cleaners")

        assert mock_start.call_count == 1
        assert state.get_requirement_by_id("competitive_landscape").status == RequirementStatus.KNOWN
        assert state.pending_requirement_id == "pricing_offer_structure"
        assert state.pending_question == "What is your pricing model?"


def test_9_non_merged_answer_processing_triggers_background_research():
    """
    Verify that process_answer_for_requirement triggers background research
    when competitor requirement is answered in non-merged flow.
    """
    state = MarketingAgentState(thread_id="test-thread-non-merged")
    state.business_context = BusinessContext(
        company_name="FitPulse",
        product_or_service="Smart fitness tracker",
    )
    load_requirements_into_state(state)
    state.active_requirement_id = "competitive_landscape"
    state.current_question = "Who are your competitors?"

    mock_llm_json = """{
        "active_requirement": {
            "status": "known",
            "value": "Fitbit and Garmin"
        },
        "incidentally_known_requirements": []
    }"""

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json), \
         patch("backend.app.services.competitor_research_manager.competitor_research_manager.start_background_research") as mock_start:
        process_answer_for_requirement(state, "Fitbit and Garmin")

        assert mock_start.call_count == 1
        assert state.get_requirement_by_id("competitive_landscape").status == RequirementStatus.KNOWN


def test_10_no_frontend_files_modified():
    """
    Programmatically verify that NO files under frontend/ have been modified or staged.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout
    modified_frontend_files = [
        line.strip() for line in output.splitlines()
        if "frontend/" in line.lower()
    ]
    assert len(modified_frontend_files) == 0, f"Frontend files were modified: {modified_frontend_files}"
