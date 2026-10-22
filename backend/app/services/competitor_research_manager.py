"""
Competitor Research Background Manager.

Manages safe, isolated, and parallel execution of competitor web research.
- Spawns background worker thread immediately upon receiving competitor-related answers.
- Operates on an independent snapshot of MarketingAgentState (never mutates caller state concurrently).
- Prevents duplicate jobs for the same thread/inputs.
- Provides a controlled synchronization point before final strategy generation with configurable timeout.
- Safely recovers from failures and timeouts without halting the questionnaire or strategy generation.
"""

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union

from backend.app.core.config import settings
from backend.app.schemas.agent.agent_state import MarketingAgentState
from backend.app.schemas.agent.competitor_analysis import CompetitorResearchReport
from backend.app.services.tavily_research_service import conduct_competitor_research

logger = logging.getLogger(__name__)

# Research Status Constants
STATUS_NOT_STARTED = "not_started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def is_useful_report(report: Optional[CompetitorResearchReport]) -> bool:
    """
    Checks whether a CompetitorResearchReport contains actual useful findings.
    An empty CompetitorResearchReport object is not treated as proof of completion.
    """
    if not report:
        return False
    if report.competitors and len(report.competitors) > 0:
        return True
    if report.overall_summary and report.overall_summary.strip():
        return True
    if report.researched_names and len(report.researched_names) > 0:
        return True
    return False


def compute_input_fingerprint(state: MarketingAgentState, candidate_answer: Optional[str] = None) -> str:
    """
    Computes a deterministic hash fingerprint representing the competitor inputs for a job.
    Used for duplicate job prevention.
    """
    ctx = state.business_context
    comps = sorted([c.strip().lower() for c in (ctx.competitors or []) if isinstance(c, str)])
    ans_clean = (candidate_answer or "").strip().lower()
    raw = f"{ctx.company_name or ''}|{ctx.product_or_service or ''}|{','.join(comps)}|{ans_clean}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ResearchJob:
    thread_id: str
    strategy_request_id: Optional[str]
    status: str
    fingerprint: str
    candidate_answer: Optional[str]
    future: Optional[Future] = None
    result: Optional[CompetitorResearchReport] = None
    discovered_competitors: List[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class CompetitorResearchManager:
    """
    Thread-safe manager for background competitor research jobs.
    Maintains bounded worker threads and job registry isolated by thread_id.
    """

    def __init__(self, max_workers: Optional[int] = None):
        self._max_workers = max_workers or getattr(settings, "competitor_research_max_workers", 5)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="comp-research-",
        )
        self._jobs: Dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def get_job(self, key: Optional[str]) -> Optional[ResearchJob]:
        if not key:
            return None
        with self._lock:
            return self._jobs.get(str(key))

    def get_status(self, key: Optional[str]) -> str:
        job = self.get_job(key)
        if not job:
            return STATUS_NOT_STARTED
        return job.status

    def start_background_research(
        self,
        state: MarketingAgentState,
        candidate_answer: Optional[str] = None,
    ) -> Optional[ResearchJob]:
        """
        Starts competitor research in the background on an independent snapshot of state.
        Never blocks the caller and never mutates the caller's state directly from worker.
        Returns the ResearchJob (or existing job if already running/completed).
        """
        thread_id = state.thread_id or state.strategy_request_id or "default"
        fingerprint = compute_input_fingerprint(state, candidate_answer)
        strat_id = state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None)

        with self._lock:
            existing = self._jobs.get(str(thread_id))
            if existing:
                if existing.status == STATUS_RUNNING:
                    logger.info(f"CompetitorResearchManager: Research already running for thread '{thread_id}'. Skipping duplicate.")
                    state.competitor_research_status = STATUS_RUNNING
                    state.competitor_research_job_id = existing.fingerprint
                    return existing
                if existing.status == STATUS_COMPLETED and existing.fingerprint == fingerprint:
                    logger.info(f"CompetitorResearchManager: Completed research exists with matching fingerprint for thread '{thread_id}'.")
                    state.competitor_research_status = STATUS_COMPLETED
                    state.competitor_research_job_id = existing.fingerprint
                    return existing

            # Independent snapshot: deep copy so caller state is completely isolated
            snapshot_state = state.model_copy(deep=True)

            job = ResearchJob(
                thread_id=str(thread_id),
                strategy_request_id=str(strat_id) if strat_id else None,
                status=STATUS_RUNNING,
                fingerprint=fingerprint,
                candidate_answer=candidate_answer,
            )
            self._jobs[str(thread_id)] = job

            # Mark state status
            state.competitor_research_status = STATUS_RUNNING
            state.competitor_research_job_id = fingerprint

        logger.info(f"CompetitorResearchManager: Submitting background research job for thread '{thread_id}' (fingerprint: {fingerprint})")

        def worker_task():
            try:
                conduct_competitor_research(
                    state=snapshot_state,
                    latest_user_answer=candidate_answer,
                    strategy_request_id=strat_id,
                )
                report = snapshot_state.competitor_research or CompetitorResearchReport()
                discovered = list(snapshot_state.business_context.competitors or [])
                with self._lock:
                    job.status = STATUS_COMPLETED
                    job.result = report
                    job.discovered_competitors = discovered
                    job.completed_at = datetime.now(timezone.utc)
                logger.info(f"CompetitorResearchManager: Background research completed for thread '{thread_id}' with {len(report.competitors)} competitor(s).")
                return report, discovered
            except Exception as e:
                logger.error(f"CompetitorResearchManager: Background research failed for thread '{thread_id}': {e}")
                with self._lock:
                    job.status = STATUS_FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now(timezone.utc)
                return CompetitorResearchReport(), []

        future = self._executor.submit(worker_task)
        job.future = future
        return job

    def check_and_merge_completed(self, state: MarketingAgentState) -> bool:
        """
        Non-blocking check: if background research completed, merges the report into state immediately.
        Returns True if a completed report was merged.
        """
        thread_id = state.thread_id or state.strategy_request_id
        if not thread_id:
            return False

        with self._lock:
            job = self._jobs.get(str(thread_id))
            if not job or job.status != STATUS_COMPLETED or not job.result:
                return False

            if is_useful_report(state.competitor_research) and len(state.competitor_research.competitors) >= len(job.result.competitors):
                return False

            report = job.result
            discovered = job.discovered_competitors

        # Merge into authoritative state
        state.competitor_research = report
        state.competitor_research_status = STATUS_COMPLETED
        if discovered:
            current_lower = {c.lower() for c in (state.business_context.competitors or []) if isinstance(c, str)}
            for d in discovered:
                if d and d.lower() not in current_lower:
                    if not state.business_context.competitors:
                        state.business_context.competitors = []
                    state.business_context.competitors.append(d)
                    current_lower.add(d.lower())

        logger.info(f"CompetitorResearchManager: Merged completed research into state for thread '{thread_id}'.")
        return True

    def sync_and_merge(
        self,
        state: MarketingAgentState,
        timeout: Optional[float] = None,
    ) -> MarketingAgentState:
        """
        Controlled synchronization point before strategy generation.
        1. Checks whether background research was started.
        2. If running, waits only at this point with configurable timeout.
        3. If completed, merges the result into state.competitor_research.
        4. If failed or timed out, falls back safely and continues without crashing.
        5. If not started or missing, executes fallback conduct_competitor_research synchronously.
        """
        effective_timeout = timeout if timeout is not None else getattr(settings, "competitor_research_timeout", 30.0)
        thread_id = state.thread_id or state.strategy_request_id

        # 1. Check if state already has a useful completed report
        if is_useful_report(state.competitor_research):
            state.competitor_research_status = STATUS_COMPLETED
            logger.info("CompetitorResearchManager: State already has useful competitor research. Synchronization complete.")
            return state

        job = None
        if thread_id:
            with self._lock:
                job = self._jobs.get(str(thread_id))

        if job:
            if job.status == STATUS_RUNNING and job.future:
                logger.info(f"CompetitorResearchManager: Waiting on running background research for thread '{thread_id}' (timeout: {effective_timeout}s)...")
                try:
                    report, discovered = job.future.result(timeout=effective_timeout)
                    with self._lock:
                        job.status = STATUS_COMPLETED
                        job.result = report
                        job.discovered_competitors = discovered
                        job.completed_at = datetime.now(timezone.utc)
                except FutureTimeoutError:
                    logger.warning(f"CompetitorResearchManager: Background research timed out after {effective_timeout}s for thread '{thread_id}'. Continuing with available evidence.")
                    with self._lock:
                        job.status = STATUS_FAILED
                        job.error = "Timeout reached"
                    state.competitor_research_status = STATUS_FAILED
                    return state
                except Exception as e:
                    logger.error(f"CompetitorResearchManager: Background research task failed with exception: {e}")
                    with self._lock:
                        job.status = STATUS_FAILED
                        job.error = str(e)
                    state.competitor_research_status = STATUS_FAILED
                    return state

            if job.status == STATUS_COMPLETED and job.result:
                state.competitor_research = job.result
                state.competitor_research_status = STATUS_COMPLETED
                if job.discovered_competitors:
                    current_lower = {c.lower() for c in (state.business_context.competitors or []) if isinstance(c, str)}
                    for d in job.discovered_competitors:
                        if d and d.lower() not in current_lower:
                            if not state.business_context.competitors:
                                state.business_context.competitors = []
                            state.business_context.competitors.append(d)
                            current_lower.add(d.lower())
                logger.info(f"CompetitorResearchManager: Successfully merged research report into state for thread '{thread_id}'.")
                return state

            if job.status == STATUS_FAILED:
                logger.info(f"CompetitorResearchManager: Background research status is failed for thread '{thread_id}'. Continuing gracefully.")
                state.competitor_research_status = STATUS_FAILED
                return state

        # 2. Fallback if not started or job not present (e.g. process restart or no competitor turn)
        logger.info(f"CompetitorResearchManager: No background research job found for thread '{thread_id}'. Running synchronous fallback conduct_competitor_research.")
        try:
            latest_ans = None
            if state.qa_history:
                for turn in reversed(state.qa_history):
                    if turn.requirement_id in ("competitive_landscape", "competitors") or "competitor" in str(turn.requirement_id or "").lower():
                        latest_ans = turn.answer
                        break
            strat_id = state.resolve_strategy_request_id() if hasattr(state, "resolve_strategy_request_id") else getattr(state, "strategy_request_id", None)
            conduct_competitor_research(state, latest_user_answer=latest_ans, strategy_request_id=strat_id)
            if is_useful_report(state.competitor_research):
                state.competitor_research_status = STATUS_COMPLETED
            else:
                state.competitor_research_status = STATUS_FAILED
        except Exception as e:
            logger.error(f"CompetitorResearchManager: Synchronous fallback conduct_competitor_research failed: {e}")
            state.competitor_research_status = STATUS_FAILED

        return state


# Global singleton instance
competitor_research_manager = CompetitorResearchManager()


def is_competitor_related_turn(
    active_req_id: Optional[str],
    user_answer: str,
    state: MarketingAgentState,
) -> bool:
    """
    Determines whether a turn or answer is competitor-related and should initiate background research.
    """
    if active_req_id and (
        active_req_id in ("competitive_landscape", "competitors")
        or "competitor" in str(active_req_id).lower()
    ):
        return True

    lower_ans = (user_answer or "").lower()
    competitor_indicators = (
        "competitor", "compete", "rival", "alternative", "versus", " vs ", "vs.",
        "better than", "cheaper than", "compared to", "http://", "https://", ".com", ".in", ".co", ".org"
    )
    if any(ind in lower_ans for ind in competitor_indicators):
        return True

    if state.business_context and state.business_context.competitors:
        # If competitors exist but research has not started
        if state.competitor_research_status == STATUS_NOT_STARTED and not is_useful_report(state.competitor_research):
            return True

    return False


def maybe_trigger_background_competitor_research(
    state: MarketingAgentState,
    user_answer: Optional[str] = None,
    active_req_id: Optional[str] = None,
) -> Optional[ResearchJob]:
    """
    Checks if current turn/answer is competitor-related and starts background research if appropriate.
    Safe and non-blocking.
    """
    try:
        if is_competitor_related_turn(active_req_id, user_answer or "", state):
            return competitor_research_manager.start_background_research(
                state=state,
                candidate_answer=user_answer,
            )
    except Exception as e:
        logger.error(f"maybe_trigger_background_competitor_research encountered error: {e}")
    return None
