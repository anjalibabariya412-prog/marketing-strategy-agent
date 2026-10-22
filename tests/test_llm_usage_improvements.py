import io
import json
import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.llm_usage import LLMUsage
from backend.app.models.conversation import Conversation
from backend.app.services.llm_service import (
    get_llm_response,
    _persist_llm_usage,
    _sanitize_error_message,
    calculate_llm_costs,
    LLMServiceError,
)
from backend.app.agent.answer_processing import (
    extract_initial_context,
    extract_competitors,
)

client = TestClient(app)


def make_pdf_bytes(text: str) -> bytes:
    """Helper to generate minimal valid PDF bytes with given text."""
    pdf_str = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {len(text) + 27} >> stream\n"
        f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET\n"
        "endstream endobj\n"
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n0 6\ntrailer << /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    return pdf_str.encode("latin1")


# ---------------------------------------------------------------------------
# Step 1: Identifier Propagation Tests
# ---------------------------------------------------------------------------

def test_initial_context_extraction_propagates_identifiers():
    """Verify extract_initial_context passes conversation_id and strategy_request_id to get_llm_response."""
    conv_id = uuid.uuid4()
    strat_id = uuid.uuid4()
    mock_llm_json = json.dumps({
        "company_name": "Acme Clean",
        "product_or_service": "Cleaning",
        "marketing_goal": "Growth",
        "target_audience": "Homeowners",
    })

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json) as mock_get_llm:
        context = extract_initial_context(
            user_message="Hello, we are Acme Clean providing home cleaning services to grow our business.",
            conversation_id=conv_id,
            strategy_request_id=strat_id,
        )

        assert context.company_name == "Acme Clean"
        assert mock_get_llm.call_count >= 1

        # Check call args of initial_context_extraction
        first_call = mock_get_llm.call_args_list[0]
        assert first_call.kwargs.get("operation") == "initial_context_extraction"
        assert first_call.kwargs.get("conversation_id") == conv_id
        assert first_call.kwargs.get("strategy_request_id") == strat_id


def test_extract_competitors_propagates_identifiers():
    """Verify extract_competitors passes conversation_id and strategy_request_id."""
    conv_id = uuid.uuid4()
    strat_id = uuid.uuid4()
    mock_llm_json = json.dumps({"competitors": ["BrandA", "BrandB"]})

    with patch("backend.app.agent.answer_processing.get_llm_response", return_value=mock_llm_json) as mock_get_llm:
        comps = extract_competitors(
            user_answer="BrandA and BrandB are our rivals",
            conversation_id=conv_id,
            strategy_request_id=strat_id,
        )
        assert comps == ["BrandA", "BrandB"]
        call_kwargs = mock_get_llm.call_args.kwargs
        assert call_kwargs.get("operation") == "competitor_extraction"
        assert call_kwargs.get("conversation_id") == conv_id
        assert call_kwargs.get("strategy_request_id") == strat_id


def test_pdf_extraction_propagates_identifiers_from_form_data():
    """Verify PDF extraction passes strategy_request_id and conversation_id to get_llm_response from form data."""
    conv_id = uuid.uuid4()
    strat_id = uuid.uuid4()
    pdf_bytes = make_pdf_bytes("CleanSeas Ocean Cleanup Marketing Strategy Document 2026. " * 3)

    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "relevance_reason": "Marketing strategy document",
        "document_type": "Marketing Plan",
        "summary": "Marketing overview",
        "pdf_insights": None,
    })

    with patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal") as mock_session_cls:

        mock_db = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_db

        response = client.post(
            "/extract-pdf-text",
            data={
                "strategy_request_id": str(strat_id),
                "conversation_id": str(conv_id),
            },
            files={"file": ("strategy.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args.kwargs
        assert llm_kwargs.get("operation") == "pdf_insight_extraction"
        assert llm_kwargs.get("strategy_request_id") == strat_id
        assert llm_kwargs.get("conversation_id") == conv_id


def test_pdf_extraction_propagates_identifiers_from_query_params():
    """Verify PDF extraction passes strategy_request_id and conversation_id from query parameters."""
    conv_id = uuid.uuid4()
    strat_id = uuid.uuid4()
    pdf_bytes = make_pdf_bytes("CleanSeas Ocean Cleanup Marketing Strategy Document 2026. " * 3)

    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "relevance_reason": "Marketing strategy document",
        "document_type": "Marketing Plan",
        "summary": "Marketing overview",
        "pdf_insights": None,
    })

    with patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal") as mock_session_cls:

        mock_db = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_db

        response = client.post(
            f"/extract-pdf-text?strategy_request_id={strat_id}&conversation_id={conv_id}",
            files={"file": ("strategy.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args.kwargs
        assert llm_kwargs.get("operation") == "pdf_insight_extraction"
        assert llm_kwargs.get("strategy_request_id") == strat_id
        assert llm_kwargs.get("conversation_id") == conv_id


def test_pdf_extraction_resolves_strategy_request_id_from_conversation():
    """Verify PDF extraction resolves strategy_request_id from DB when only conversation_id is supplied."""
    conv_id = uuid.uuid4()
    resolved_strat_id = uuid.uuid4()
    pdf_bytes = make_pdf_bytes("CleanSeas Ocean Cleanup Marketing Strategy Document 2026. " * 3)

    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "relevance_reason": "Marketing strategy document",
        "document_type": "Marketing Plan",
        "summary": "Marketing overview",
        "pdf_insights": None,
    })

    with patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal") as mock_session_cls:

        mock_db = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_db
        mock_conv = MagicMock()
        mock_conv.id = conv_id
        mock_conv.strategy_request_id = resolved_strat_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_conv

        response = client.post(
            "/extract-pdf-text",
            data={"conversation_id": str(conv_id)},
            files={"file": ("strategy.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args.kwargs
        assert llm_kwargs.get("strategy_request_id") == resolved_strat_id
        assert llm_kwargs.get("conversation_id") == conv_id


def test_pdf_extraction_without_identifiers_persists_nullable():
    """Verify PDF extraction without identifiers does not fabricate UUIDs and passes None."""
    pdf_bytes = make_pdf_bytes("CleanSeas Ocean Cleanup Marketing Strategy Document 2026. " * 3)
    mock_llm_json = json.dumps({
        "is_marketing_relevant": True,
        "relevance_reason": "Marketing strategy document",
        "document_type": "Marketing Plan",
        "summary": "Marketing overview",
        "pdf_insights": None,
    })

    with patch("backend.app.api.pdf_extraction.get_llm_response", return_value=mock_llm_json) as mock_llm, \
         patch("backend.app.api.pdf_extraction.SessionLocal") as mock_session_cls:

        mock_db = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_db

        response = client.post(
            "/extract-pdf-text",
            files={"file": ("strategy.pdf", pdf_bytes, "application/pdf")},
        )

        assert response.status_code == 200
        mock_llm.assert_called_once()
        llm_kwargs = mock_llm.call_args.kwargs
        assert llm_kwargs.get("strategy_request_id") is None
        assert llm_kwargs.get("conversation_id") is None


# ---------------------------------------------------------------------------
# Step 2: Failure & Retry Tracking Tests
# ---------------------------------------------------------------------------

def test_failed_llm_call_tracks_failure_without_fabricating_tokens():
    """Verify failed LLM call records a failed usage attempt with status='failed' and None token counts."""
    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
         patch("backend.app.services.llm_service._persist_llm_usage") as mock_persist:

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Service Unavailable 503")

        with pytest.raises(LLMServiceError, match="Failed to get response from LLM"):
            get_llm_response(
                prompt="test prompt",
                operation="test_fail_op",
                strategy_request_id=uuid.uuid4(),
            )

        mock_persist.assert_called_once()
        call_kwargs = mock_persist.call_args.kwargs
        assert call_kwargs["status"] == "failed"
        assert call_kwargs["attempt_number"] == 1
        assert "RuntimeError" in call_kwargs["error_message"]
        # Ensure tokens and costs were not fabricated
        assert call_kwargs["usage_info"]["prompt_tokens"] is None
        assert call_kwargs["usage_info"]["completion_tokens"] is None
        assert call_kwargs["usage_info"]["total_tokens"] is None


def test_retries_record_each_attempt_separately():
    """Verify retries record each failed attempt and the final successful attempt separately."""
    mock_success = MagicMock()
    mock_success.model = "gpt-4o-mini"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = '{"result": "ok"}'
    mock_success.choices = [mock_choice]
    mock_success.usage.prompt_tokens = 50
    mock_success.usage.completion_tokens = 20
    mock_success.usage.total_tokens = 70

    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
         patch("backend.app.services.llm_service._persist_llm_usage") as mock_persist, \
         patch("time.sleep"):  # don't actually sleep in tests

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # Fail first attempt, succeed second attempt
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("Transient timeout"),
            mock_success,
        ]

        result = get_llm_response(
            prompt="retry test",
            operation="retry_op",
            max_retries=1,
        )

        assert result == '{"result": "ok"}'
        assert mock_persist.call_count == 2

        # First call was failed attempt 1
        first_call = mock_persist.call_args_list[0].kwargs
        assert first_call["status"] == "failed"
        assert first_call["attempt_number"] == 1
        assert "Transient timeout" in first_call["error_message"]

        # Second call was success attempt 2
        second_call = mock_persist.call_args_list[1].kwargs
        assert second_call["status"] == "success"
        assert second_call["attempt_number"] == 2
        assert second_call["error_message"] is None


def test_all_retries_fail_records_all_and_raises():
    """Verify when all retries fail, each attempt is recorded and LLMServiceError is raised."""
    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
         patch("backend.app.services.llm_service._persist_llm_usage") as mock_persist, \
         patch("time.sleep"):

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Persistent error")

        with pytest.raises(LLMServiceError, match="Persistent error"):
            get_llm_response(
                prompt="test prompt",
                operation="fail_all_op",
                max_retries=2,
            )

        assert mock_persist.call_count == 3
        for idx, call in enumerate(mock_persist.call_args_list, start=1):
            assert call.kwargs["status"] == "failed"
            assert call.kwargs["attempt_number"] == idx


def test_sanitize_error_message_scrubs_secrets():
    """Verify _sanitize_error_message removes API keys and tokens from error messages."""
    raw_error = "OpenAIError: Invalid API key sk-proj-1234567890abcdef1234567890 with Bearer secret_token_xyz"
    sanitized = _sanitize_error_message(raw_error)
    assert "sk-proj-1234567890abcdef1234567890" not in sanitized
    assert "secret_token_xyz" not in sanitized
    assert "REDACTED" in sanitized


# ---------------------------------------------------------------------------
# Step 3: Observability, Transaction Safety & Recovery Tests
# ---------------------------------------------------------------------------

def test_persistence_failure_logs_and_does_not_mask_successful_llm_response(caplog):
    """Verify usage persistence DB failure is logged and does not mask successful response."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "user answer"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with caplog.at_level(logging.ERROR):
        with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
             patch("backend.app.db.session.SessionLocal", side_effect=RuntimeError("Database connection lost")):

            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            res = get_llm_response(prompt="hello")

    assert res == "user answer"
    assert any("Failed to persist LLM usage record" in r.message for r in caplog.records)


def test_persistence_failure_does_not_mask_original_llm_api_exception(caplog):
    """Verify if both LLM API and DB fail, the original LLMServiceError is still raised."""
    with caplog.at_level(logging.ERROR):
        with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
             patch("backend.app.db.session.SessionLocal", side_effect=RuntimeError("DB is down")):

            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI rate limit exceeded")

            with pytest.raises(LLMServiceError, match="OpenAI rate limit exceeded"):
                get_llm_response(prompt="hello")

    # DB error was logged, but LLM error was raised to caller
    assert any("Failed to persist LLM" in r.message for r in caplog.records)


def test_persist_llm_usage_rolls_back_on_commit_error():
    """Verify _persist_llm_usage rolls back session when commit encounters an error."""
    mock_db = MagicMock()
    mock_db.commit.side_effect = RuntimeError("Commit failed")

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        res = _persist_llm_usage(
            operation="test_rb",
            model="gpt-4o-mini",
            provider="openai",
            usage_info={"prompt_tokens": 10},
            max_db_retries=0,
        )

    assert res is None
    mock_db.rollback.assert_called()


def test_persist_llm_usage_bounded_retry_recovers():
    """Verify transient DB error on attempt 1 recovers on attempt 2."""
    mock_db = MagicMock()
    # On attempt 2 idempotency check, report no existing record
    mock_db.query.return_value.filter.return_value.first.return_value = None
    # First attempt commit fails, second attempt succeeds
    mock_db.commit.side_effect = [RuntimeError("Transient DB lock"), None]

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db), \
         patch("time.sleep"):

        mock_db.__enter__.return_value = mock_db
        res = _persist_llm_usage(
            operation="test_recovery",
            model="gpt-4o-mini",
            provider="openai",
            usage_info={"prompt_tokens": 10},
            max_db_retries=1,
        )

    assert res is not None
    assert mock_db.commit.call_count == 2


def test_persist_llm_usage_idempotency_prevents_duplicate():
    """Verify that if record was already persisted, retry returns existing id without duplicate add/commit."""
    mock_db = MagicMock()
    existing_id = uuid.uuid4()
    mock_existing = MagicMock()
    mock_existing.id = existing_id
    mock_db.query.return_value.filter.return_value.first.return_value = mock_existing

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        res = _persist_llm_usage(
            operation="test_idempotent",
            model="gpt-4o-mini",
            provider="openai",
            usage_info={"prompt_tokens": 10},
            record_id=existing_id,
        )

    assert res == existing_id
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()
