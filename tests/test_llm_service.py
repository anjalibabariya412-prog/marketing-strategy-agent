import logging
import uuid
from unittest.mock import MagicMock, patch
import pytest

from backend.app.services.llm_service import (
    get_llm_response,
    _extract_usage_metadata,
    _persist_llm_usage,
    LLMServiceError,
)
from backend.app.db.orm.llm_usage import LLMUsage


def test_extract_usage_metadata_with_object():
    """Verify usage metadata is correctly extracted from object-like response."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini-2024-07-18"
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 45
    mock_response.usage.total_tokens = 165

    result = _extract_usage_metadata(mock_response, fallback_model="gpt-4o-mini")
    assert result == {
        "model": "gpt-4o-mini-2024-07-18",
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
    }


def test_extract_usage_metadata_with_dict():
    """Verify usage metadata is correctly extracted from dictionary response."""
    dict_response = {
        "model": "gpt-4o-2024-08-06",
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
        },
    }

    result = _extract_usage_metadata(dict_response, fallback_model="gpt-4o")
    assert result == {
        "model": "gpt-4o-2024-08-06",
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "total_tokens": 280,
    }


def test_extract_usage_metadata_missing_usage_safe():
    """Verify missing usage is handled safely without estimating from character length."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = None

    result = _extract_usage_metadata(mock_response, fallback_model="gpt-4o-mini")
    assert result == {
        "model": "gpt-4o-mini",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_extract_usage_metadata_missing_model_uses_fallback():
    """Verify fallback model is used if response.model is unavailable."""
    mock_response = MagicMock()
    mock_response.model = None
    mock_response.usage = None

    result = _extract_usage_metadata(mock_response, fallback_model="gpt-4o-mini")
    assert result["model"] == "gpt-4o-mini"


def test_get_llm_response_logs_token_usage_and_returns_text(caplog):
    """Verify get_llm_response returns cleaned string and logs token usage."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini-2024-07-18"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "```json\n{\"status\": \"ok\"}\n```"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 180

    with caplog.at_level(logging.INFO):
        with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
             patch("backend.app.services.llm_service._persist_llm_usage") as mock_persist:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            result = get_llm_response(
                prompt="test prompt",
                system_prompt="test system prompt",
                model="gpt-4o-mini",
                operation="test_op",
            )

    assert result == '{"status": "ok"}'
    assert isinstance(result, str)
    assert any(
        "LLM token usage: model=gpt-4o-mini-2024-07-18 | prompt_tokens=150 | completion_tokens=30 | total_tokens=180"
        in record.message
        for record in caplog.records
    )
    mock_persist.assert_called_once()


def test_get_llm_response_handles_missing_usage_metadata_safely(caplog):
    """Verify missing usage metadata does not crash and is logged as unavailable."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "plain text response"
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    with caplog.at_level(logging.INFO):
        with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
             patch("backend.app.services.llm_service._persist_llm_usage") as mock_persist:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            result = get_llm_response(prompt="hello", operation="test_op")

    assert result == "plain text response"
    assert isinstance(result, str)
    assert any(
        "usage metadata unavailable" in record.message
        for record in caplog.records
    )
    mock_persist.assert_called_once()
    call_kwargs = mock_persist.call_args[1]
    assert call_kwargs["usage_info"]["prompt_tokens"] is None
    assert call_kwargs["usage_info"]["completion_tokens"] is None
    assert call_kwargs["usage_info"]["total_tokens"] is None


def test_get_llm_response_raises_llm_service_error_on_failure():
    """Verify exception handling raises LLMServiceError."""
    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timeout")

        with pytest.raises(LLMServiceError, match="Failed to get response from LLM"):
            get_llm_response(prompt="fail")


def test_persist_llm_usage_success():
    """Verify _persist_llm_usage saves a record to DB with proper fields."""
    mock_db = MagicMock()
    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    usage_info = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 150,
        "completion_tokens": 40,
        "total_tokens": 190,
    }

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        record_id = _persist_llm_usage(
            operation="relevance_analysis",
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage_info,
            duration_ms=350,
            strategy_request_id=strat_id,
            conversation_id=conv_id,
        )

    mock_db.add.assert_called_once()
    added_obj = mock_db.add.call_args[0][0]
    assert isinstance(added_obj, LLMUsage)
    assert added_obj.strategy_request_id == strat_id
    assert added_obj.conversation_id == conv_id
    assert added_obj.operation == "relevance_analysis"
    assert added_obj.provider == "openai"
    assert added_obj.model == "gpt-4o-mini"
    assert added_obj.input_tokens == 150
    assert added_obj.output_tokens == 40
    assert added_obj.total_tokens == 190
    assert added_obj.status == "success"
    assert added_obj.duration_ms == 350
    mock_db.commit.assert_called_once()


def test_persist_llm_usage_lookup_strategy_request_from_conversation():
    """Verify _persist_llm_usage resolves strategy_request_id when conversation_id is provided."""
    mock_db = MagicMock()
    conv_id = uuid.uuid4()
    resolved_strat_id = uuid.uuid4()

    mock_conv = MagicMock()
    mock_conv.id = conv_id
    mock_conv.strategy_request_id = resolved_strat_id
    mock_db.query.return_value.filter.return_value.first.return_value = mock_conv

    usage_info = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
    }

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        _persist_llm_usage(
            operation="question_generation",
            model="gpt-4o-mini",
            provider="openai",
            usage_info=usage_info,
            duration_ms=210,
            strategy_request_id=None,
            conversation_id=conv_id,
        )

    added_obj = mock_db.add.call_args[0][0]
    assert added_obj.conversation_id == conv_id
    assert added_obj.strategy_request_id == resolved_strat_id


def test_get_llm_response_database_write_failure_does_not_fail_request(caplog):
    """Verify that a database failure in _persist_llm_usage does not fail the LLM request."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "working response text"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150

    with caplog.at_level(logging.ERROR):
        with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
             patch("backend.app.db.session.SessionLocal", side_effect=RuntimeError("Database down")):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            # Calling get_llm_response should NOT raise an error despite DB failure
            result = get_llm_response(
                prompt="test prompt",
                operation="strategy_generation",
            )

    assert result == "working response text"
    assert isinstance(result, str)
    assert any("Failed to persist LLM usage record to database" in record.message for record in caplog.records)
