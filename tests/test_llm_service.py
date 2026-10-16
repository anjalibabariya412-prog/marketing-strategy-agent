from decimal import Decimal
import logging
import uuid
from unittest.mock import MagicMock, patch
import pytest

from backend.app.services.llm_service import (
    get_llm_response,
    _extract_usage_metadata,
    _persist_llm_usage,
    calculate_llm_costs,
    LLMServiceError,
)
from backend.app.models.llm_usage import LLMUsage


def test_extract_usage_metadata_with_object():
    """Verify usage metadata is correctly extracted from object-like response with cached tokens."""
    mock_response = MagicMock(spec=["model", "usage"])
    mock_response.model = "gpt-4o-mini-2024-07-18"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 120
    mock_usage.completion_tokens = 45
    mock_usage.total_tokens = 165
    mock_usage.prompt_tokens_details.cached_tokens = 50
    mock_response.usage = mock_usage

    result = _extract_usage_metadata(mock_response, fallback_model="gpt-4o-mini")
    assert result == {
        "model": "gpt-4o-mini-2024-07-18",
        "prompt_tokens": 120,
        "cached_tokens": 50,
        "cache_write_tokens": None,
        "completion_tokens": 45,
        "total_tokens": 165,
    }


def test_extract_usage_metadata_with_dict():
    """Verify usage metadata is correctly extracted from dictionary response with cached tokens."""
    dict_response = {
        "model": "gpt-4o-2024-08-06",
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
            "prompt_tokens_details": {
                "cached_tokens": 60,
            },
        },
    }

    result = _extract_usage_metadata(dict_response, fallback_model="gpt-4o")
    assert result == {
        "model": "gpt-4o-2024-08-06",
        "prompt_tokens": 200,
        "cached_tokens": 60,
        "cache_write_tokens": None,
        "completion_tokens": 80,
        "total_tokens": 280,
    }


def test_extract_usage_metadata_with_cache_write_tokens_object():
    """Verify cache_write_tokens is extracted from object-like response."""
    mock_response = MagicMock(spec=["model", "usage"])
    mock_response.model = "gpt-6-luna"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 500
    mock_usage.completion_tokens = 100
    mock_usage.total_tokens = 600
    mock_usage.prompt_tokens_details.cached_tokens = 200
    mock_usage.prompt_tokens_details.cache_write_tokens = 150
    mock_response.usage = mock_usage

    result = _extract_usage_metadata(mock_response, fallback_model="gpt-6-luna")
    assert result == {
        "model": "gpt-6-luna",
        "prompt_tokens": 500,
        "cached_tokens": 200,
        "cache_write_tokens": 150,
        "completion_tokens": 100,
        "total_tokens": 600,
    }


def test_extract_usage_metadata_with_cache_write_tokens_dict():
    """Verify cache_write_tokens is extracted from dict-like response."""
    dict_response = {
        "model": "gpt-6-luna",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 250,
            "total_tokens": 1250,
            "prompt_tokens_details": {
                "cached_tokens": 400,
                "cache_write_tokens": 300,
            },
        },
    }

    result = _extract_usage_metadata(dict_response)
    assert result == {
        "model": "gpt-6-luna",
        "prompt_tokens": 1000,
        "cached_tokens": 400,
        "cache_write_tokens": 300,
        "completion_tokens": 250,
        "total_tokens": 1250,
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
        "cached_tokens": None,
        "cache_write_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_extract_usage_metadata_missing_cached_tokens():
    """Verify missing prompt_tokens_details safely defaults cached_tokens and cache_write_tokens to None."""
    dict_response = {
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    result = _extract_usage_metadata(dict_response)
    assert result == {
        "model": "gpt-4o-mini",
        "prompt_tokens": 100,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "completion_tokens": 50,
        "total_tokens": 150,
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
    mock_response = MagicMock(spec=["model", "usage", "choices"])
    mock_response.model = "gpt-4o-mini-2024-07-18"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "```json\n{\"status\": \"ok\"}\n```"
    mock_response.choices = [mock_choice]
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 150
    mock_usage.completion_tokens = 30
    mock_usage.total_tokens = 180
    mock_usage.prompt_tokens_details.cached_tokens = 25
    mock_response.usage = mock_usage

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
        "LLM token usage: model=gpt-4o-mini-2024-07-18 | prompt_tokens=150 (cached=25) | completion_tokens=30 | total_tokens=180"
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


def test_get_llm_response_raises_llm_service_error_on_api_failure():
    """Verify LLMServiceError is raised when OpenAI client fails."""
    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection timeout")

        with pytest.raises(LLMServiceError, match="Failed to get response from LLM"):
            get_llm_response(prompt="fail")


def test_persist_llm_usage_success():
    """Verify _persist_llm_usage saves a record to DB with proper fields including cached_input_tokens and calculated costs."""
    mock_db = MagicMock()
    strat_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    usage_info = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 150,
        "cached_tokens": 30,
        "cache_write_tokens": None,
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
    assert added_obj.cached_input_tokens == 30
    assert added_obj.cache_write_tokens is None
    assert added_obj.output_tokens == 40
    assert added_obj.duration_ms == 350
    # Costs: fresh = 150 - 30 = 120 tokens @ $0.10/1M = 0.000012
    assert added_obj.input_cost == Decimal("0.000012")
    # cached = 30 tokens @ $0.01/1M = 0.0000003 -> rounds to 0.000000
    assert added_obj.cached_input_cost == Decimal("0.000000")
    assert added_obj.cache_write_cost is None
    # output = 40 tokens @ $0.50/1M = 0.000020
    assert added_obj.output_cost == Decimal("0.000020")
    # total = 0.000012 + 0.000000 + 0.000020 = 0.000032
    assert added_obj.total_cost == Decimal("0.000032")
    # Ensure removed columns are not set on the ORM object
    assert not hasattr(added_obj, "total_tokens")
    assert not hasattr(added_obj, "status")
    assert not hasattr(added_obj, "error_message")
    mock_db.commit.assert_called_once()


def test_persist_llm_usage_with_cache_write_persists_all_fields():
    """Verify _persist_llm_usage persists cache_write_tokens and all individual costs matching total."""
    mock_db = MagicMock()
    strat_id = uuid.uuid4()

    usage_info = {
        "model": "gpt-6-luna",
        "prompt_tokens": 1000,
        "cached_tokens": 300,
        "cache_write_tokens": 200,
        "completion_tokens": 150,
        "total_tokens": 1150,
    }

    with patch("backend.app.db.session.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        _persist_llm_usage(
            operation="strategy_generation",
            model="gpt-6-luna",
            provider="openai",
            usage_info=usage_info,
            duration_ms=620,
            strategy_request_id=strat_id,
        )

    added_obj = mock_db.add.call_args[0][0]
    assert added_obj.input_tokens == 1000
    assert added_obj.cached_input_tokens == 300
    assert added_obj.cache_write_tokens == 200
    assert added_obj.output_tokens == 150
    # Fresh = 1000 - 300 - 200 = 500 @ $0.10/1M = 0.000050
    assert added_obj.input_cost == Decimal("0.000050")
    # Cached = 300 @ $0.01/1M = 0.000003
    assert added_obj.cached_input_cost == Decimal("0.000003")
    # Cache write = 200 @ $0.125/1M = 0.000025
    assert added_obj.cache_write_cost == Decimal("0.000025")
    # Output = 150 @ $0.50/1M = 0.000075
    assert added_obj.output_cost == Decimal("0.000075")
    # Total = 0.000050 + 0.000003 + 0.000025 + 0.000075 = 0.000153
    assert added_obj.total_cost == Decimal("0.000153")
    assert added_obj.total_cost == (
        added_obj.input_cost + added_obj.cached_input_cost + added_obj.cache_write_cost + added_obj.output_cost
    )


def test_calculate_llm_costs_with_zero_tokens_treated_as_zero():
    """Verify explicitly reported 0 token counts produce Decimal('0.000000')."""
    usage = {
        "prompt_tokens": 7,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "completion_tokens": 22,
    }
    costs = calculate_llm_costs(usage)
    # Fresh = 7 - 0 - 0 = 7 @ $0.10/1M = 0.0000007 -> ROUND_HALF_UP = 0.000001
    assert costs["input_cost"] == Decimal("0.000001")
    assert costs["cached_input_cost"] == Decimal("0.000000")
    assert costs["cache_write_cost"] == Decimal("0.000000")
    # Output = 22 @ $0.50/1M = 0.000011
    assert costs["output_cost"] == Decimal("0.000011")
    assert costs["total_cost"] == Decimal("0.000012")
    assert costs["total_cost"] == (
        costs["input_cost"] + costs["cached_input_cost"] + costs["cache_write_cost"] + costs["output_cost"]
    )


def test_calculate_llm_costs_negative_fresh_tokens_guarded():
    """Verify that if cached + cache_write exceed prompt_tokens, fresh tokens guard at 0."""
    usage = {
        "prompt_tokens": 100,
        "cached_tokens": 80,
        "cache_write_tokens": 50,  # 80 + 50 = 130 > 100
        "completion_tokens": 10,
    }
    costs = calculate_llm_costs(usage)
    # Fresh = max(0, 100 - 130) = 0 tokens -> cost = 0.000000
    assert costs["input_cost"] == Decimal("0.000000")
    assert costs["cached_input_cost"] == Decimal("0.000001")  # 80 * 0.01 / 1M = 0.0000008 -> 0.000001
    assert costs["cache_write_cost"] == Decimal("0.000006")   # 50 * 0.125 / 1M = 0.00000625 -> 0.000006
    assert costs["output_cost"] == Decimal("0.000005")        # 10 * 0.50 / 1M = 0.000005
    assert costs["total_cost"] == Decimal("0.000012")


def test_calculate_llm_costs_missing_usage_metadata_returns_none():
    """Verify missing usage metadata or rates stores None (NULL) instead of assuming zero."""
    usage = {
        "prompt_tokens": None,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "completion_tokens": None,
    }
    costs = calculate_llm_costs(usage)
    assert costs["input_cost"] is None
    assert costs["cached_input_cost"] is None
    assert costs["cache_write_cost"] is None
    assert costs["output_cost"] is None
    assert costs["total_cost"] is None


def test_calculate_llm_costs_partial_missing_cache_write_keeps_it_none():
    """Verify that if cache_write_tokens is None, cache_write_cost is None but other costs work."""
    usage = {
        "prompt_tokens": 500,
        "cached_tokens": 100,
        "cache_write_tokens": None,
        "completion_tokens": 50,
    }
    costs = calculate_llm_costs(usage)
    assert costs["input_cost"] == Decimal("0.000040")       # (500 - 100) * 0.10 / 1M
    assert costs["cached_input_cost"] == Decimal("0.000001") # 100 * 0.01 / 1M
    assert costs["cache_write_cost"] is None
    assert costs["output_cost"] == Decimal("0.000025")       # 50 * 0.50 / 1M
    assert costs["total_cost"] == Decimal("0.000066")


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
    assert added_obj.cached_input_tokens is None


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

            result = get_llm_response(prompt="hello")

    assert result == "working response text"
    assert any("Failed to persist LLM usage record to database" in record.message for record in caplog.records)


def test_get_llm_response_never_passes_temperature_to_api():
    """Verify that temperature parameter is never passed to client.chat.completions.create."""
    mock_response = MagicMock()
    mock_response.model = "gpt-4o-mini"
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "result text"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 70

    with patch("backend.app.services.llm_service.OpenAI") as mock_openai_cls, \
         patch("backend.app.services.llm_service._persist_llm_usage"):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        # Call with explicit temperature
        result = get_llm_response(prompt="test", temperature=0.7)

        assert result == "result text"
        mock_client.chat.completions.create.assert_called_once()
        create_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in create_kwargs, f"temperature should not be in API call kwargs, got {create_kwargs}"
