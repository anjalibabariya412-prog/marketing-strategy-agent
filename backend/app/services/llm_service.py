from decimal import Decimal, ROUND_HALF_UP
import logging
import time
import uuid
from typing import Optional, Union

from openai import OpenAI
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

SIX_DECIMAL_PLACES = Decimal("0.000001")
ONE_MILLION = Decimal("1000000")


class LLMServiceError(Exception):
    """Raised when the LLM service fails to get a response."""
    pass


def _clean_json_string(text: str) -> str:
    """
    Strips markdown code blocks (e.g. ```json ... ```) from LLM output.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_usage_metadata(response, fallback_model: str = None) -> dict:
    """
    Safely captures actual token usage counts and model from the OpenAI response.
    Never estimates tokens from character length if metadata is missing.
    Extracts cache_write_tokens from prompt_tokens_details if available.
    """
    actual_model = getattr(response, "model", None)
    if not actual_model and isinstance(response, dict):
        actual_model = response.get("model")
    actual_model = actual_model or fallback_model

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    cached_tokens = None
    cache_write_tokens = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            prompt_tokens_details = usage.get("prompt_tokens_details")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)

        if prompt_tokens_details is not None:
            if isinstance(prompt_tokens_details, dict):
                cached_tokens = prompt_tokens_details.get("cached_tokens")
                raw_cw = prompt_tokens_details.get("cache_write_tokens")
            else:
                cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
                raw_cw = getattr(prompt_tokens_details, "cache_write_tokens", None)

            # Avoid MagicMock auto-attributes when mocking in tests
            if raw_cw is not None and not isinstance(raw_cw, int):
                raw_cw = None
            cache_write_tokens = raw_cw

            if cached_tokens is not None and not isinstance(cached_tokens, int):
                cached_tokens = None

    return {
        "model": actual_model,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _calculate_component_cost(token_count: Optional[int], rate_per_1m: Optional[Decimal]) -> Optional[Decimal]:
    """
    Calculates cost for a token count rounded to 6 decimal places using ROUND_HALF_UP.
    Returns None if token count or pricing rate is unavailable.
    Treats an explicitly reported 0 token count as Decimal('0.000000').
    """
    if token_count is None or rate_per_1m is None:
        return None
    cost = (Decimal(token_count) * Decimal(rate_per_1m)) / ONE_MILLION
    return cost.quantize(SIX_DECIMAL_PLACES, rounding=ROUND_HALF_UP)


def calculate_llm_costs(usage_info: dict) -> dict:
    """
    Centralized helper for LLM token cost calculations using Decimal and ROUND_HALF_UP.

    Accounting Semantics:
    - prompt_tokens is the total prompt token count reported by the API.
    - prompt_tokens_details.cached_tokens is the prompt tokens read from cache.
    - prompt_tokens_details.cache_write_tokens is the prompt tokens written to cache.
    - Fresh uncached input tokens = max(0, prompt_tokens - (cached_tokens or 0) - (cache_write_tokens or 0)).
    - completion_tokens includes reasoning_tokens; reasoning_tokens are not added separately.

    Total cost is calculated as the sum of the individually rounded components.
    If required usage metadata or applicable pricing rates are unavailable, stores None (NULL).
    """
    prompt_tokens = usage_info.get("prompt_tokens")
    cached_tokens = usage_info.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = usage_info.get("cached_input_tokens")
    cache_write_tokens = usage_info.get("cache_write_tokens")
    completion_tokens = usage_info.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage_info.get("output_tokens")

    input_rate = settings.llm_input_price_per_1m
    cached_rate = settings.llm_cached_input_price_per_1m
    cache_write_rate = settings.llm_cache_write_price_per_1m
    output_rate = settings.llm_output_price_per_1m

    # Fresh uncached input tokens
    if prompt_tokens is None:
        fresh_tokens = None
    else:
        deductions = (cached_tokens or 0) + (cache_write_tokens or 0)
        fresh_tokens = max(0, prompt_tokens - deductions)

    input_cost = _calculate_component_cost(fresh_tokens, input_rate)
    cached_input_cost = _calculate_component_cost(cached_tokens, cached_rate)
    cache_write_cost = _calculate_component_cost(cache_write_tokens, cache_write_rate)
    output_cost = _calculate_component_cost(completion_tokens, output_rate)

    # Determine total cost:
    # If base input or output cost cannot be calculated, total_cost is None.
    # If cached or cache-write tokens exist but their rates are missing, total_cost is None.
    if input_cost is None or output_cost is None:
        total_cost = None
    elif cached_tokens is not None and cached_input_cost is None:
        total_cost = None
    elif cache_write_tokens is not None and cache_write_cost is None:
        total_cost = None
    else:
        active_components = [
            c for c in [input_cost, cached_input_cost, cache_write_cost, output_cost]
            if c is not None
        ]
        total_cost = sum(active_components, Decimal("0.000000"))

    return {
        "input_cost": input_cost,
        "cached_input_cost": cached_input_cost,
        "cache_write_cost": cache_write_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def _persist_llm_usage(
    operation: str,
    model: str,
    provider: str,
    usage_info: dict,
    duration_ms: Optional[int] = None,
    strategy_request_id: Optional[Union[uuid.UUID, str]] = None,
    conversation_id: Optional[Union[uuid.UUID, str]] = None,
) -> Optional[uuid.UUID]:
    """
    Persists one row to the llm_usage table using SessionLocal.
    Catches and logs any database error so LLM invocation never fails due to tracking errors.
    """
    try:
        strat_uuid = None
        if strategy_request_id:
            try:
                strat_uuid = uuid.UUID(str(strategy_request_id))
            except (ValueError, TypeError):
                logger.warning("Invalid strategy_request_id UUID format: %s", strategy_request_id)

        conv_uuid = None
        if conversation_id:
            try:
                conv_uuid = uuid.UUID(str(conversation_id))
            except (ValueError, TypeError):
                logger.warning("Invalid conversation_id UUID format: %s", conversation_id)

        from backend.app.db.session import SessionLocal
        from backend.app.db.orm.llm_usage import LLMUsage
        from backend.app.db.orm.conversation import Conversation

        with SessionLocal() as db:
            # If conversation_id is known but strategy_request_id is not,
            # resolve strategy_request_id from the conversation record if it exists
            if conv_uuid and not strat_uuid:
                try:
                    conv_rec = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
                    if conv_rec and conv_rec.strategy_request_id:
                        strat_uuid = conv_rec.strategy_request_id
                except Exception as lookup_err:
                    logger.debug("Could not lookup strategy_request_id from conversation: %s", lookup_err)

            cached_input = usage_info.get("cached_tokens")
            if cached_input is None:
                cached_input = usage_info.get("cached_input_tokens")

            cache_write = usage_info.get("cache_write_tokens")
            costs = calculate_llm_costs(usage_info)

            usage_record = LLMUsage(
                strategy_request_id=strat_uuid,
                conversation_id=conv_uuid,
                operation=operation,
                provider=provider,
                model=model,
                input_tokens=usage_info.get("prompt_tokens"),
                cached_input_tokens=cached_input,
                cache_write_tokens=cache_write,
                output_tokens=usage_info.get("completion_tokens"),
                duration_ms=duration_ms,
                input_cost=costs.get("input_cost"),
                cached_input_cost=costs.get("cached_input_cost"),
                cache_write_cost=costs.get("cache_write_cost"),
                output_cost=costs.get("output_cost"),
                total_cost=costs.get("total_cost"),
            )
            db.add(usage_record)
            db.commit()
            db.refresh(usage_record)
            return usage_record.id

    except Exception as e:
        logger.error("Failed to persist LLM usage record to database: %s", e)
        return None


def get_llm_response(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None,
    response_format: dict = None,
    temperature: float = None,
    model: str = None,
    max_tokens: int = None,
    operation: str = "general_llm_call",
    strategy_request_id: Union[uuid.UUID, str] = None,
    conversation_id: Union[uuid.UUID, str] = None,
    provider: str = "openai",
) -> str:
    """
    Sends a prompt or message list to OpenAI LLM, logs and persists usage to the database, and returns the response text.
    Preserves existing return type (str) and behavior for all callers.
    """
    target_model = model or settings.openai_model
    client = OpenAI(api_key=settings.openai_api_key)

    if messages is not None:
        call_messages = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in call_messages):
            call_messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        call_messages = []
        if system_prompt:
            call_messages.append({"role": "system", "content": system_prompt})
        if prompt is not None:
            call_messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": target_model,
        "messages": call_messages,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        logger.info(f"Calling LLM ({target_model}) [operation={operation}]")
        start_time = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        choice = response.choices[0]
        content = choice.message.content

        logger.info(
            "LLM response finish_reason=%s | content_length=%s | duration_ms=%s",
            choice.finish_reason,
            len(content or ""),
            duration_ms,
        )

        usage_info = _extract_usage_metadata(response, fallback_model=target_model)
        if (
            usage_info["prompt_tokens"] is not None
            and usage_info["completion_tokens"] is not None
            and usage_info["total_tokens"] is not None
        ):
            cw_part = ""
            if usage_info.get("cache_write_tokens") is not None:
                cw_part = f", cache_write={usage_info['cache_write_tokens']}"
            logger.info(
                "LLM token usage: model=%s | prompt_tokens=%s (cached=%s%s) | completion_tokens=%s | total_tokens=%s | duration_ms=%s",
                usage_info["model"],
                usage_info["prompt_tokens"],
                usage_info.get("cached_tokens"),
                cw_part,
                usage_info["completion_tokens"],
                usage_info["total_tokens"],
                duration_ms,
            )
        else:
            logger.info(
                "LLM token usage: model=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s (usage metadata unavailable)",
                usage_info["model"],
                usage_info["prompt_tokens"],
                usage_info["completion_tokens"],
                usage_info["total_tokens"],
            )

        # Persist usage to database without failing the LLM response if DB write fails
        try:
            _persist_llm_usage(
                operation=operation,
                model=usage_info["model"],
                provider=provider,
                usage_info=usage_info,
                duration_ms=duration_ms,
                strategy_request_id=strategy_request_id,
                conversation_id=conversation_id,
            )
        except Exception as persist_err:
            logger.error("Failed to persist LLM usage record to database: %s", persist_err)

        return _clean_json_string(content)

    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        raise LLMServiceError(f"Failed to get response from LLM: {e}") from e
