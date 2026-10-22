from decimal import Decimal, ROUND_HALF_UP
import logging
import re
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


def _sanitize_error_message(error: Union[Exception, str, None]) -> str:
    """
    Safely sanitizes error messages by scrubbing potential API keys, authorization headers,
    bearer tokens, and sensitive details while retaining the error category and safe description.
    Truncates to 500 characters max.
    """
    if error is None:
        return "Unknown error"

    msg = str(error)
    # Mask potential OpenAI or other API keys (e.g. sk-...)
    msg = re.sub(r"sk-[a-zA-Z0-9_\-]{8,}", "sk-***REDACTED***", msg)
    # Mask Bearer tokens
    msg = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{8,}", "Bearer ***REDACTED***", msg)
    # Mask basic authorization headers / tokens
    msg = re.sub(r"(?i)(authorization|api[-_]?key|token)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]+['\"]?", r"\1: ***REDACTED***", msg)

    err_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    if err_type not in msg:
        msg = f"{err_type}: {msg}"

    return msg.strip()[:500]


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
    status: str = "success",
    attempt_number: int = 1,
    error_message: Optional[str] = None,
    record_id: Optional[uuid.UUID] = None,
    max_db_retries: int = 2,
) -> Optional[uuid.UUID]:
    """
    Persists one row to the llm_usage table using SessionLocal.
    Catches and logs any database error with structured context so LLM invocation
    never fails due to tracking errors. Includes bounded retry with idempotency for transient database errors.
    """
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

    target_record_id = record_id or uuid.uuid4()

    cached_input = usage_info.get("cached_tokens")
    if cached_input is None:
        cached_input = usage_info.get("cached_input_tokens")

    cache_write = usage_info.get("cache_write_tokens")
    costs = calculate_llm_costs(usage_info)

    from backend.app.db.session import SessionLocal
    from backend.app.models.llm_usage import LLMUsage
    from backend.app.models.conversation import Conversation

    last_error = None
    sanitized_err_msg = _sanitize_error_message(error_message) if error_message else None

    for db_attempt in range(1, max_db_retries + 2):
        try:
            with SessionLocal() as db:
                try:
                    # If conversation_id is known but strategy_request_id is not,
                    # resolve strategy_request_id from the conversation record if it exists
                    if conv_uuid and not strat_uuid:
                        try:
                            conv_rec = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
                            if conv_rec and getattr(conv_rec, "strategy_request_id", None):
                                resolved_strat = conv_rec.strategy_request_id
                                if isinstance(resolved_strat, uuid.UUID):
                                    strat_uuid = resolved_strat
                                elif isinstance(resolved_strat, str):
                                    strat_uuid = uuid.UUID(resolved_strat)
                        except Exception as lookup_err:
                            logger.debug("Could not lookup strategy_request_id from conversation: %s", lookup_err)

                    # If strategy_request_id is known but conversation_id is not,
                    # resolve conversation_id from the conversation record linked 1-to-1 if it exists
                    if strat_uuid and not conv_uuid:
                        try:
                            conv_rec = db.query(Conversation).filter(Conversation.strategy_request_id == strat_uuid).first()
                            if conv_rec and getattr(conv_rec, "id", None):
                                resolved_conv = conv_rec.id
                                if isinstance(resolved_conv, uuid.UUID):
                                    conv_uuid = resolved_conv
                                elif isinstance(resolved_conv, str):
                                    conv_uuid = uuid.UUID(resolved_conv)
                        except Exception as lookup_err:
                            logger.debug("Could not lookup conversation_id from strategy_request: %s", lookup_err)

                    # Idempotency check: on retry (db_attempt > 1) or when record_id is explicitly provided,
                    # verify if record with target_record_id was already persisted by a previous attempt
                    if (db_attempt > 1 or record_id is not None):
                        try:
                            existing = db.query(LLMUsage).filter(LLMUsage.id == target_record_id).first()
                            if existing:
                                logger.info("LLM usage record %s already persisted (idempotent write)", target_record_id)
                                return existing.id
                        except Exception:
                            pass

                    usage_record = LLMUsage(
                        id=target_record_id,
                        strategy_request_id=strat_uuid,
                        conversation_id=conv_uuid,
                        operation=operation,
                        provider=provider,
                        model=model,
                        status=status,
                        attempt_number=attempt_number,
                        error_message=sanitized_err_msg,
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
                    return usage_record.id
                except Exception:
                    db.rollback()
                    raise
        except Exception as e:
            last_error = e
            logger.warning(
                "Transient DB write failure on attempt %d/%d for LLM usage [record_id=%s, op=%s, model=%s]: %s",
                db_attempt,
                max_db_retries + 1,
                target_record_id,
                operation,
                model,
                e,
            )
            if db_attempt < max_db_retries + 1:
                time.sleep(0.05 * db_attempt)

    # Observable structured error log on unrecoverable persistence failure
    sanitized_db_err = _sanitize_error_message(last_error)
    logger.error(
        "Failed to persist LLM usage record to database [record_id=%s, op=%s, model=%s, provider=%s, "
        "status=%s, attempt_number=%s, strategy_request_id=%s, conversation_id=%s, error_category=%s]: %s",
        target_record_id,
        operation,
        model,
        provider,
        status,
        attempt_number,
        strat_uuid,
        conv_uuid,
        type(last_error).__name__ if last_error else "UnknownError",
        sanitized_db_err,
    )
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
    max_retries: Optional[int] = None,
) -> str:
    """
    Sends a prompt or message list to OpenAI LLM, logs and persists usage to the database, and returns the response text.
    Preserves existing return type (str) and behavior for all callers.
    Tracks each attempt (successful and failed retries) in the llm_usage table.
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

    retry_count = max_retries if max_retries is not None else getattr(settings, "llm_max_retries", 0)
    total_attempts = max(1, 1 + retry_count)

    for attempt in range(1, total_attempts + 1):
        start_time = time.perf_counter()
        try:
            logger.info(f"Calling LLM ({target_model}) [operation={operation}, attempt={attempt}/{total_attempts}]")
            response = client.chat.completions.create(**kwargs)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            choice = response.choices[0]
            content = choice.message.content

            logger.info(
                "LLM response finish_reason=%s | content_length=%s | duration_ms=%s | attempt=%s",
                choice.finish_reason,
                len(content or ""),
                duration_ms,
                attempt,
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
                    status="success",
                    attempt_number=attempt,
                    error_message=None,
                )
            except Exception as persist_err:
                logger.error("Failed to persist LLM usage record to database: %s", persist_err)

            return _clean_json_string(content)

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            sanitized_err = _sanitize_error_message(e)

            # Extract usage metadata if present on the exception response; do not invent tokens
            failed_usage = {
                "prompt_tokens": None,
                "cached_tokens": None,
                "cache_write_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            if hasattr(e, "response"):
                try:
                    failed_usage = _extract_usage_metadata(getattr(e, "response"), fallback_model=target_model)
                except Exception:
                    pass

            # Persist failed attempt record; does not raise if DB write fails
            try:
                _persist_llm_usage(
                    operation=operation,
                    model=target_model,
                    provider=provider,
                    usage_info=failed_usage,
                    duration_ms=duration_ms,
                    strategy_request_id=strategy_request_id,
                    conversation_id=conversation_id,
                    status="failed",
                    attempt_number=attempt,
                    error_message=sanitized_err,
                )
            except Exception as persist_err:
                logger.error("Failed to persist LLM failure record to database: %s", persist_err)

            if attempt < total_attempts:
                logger.warning(
                    "LLM attempt %d/%d failed: %s. Retrying...",
                    attempt,
                    total_attempts,
                    sanitized_err,
                )
                time.sleep(0.1)
                continue

            logger.error("OpenAI API request failed: %s", e)
            raise LLMServiceError(f"Failed to get response from LLM: {e}") from e

