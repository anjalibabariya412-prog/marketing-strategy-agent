import logging
import time
import uuid
from typing import Optional, Union

from openai import OpenAI
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


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

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

    return {
        "model": actual_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
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

            usage_record = LLMUsage(
                strategy_request_id=strat_uuid,
                conversation_id=conv_uuid,
                operation=operation,
                provider=provider,
                model=model,
                input_tokens=usage_info.get("prompt_tokens"),
                output_tokens=usage_info.get("completion_tokens"),
                total_tokens=usage_info.get("total_tokens"),
                status="success",
                duration_ms=duration_ms,
            )
            db.add(usage_record)
            db.commit()
            db.refresh(usage_record)
            return usage_record.id

    except Exception as e:
        logger.error("Failed to persist LLM usage record to database: %s", e)
        return None


def get_llm_response(
    prompt: str,
    system_prompt: str = None,
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
    Sends a prompt to OpenAI LLM, logs and persists usage to the database, and returns the response text.
    Preserves existing return type (str) and behavior for all callers.
    """
    target_model = model or settings.openai_model
    client = OpenAI(api_key=settings.openai_api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": target_model,
        "messages": messages,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
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
            logger.info(
                "LLM token usage: model=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s | duration_ms=%s",
                usage_info["model"],
                usage_info["prompt_tokens"],
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
