import logging
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


def get_llm_response(
    prompt: str,
    system_prompt: str = None,
    response_format: dict = None,
    temperature: float = None,
    model: str = None,
    max_tokens: int = None
) -> str:
    """
    Sends a prompt to OpenAI LLM and returns the response text.
    Supports optional model override, response_format (e.g. {"type": "json_object"}), temperature, and max_tokens.
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
        logger.info(f"Calling LLM ({target_model})")
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content

        logger.info(
            "LLM response finish_reason=%s | content_length=%s",
            choice.finish_reason,
            len(content or "")
        )

        return _clean_json_string(content)

    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        raise LLMServiceError(f"Failed to get response from LLM: {e}") from e
