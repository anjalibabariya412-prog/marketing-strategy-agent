import logging
from groq import Groq
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
    model: str = None
) -> str:
    """
    Sends a prompt to the Groq LLM and returns the response text.
    Supports optional model override, response_format (e.g. {"type": "json_object"}), and temperature.
    Includes fallback retry if Groq's strict server-side json_object validation fails.
    """
    target_model = model or settings.groq_model
    client = Groq(api_key=settings.groq_api_key)

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

    try:
        logger.info(f"Calling LLM ({target_model})")
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return _clean_json_string(content)

    except Exception as e:
        err_msg = str(e)
        # Fallback: If Groq's strict json_object validation failed (HTTP 400 json_validate_failed),
        # retry without the strict response_format parameter.
        if response_format and ("json_validate_failed" in err_msg or "400" in err_msg):
            logger.warning(f"Groq json_object validation error ({e}). Retrying request without response_format constraint...")
            try:
                kwargs_no_fmt = {
                    "model": target_model,
                    "messages": messages,
                }
                if temperature is not None:
                    kwargs_no_fmt["temperature"] = temperature
                response = client.chat.completions.create(**kwargs_no_fmt)
                content = response.choices[0].message.content
                return _clean_json_string(content)
            except Exception as retry_err:
                raise LLMServiceError(f"Failed to get response from LLM on fallback retry: {retry_err}") from retry_err

        raise LLMServiceError(f"Failed to get response from LLM: {e}") from e