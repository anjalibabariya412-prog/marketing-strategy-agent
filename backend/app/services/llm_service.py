from groq import Groq
from backend.app.core.config import settings


class LLMServiceError(Exception):
    """Raised when the LLM service fails to get a response."""
    pass


def get_llm_response(prompt: str, system_prompt: str = None, response_format: dict = None) -> str:
    """
    Sends a prompt to the Groq LLM and returns the response text.
    Supports optional response_format (e.g. {"type": "json_object"}).
    """
    try:
        client = Groq(api_key=settings.groq_api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": settings.groq_model,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)

        return response.choices[0].message.content

    except Exception as e:
        raise LLMServiceError(f"Failed to get response from LLM: {e}")