"""
Thin wrapper around the Groq chat-completions API.

Keeping this in one place means every agent calls the LLM the same way,
with the same retry/backoff and error handling, and makes it trivial to
swap models via utils.config.MODEL_NAME.
"""
from __future__ import annotations

from groq import Groq

from utils.config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_NAME,
    get_groq_api_key,
)
from utils.helpers import with_retry

_client: Groq | None = None


class GroqNotConfiguredError(Exception):
    """Raised when no API key is available."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = get_groq_api_key()
        if not api_key:
            raise GroqNotConfiguredError(
                "GROQ_API_KEY is not set. Add it under Streamlit Secrets "
                "or as an environment variable."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat_completion(system_prompt: str, user_prompt: str,
                     temperature: float = LLM_TEMPERATURE,
                     max_tokens: int = LLM_MAX_TOKENS,
                     model: str = MODEL_NAME) -> str:
    """Call the Groq chat API once (with retry) and return the raw text."""
    client = _get_client()

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return with_retry(_call)
