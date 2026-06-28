"""Unified LLM client supporting Ollama and OpenAI-compatible APIs.

This client lets the pipeline use either a local Ollama server or any
OpenAI-compatible endpoint (e.g. OpenCode's opencode-llm-proxy running on
http://127.0.0.1:4010/v1) for topic extraction and answer synthesis.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around Ollama and OpenAI-compatible chat completions."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or settings.LLM_MODEL
        self.fallback_model = fallback_model or settings.LLM_FALLBACK_MODEL
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.api_key = api_key or settings.OPENAI_API_KEY

        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI provider selected but the 'openai' package is not installed. "
                    "Run: pip install openai"
                ) from exc
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            logger.info(
                "LLMClient initialized with OpenAI-compatible endpoint: %s (model=%s)",
                self.base_url,
                self.model,
            )
        elif self.provider == "ollama":
            import ollama

            self.client = ollama.Client(host=base_url or settings.OLLAMA_BASE_URL)
            logger.info(
                "LLMClient initialized with Ollama (model=%s)",
                self.model,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call an OpenAI-compatible chat completions endpoint."""
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """Call an Ollama chat endpoint."""
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["options"] = {"temperature": temperature}

        response = self.client.chat(**kwargs)
        return response["message"]["content"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Optional model override.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum tokens (OpenAI-compatible only).

        Returns:
            Generated text content.
        """
        model = model or self.model
        if self.provider == "openai":
            return self._call_openai(model, messages, temperature, max_tokens)
        return self._call_ollama(model, messages, temperature)

    def chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a chat completion, falling back to a secondary model on failure."""
        primary = model or self.model
        fallback = fallback_model or self.fallback_model

        try:
            return self.chat(messages, model=primary, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("Primary LLM %s failed: %s", primary, exc)
            if not fallback:
                raise
            logger.info("Falling back to LLM: %s", fallback)
            try:
                return self.chat(messages, model=fallback, temperature=temperature, max_tokens=max_tokens)
            except Exception as fallback_exc:
                logger.error(
                    "Fallback LLM %s also failed: %s",
                    fallback,
                    fallback_exc,
                    exc_info=True,
                )
                raise fallback_exc from exc
