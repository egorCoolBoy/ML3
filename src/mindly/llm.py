from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

logger = logging.getLogger("mindly.llm")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        fallback_model: str,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        if not api_key or not api_key.strip():
            raise ValueError(
                "Не задан OPENROUTER_API_KEY. Добавьте ключ в файл .env в корне проекта."
            )
        self._client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/mindly",
                "X-Title": "Mindly AI Coach",
            },
        )

    def _create(self, **kwargs: Any) -> Any:
        models = [self.model, self.fallback_model]
        last_error: Exception | None = None
        for model_name in models:
            try:
                response = self._client.chat.completions.create(model=model_name, **kwargs)
                logger.info("вызов_модели успех model=%s", model_name)
                return response
            except Exception as exc:
                last_error = exc
                logger.error("вызов_модели ошибка model=%s error=%s", model_name, exc)
        raise RuntimeError(f"Все модели недоступны: {last_error}")

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str:
        response = self._create(messages=messages, temperature=temperature, stream=False)
        content = response.choices[0].message.content or ""
        logger.debug("ответ_модели символов=%d", len(content))
        return content

    def complete_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        response = self.complete(messages, temperature=temperature)
        if not response.strip():
            logger.error("empty_llm_response")
            return {"facts": []}

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("bad_json_from_llm=%s", cleaned[:1000])
            return {"facts": []}

    def stream(self, messages: list[dict[str, str]], temperature: float = 0.7) -> Iterator[str]:
        stream = self._create(messages=messages, temperature=temperature, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

