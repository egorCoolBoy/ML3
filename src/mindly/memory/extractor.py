from __future__ import annotations

import logging
import re
from typing import Any

from mindly.llm import LLMClient
from mindly.memory.models import Fact, RecallPolicy
from mindly.memory.store import MemoryStore

logger = logging.getLogger("mindly.extractor")

EXTRACTION_PROMPT = """Извлеки атомарные факты для долгосрочной памяти только из сообщения ПОЛЬЗОВАТЕЛЯ.
Верни JSON:
{
  "facts": [
    {
      "subject": "строка",
      "predicate": "строка",
      "object": "строка",
      "text": "одно краткое предложение-факт",
      "recall_policy": "active или passive_only",
      "source_quote": "точная цитата из сообщения пользователя"
    }
  ]
}
Правила:
- Только факты, явно сказанные пользователем.
- Если пользователь просит больше не поднимать тему — recall_policy = passive_only.
- Если пользователь просит забыть — верни пустой массив facts.
- Не выдумывай факты.
- Поддерживай русский и английский.
"""

SUMMARY_PROMPT = """Сделай краткую сводку прогресса диалога для велнес-коучингового приложения.
Сохрани цели, семью, предпочтения, паттерны. Максимум 120 слов. Язык — как в диалоге."""


class FactExtractor:
    def __init__(self, llm: LLMClient, memory: MemoryStore) -> None:
        self.llm = llm
        self.memory = memory

    def extract_from_message(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
    ) -> list[Fact]:
        if self._is_forget_command(message):
            return []
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": message},
        ]
        try:
            payload = self.llm.complete_json(messages)
        except Exception as exc:
            logger.error("ошибка_извлечения_фактов user_id=%s error=%s", user_id, exc)
            return self._heuristic_extract(user_id, message, session_id)

        facts: list[Fact] = []
        for item in payload.get("facts", []):
            fact = self._build_fact(user_id, item, message, session_id)
            if fact:
                facts.append(fact)
        return facts

    def update_summary(self, user_id: str, user_message: str, assistant_message: str) -> None:
        previous = self.memory.get_summary(user_id) or ""
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Предыдущая сводка:\n{previous}\n\n"
                    f"Новый ход:\nПользователь: {user_message}\nАссистент: {assistant_message}"
                ),
            },
        ]
        try:
            summary = self.llm.complete(messages, temperature=0.2)
            self.memory.upsert_summary(user_id, summary.strip())
            logger.info("сводка_обновлена user_id=%s", user_id)
        except Exception as exc:
            logger.error("ошибка_обновления_сводки user_id=%s error=%s", user_id, exc)

    def _build_fact(
        self,
        user_id: str,
        item: dict[str, Any],
        message: str,
        session_id: str | None,
    ) -> Fact | None:
        text = (item.get("text") or "").strip()
        source_quote = (item.get("source_quote") or "").strip()
        if not text:
            return None
        if source_quote and source_quote not in message:
            return None
        policy: RecallPolicy = (
            "passive_only" if item.get("recall_policy") == "passive_only" else "active"
        )
        return self.memory.make_fact(
            user_id=user_id,
            text=text,
            subject=item.get("subject", ""),
            predicate=item.get("predicate", ""),
            object_value=item.get("object", ""),
            recall_policy=policy,
            source_quote=source_quote or text,
            session_id=session_id,
        )

    def _heuristic_extract(
        self,
        user_id: str,
        message: str,
        session_id: str | None,
    ) -> list[Fact]:
        facts: list[Fact] = []
        passive_markers = [
            "не поднимай",
            "don't bring up",
            "do not mention",
            "больше не будем об этом",
            "не упоминай",
        ]
        policy: RecallPolicy = "passive_only" if any(m in message.lower() for m in passive_markers) else "active"
        sentences = re.split(r"[.!?\n]+", message)
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 12:
                continue
            facts.append(
                self.memory.make_fact(
                    user_id=user_id,
                    text=cleaned,
                    subject="пользователь",
                    predicate="упомянул",
                    object_value=cleaned,
                    recall_policy=policy,
                    source_quote=cleaned,
                    session_id=session_id,
                )
            )
        return facts[:5]

    @staticmethod
    def _is_forget_command(message: str) -> bool:
        lowered = message.lower().strip()
        patterns = [
            r"^забудь\b",
            r"^forget\b",
            r"^удали память",
            r"^delete all",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)
