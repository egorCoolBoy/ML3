from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Literal

from mindly.config import Settings, get_settings
from mindly.embeddings import EmbeddingService
from mindly.llm import LLMClient
from mindly.logging_setup import setup_logging
from mindly.memory.extractor import FactExtractor
from mindly.memory.store import MemoryStore
from mindly.personas import get_persona_prompt, load_personas

logger = logging.getLogger("mindly.agent")


class MindlyAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.settings.log_file)
        self.embedding_service = EmbeddingService(self.settings.embedding_model)
        self.memory = MemoryStore(
            sqlite_path=self.settings.sqlite_path,
            chroma_dir=self.settings.chroma_dir,
            embedding_service=self.embedding_service,
        )
        self.llm = LLMClient(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            model=self.settings.llm_model,
            fallback_model=self.settings.llm_fallback_model,
        )
        self.extractor = FactExtractor(self.llm, self.memory)
        self.personas = load_personas()

    def chat(
        self,
        user_id: str,
        persona: str,
        message: str,
        stream: bool = True,
        session_id: str | None = None,
    ) -> Iterator[str] | str:
        if self._is_forget_intent(message):
            query = self._parse_forget_query(message)
            deleted = self.forget(user_id, query)
            response = f"Удалено записей памяти: {deleted}."
            if stream:
                def _gen() -> Iterator[str]:
                    yield response
                return _gen()
            return response

        user_turn = self.memory.make_turn(user_id, "user", message, persona, session_id)
        self.memory.add_turn(user_turn)

        messages = self._build_messages(user_id, persona, message)
        logger.info(
            "начало_чата user_id=%s persona=%s stream=%s",
            user_id,
            persona,
            stream,
        )

        if stream:
            return self._stream_response(user_id, persona, message, messages, session_id)
        content = self.llm.complete(messages)
        self._post_process_turn(user_id, persona, message, content, session_id)
        return content

    def forget(self, user_id: str, query: str | Literal["all"]) -> int:
        if query == "all":
            return self.memory.forget_all(user_id)
        return self.memory.forget_matching(user_id, query)

    def _stream_response(
        self,
        user_id: str,
        persona: str,
        message: str,
        messages: list[dict[str, str]],
        session_id: str | None,
    ) -> Iterator[str]:
        chunks: list[str] = []

        def _gen() -> Iterator[str]:
            for token in self.llm.stream(messages):
                chunks.append(token)
                yield token
            full_response = "".join(chunks)
            assistant_turn = self.memory.make_turn(user_id, "assistant", full_response, persona, session_id)
            self.memory.add_turn(assistant_turn)
            self._post_process_turn(user_id, persona, message, full_response, session_id)

        return _gen()

    def _post_process_turn(
        self,
        user_id: str,
        persona: str,
        message: str,
        response: str,
        session_id: str | None,
    ) -> None:
        facts = self.extractor.extract_from_message(user_id, message, session_id)
        for fact in facts:
            self.memory.add_fact(fact)
        self.extractor.update_summary(user_id, message, response)

    def _build_messages(self, user_id: str, persona: str, message: str) -> list[dict[str, str]]:
        system_parts = [get_persona_prompt(persona, self.personas)]
        memory_context = self._build_memory_context(user_id, message)
        if memory_context:
            system_parts.append(memory_context)
        proactive_hint = self._build_proactive_hint(user_id, message)
        if proactive_hint:
            system_parts.append(proactive_hint)

        messages: list[dict[str, str]] = [{"role": "system", "content": "\n\n".join(system_parts)}]
        recent_turns = self.memory.get_recent_turns(user_id, self.settings.recent_turns_limit)
        for turn in recent_turns[:-1]:
            role = "user" if turn.role == "user" else "assistant"
            messages.append({"role": role, "content": turn.content})
        messages.append({"role": "user", "content": message})
        return messages

    def _build_memory_context(self, user_id: str, message: str) -> str:
        include_passive = self._user_references_past(message)
        facts = self.memory.retrieve_facts(
            user_id=user_id,
            query=message,
            top_k=self.settings.retrieval_top_k,
            include_passive=include_passive,
        )
        summary = self.memory.get_summary(user_id)
        lines = []
        if summary:
            lines.append(f"Сводка сессии:\n{summary}")
        if facts:
            formatted = []
            for fact in facts:
                tag = "только по запросу" if fact.recall_policy == "passive_only" else "активная"
                formatted.append(f"- [{tag}] {fact.text}")
            lines.append("Извлечённые факты из памяти:\n" + "\n".join(formatted))
        return "Контекст памяти:\n" + "\n\n".join(lines) if lines else ""

    def _build_proactive_hint(self, user_id: str, message: str) -> str:
        if "?" in message:
            return ""
        if len(message.split()) > self.settings.proactive_word_threshold:
            return ""
        facts = self.memory.get_active_facts_for_proactive(user_id, limit=1)
        if not facts:
            return ""
        fact = facts[0]
        if fact.text.lower() in message.lower():
            return ""
        return (
            "Подсказка для проактивного recall: если уместно, мягко спроси об этом факте одним коротким вопросом: "
            f"{fact.text}"
        )

    @staticmethod
    def _user_references_past(message: str) -> bool:
        markers = [
            "помнишь",
            "remember",
            "в прошлый раз",
            "last time",
            "раньше",
            "before",
            "мы говорили",
            "we talked",
        ]
        lowered = message.lower()
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _is_forget_intent(message: str) -> bool:
        lowered = message.lower().strip()
        return bool(
            re.match(r"^(забудь|forget|удали память|delete all memory|delete all)\b", lowered)
        )

    @staticmethod
    def _parse_forget_query(message: str) -> str | Literal["all"]:
        lowered = message.lower().strip()
        if lowered in {"забудь всё", "забудь все", "delete all", "delete all memory", "удали память"}:
            return "all"
        for prefix in ("забудь ", "forget "):
            if lowered.startswith(prefix):
                return message[len(prefix) :].strip() or "all"
        return message

    def ingest_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        persona: str = "wellness_friend",
        session_id: str | None = None,
        extract_facts: bool = True,
    ) -> None:
        turn = self.memory.make_turn(user_id, role, content, persona, session_id)
        self.memory.add_turn(turn)
        if extract_facts and role == "user":
            facts = self.extractor.extract_from_message(user_id, content, session_id)
            for fact in facts:
                self.memory.add_fact(fact)

    def answer_from_memory(self, user_id: str, question: str, persona: str = "wellness_friend") -> str:
        messages = self._build_messages(user_id, persona, question)
        return self.llm.complete(messages, temperature=0.0)
