from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mindly.embeddings import EmbeddingService
from mindly.memory.models import Fact, RecallPolicy, Turn
from mindly.memory.vector_index import VectorIndex

logger = logging.getLogger("mindly.memory")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(
        self,
        sqlite_path: Path,
        chroma_dir: Path,
        embedding_service: EmbeddingService,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.embedding_service = embedding_service
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._vector_index = VectorIndex(chroma_dir / "vectors.json")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                recall_policy TEXT NOT NULL,
                source_quote TEXT,
                created_at TEXT NOT NULL,
                session_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);

            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                persona TEXT,
                created_at TEXT NOT NULL,
                session_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);

            CREATE TABLE IF NOT EXISTS summaries (
                user_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def add_turn(self, turn: Turn) -> None:
        self._conn.execute(
            """
            INSERT INTO turns (id, user_id, role, content, persona, created_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.id,
                turn.user_id,
                turn.role,
                turn.content,
                turn.persona,
                turn.created_at,
                turn.session_id,
            ),
        )
        self._conn.commit()
        logger.info("ход_сохранён user_id=%s role=%s", turn.user_id, turn.role)

    def get_recent_turns(self, user_id: str, limit: int) -> list[Turn]:
        rows = self._conn.execute(
            """
            SELECT * FROM turns
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        turns = [
            Turn(
                id=row["id"],
                user_id=row["user_id"],
                role=row["role"],
                content=row["content"],
                persona=row["persona"] or "",
                created_at=row["created_at"],
                session_id=row["session_id"],
            )
            for row in rows
        ]
        return list(reversed(turns))

    def add_fact(self, fact: Fact) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO facts
            (id, user_id, text, subject, predicate, object, recall_policy, source_quote, created_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.id,
                fact.user_id,
                fact.text,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.recall_policy,
                fact.source_quote,
                fact.created_at,
                fact.session_id,
            ),
        )
        self._conn.commit()
        embedding = self.embedding_service.embed_passage(fact.text)
        self._vector_index.upsert(
            fact.id,
            fact.user_id,
            embedding,
            {
                "recall_policy": fact.recall_policy,
                "subject": fact.subject,
                "created_at": fact.created_at,
            },
        )
        logger.info(
            "запись_памяти user_id=%s fact_id=%s policy=%s text=%s",
            fact.user_id,
            fact.id,
            fact.recall_policy,
            fact.text[:120],
        )

    def retrieve_facts(
        self,
        user_id: str,
        query: str,
        top_k: int,
        include_passive: bool = True,
    ) -> list[Fact]:
        embedding = self.embedding_service.embed_query(query)
        if include_passive:
            fact_ids = self._vector_index.query(user_id, embedding, top_k=top_k)
        else:
            fact_ids = self._vector_index.query(
                user_id,
                embedding,
                top_k=top_k,
                recall_policy="active",
            )
        if not fact_ids:
            return self._retrieve_facts_sql(user_id, query, top_k, include_passive)

        placeholders = ",".join("?" for _ in fact_ids)
        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE user_id = ? AND id IN ({placeholders})",
            [user_id, *fact_ids],
        ).fetchall()
        facts = [self._row_to_fact(row) for row in rows]
        logger.info(
            "извлечение_памяти user_id=%s query=%s count=%d ids=%s",
            user_id,
            query[:80],
            len(facts),
            fact_ids,
        )
        return facts

    def _retrieve_facts_sql(
        self,
        user_id: str,
        query: str,
        top_k: int,
        include_passive: bool,
    ) -> list[Fact]:
        sql = "SELECT * FROM facts WHERE user_id = ?"
        params: list = [user_id]
        if not include_passive:
            sql += " AND recall_policy = 'active'"
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(top_k * 3)
        rows = self._conn.execute(sql, params).fetchall()
        query_lower = query.lower()
        scored = []
        for row in rows:
            text = row["text"].lower()
            score = sum(1 for token in query_lower.split() if token in text)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        facts = [self._row_to_fact(row) for _, row in scored[:top_k]]
        logger.info("извлечение_памяти_sql user_id=%s count=%d", user_id, len(facts))
        return facts

    def get_active_facts_for_proactive(self, user_id: str, limit: int = 3) -> list[Fact]:
        rows = self._conn.execute(
            """
            SELECT * FROM facts
            WHERE user_id = ? AND recall_policy = 'active'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def forget_matching(self, user_id: str, query: str) -> int:
        facts = self.retrieve_facts(user_id, query, top_k=10, include_passive=True)
        if not facts:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE user_id = ? AND text LIKE ?",
                (user_id, f"%{query}%"),
            ).fetchall()
            facts = [self._row_to_fact(row) for row in rows]
        deleted = 0
        for fact in facts:
            self._delete_fact(fact.id, user_id)
            deleted += 1
        logger.info("удаление_памяти user_id=%s query=%s deleted=%d", user_id, query, deleted)
        return deleted

    def forget_all(self, user_id: str) -> int:
        rows = self._conn.execute("SELECT id FROM facts WHERE user_id = ?", (user_id,)).fetchall()
        for row in rows:
            self._delete_fact(row["id"], user_id)
        self._vector_index.delete_user(user_id)
        self._conn.execute("DELETE FROM turns WHERE user_id = ?", (user_id,))
        self._conn.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
        self._conn.commit()
        logger.info("полное_удаление_памяти user_id=%s deleted=%d", user_id, len(rows))
        return len(rows)

    def _delete_fact(self, fact_id: str, user_id: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE id = ? AND user_id = ?", (fact_id, user_id))
        self._conn.commit()
        self._vector_index.delete(fact_id)

    def get_summary(self, user_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content FROM summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["content"] if row else None

    def upsert_summary(self, user_id: str, content: str) -> None:
        self._conn.execute(
            """
            INSERT INTO summaries (user_id, content, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at
            """,
            (user_id, content, _utc_now()),
        )
        self._conn.commit()

    def count_facts(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM facts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["cnt"])

    def list_facts(self, user_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            user_id=row["user_id"],
            text=row["text"],
            subject=row["subject"] or "",
            predicate=row["predicate"] or "",
            object=row["object"] or "",
            recall_policy=row["recall_policy"],
            source_quote=row["source_quote"] or "",
            created_at=row["created_at"],
            session_id=row["session_id"],
        )

    def make_fact(
        self,
        user_id: str,
        text: str,
        subject: str,
        predicate: str,
        object_value: str,
        recall_policy: RecallPolicy,
        source_quote: str,
        session_id: str | None = None,
    ) -> Fact:
        return Fact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            text=text,
            subject=subject,
            predicate=predicate,
            object=object_value,
            recall_policy=recall_policy,
            source_quote=source_quote,
            created_at=_utc_now(),
            session_id=session_id,
        )

    def make_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        persona: str,
        session_id: str | None = None,
    ) -> Turn:
        return Turn(
            id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            content=content,
            persona=persona,
            created_at=_utc_now(),
            session_id=session_id,
        )

    def close(self) -> None:
        self._conn.close()
