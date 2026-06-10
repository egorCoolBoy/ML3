from __future__ import annotations

import json
import math
from pathlib import Path


class VectorIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors: dict[str, dict] = {}
        if self.path.exists():
            self._vectors = json.loads(self.path.read_text(encoding="utf-8"))

    def upsert(self, fact_id: str, user_id: str, embedding: list[float], metadata: dict) -> None:
        self._vectors[fact_id] = {
            "user_id": user_id,
            "embedding": embedding,
            "metadata": metadata,
        }
        self._persist()

    def delete(self, fact_id: str) -> None:
        if fact_id in self._vectors:
            del self._vectors[fact_id]
            self._persist()

    def delete_user(self, user_id: str) -> None:
        to_delete = [fact_id for fact_id, item in self._vectors.items() if item["user_id"] == user_id]
        for fact_id in to_delete:
            del self._vectors[fact_id]
        if to_delete:
            self._persist()

    def query(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int,
        recall_policy: str | None = None,
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for fact_id, item in self._vectors.items():
            if item["user_id"] != user_id:
                continue
            policy = item["metadata"].get("recall_policy", "active")
            if recall_policy and policy != recall_policy:
                continue
            score = _cosine_similarity(embedding, item["embedding"])
            scored.append((score, fact_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact_id for _, fact_id in scored[:top_k]]

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._vectors), encoding="utf-8")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
