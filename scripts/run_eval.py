import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindly.agent import MindlyAgent
from mindly.config import get_settings
from mindly.llm import LLMClient


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def llm_judge(llm: LLMClient, question: str, expected: str, actual: str) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluator for memory QA. "
                "Reply with only YES or NO. "
                "YES if the actual answer contains the expected information, even if phrased differently."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nExpected: {expected}\nActual: {actual}",
        },
    ]
    verdict = llm.complete(messages, temperature=0.0).strip().upper()
    return verdict.startswith("YES")


def exact_or_substring_match(expected: str, actual: str) -> bool:
    expected_n = normalize(expected)
    actual_n = normalize(actual)
    if not expected_n:
        return False
    return expected_n in actual_n or actual_n in expected_n


def load_dataset(path: Optional[Path], limit: int) -> list[dict]:
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[:limit]

    hf_path = Path.home() / ".cache" / "huggingface" / "hub"
    cached_files = list(hf_path.rglob("longmemeval_s_cleaned.json"))
    if cached_files:
        data = json.loads(cached_files[0].read_text(encoding="utf-8"))
        return data[:limit]

    from huggingface_hub import hf_hub_download

    dataset_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_s_cleaned.json",
        repo_type="dataset",
    )
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    return data[:limit]


def evaluate_item(agent: MindlyAgent, llm: LLMClient, item: dict, use_judge: bool) -> dict:
    user_id = f"eval_{item['question_id']}"
    agent.forget(user_id, "all")

    for session in item.get("haystack_sessions", []):
        user_messages = []

        for turn in session:
            role = turn["role"]
            content = turn["content"]
            agent.ingest_turn(
                user_id,
                role,
                content,
                extract_facts=False,
            )
            if role == "user":
                user_messages.append(content)

        facts = agent.extractor.extract_from_session(
            user_id=user_id,
            messages=user_messages,
        )
        for fact in facts:
            agent.memory.add_fact(fact)

    question = item["question"]
    expected = item["answer"]
    actual = agent.answer_from_memory(user_id, question)

    em = exact_or_substring_match(expected, actual)
    judged = llm_judge(llm, question, expected, actual) if use_judge else False
    correct = em or judged
    return {
        "question_id": item["question_id"],
        "question_type": item.get("question_type"),
        "question": question,
        "expected": expected,
        "actual": actual,
        "exact_or_substring": em,
        "llm_judge": judged,
        "correct": correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон бенчмарка LongMemEval")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--use-judge", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=Path("data/eval_results/results.json"))
    args = parser.parse_args()

    settings = get_settings()
    os.environ.setdefault("WANDB_MODE", settings.wandb_mode)
    agent = MindlyAgent(settings=settings)
    llm = agent.llm

    data = load_dataset(args.dataset, args.limit)
    results = []
    for idx, item in enumerate(data, start=1):
        print(f"Оценка {idx}/{len(data)}: {item['question_id']}")
        results.append(evaluate_item(agent, llm, item, args.use_judge))

    accuracy = sum(1 for r in results if r["correct"]) / max(len(results), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    memory_config = {
        "retrieval_top_k": settings.retrieval_top_k,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }
    payload = {
        "benchmark": "LongMemEval-S",
        "limit": args.limit,
        "accuracy": accuracy,
        "results": results,
        "memory_config": memory_config,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    wandb.init(
        project=settings.wandb_project,
        config=memory_config,
        name=f"longmemeval-{args.limit}",
    )
    wandb.log({"accuracy": accuracy, "num_items": len(results)})
    wandb.finish()

    print(f"Точность (accuracy): {accuracy:.3f}")
    print(f"Результаты сохранены: {args.output}")


if __name__ == "__main__":
    main()
