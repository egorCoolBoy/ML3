import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindly.agent import MindlyAgent


def main() -> None:
    agent = MindlyAgent()
    user_a = "demo_user_a"
    user_b = "demo_user_b"
    agent.forget(user_a, "all")
    agent.forget(user_b, "all")

    fact_a = agent.memory.make_fact(
        user_id=user_a,
        text="У пользователя A сын в 9 классе, готовится к экзаменам.",
        subject="сын",
        predicate="класс",
        object_value="9 класс",
        recall_policy="active",
        source_quote="сын в 9 классе",
    )
    fact_b = agent.memory.make_fact(
        user_id=user_b,
        text="Пользователь B готовится к марафону в Берлине.",
        subject="цель",
        predicate="готовится_к",
        object_value="марафон",
        recall_policy="active",
        source_quote="марафон в Берлине",
    )
    agent.memory.add_fact(fact_a)
    agent.memory.add_fact(fact_b)

    leak_query = "экзамены сын 9 класс"
    leaked = agent.memory.retrieve_facts(user_b, leak_query, top_k=5)
    print("Поиск по памяти пользователя B по теме пользователя A:")
    for fact in leaked:
        print(f"  - {fact.text}")
    if any("пользователя A" in f.text or "9 класс" in f.text for f in leaked):
        print("Обнаружена утечка между тенантами.")
    else:
        print("Утечка не обнаружена.")


if __name__ == "__main__":
    main()
