import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindly.agent import MindlyAgent


def main() -> None:
    agent = MindlyAgent()
    user_id = "ttft_probe"
    agent.forget(user_id, "all")
    agent.ingest_turn(user_id, "user", "У меня сын в 9 классе, скоро экзамены.")

    samples = []
    for _ in range(5):
        start = time.perf_counter()
        stream = agent.chat(user_id, "wellness_friend", "Тяжёлый день на работе.", stream=True)
        first_token_time = None
        if isinstance(stream, str):
            first_token_time = time.perf_counter() - start
        else:
            for token in stream:
                if token:
                    first_token_time = time.perf_counter() - start
                    break
        if first_token_time is not None:
            samples.append(first_token_time)

    if not samples:
        print("Не удалось измерить TTFT. Проверьте API-ключ и доступность модели.")
        return

    print(f"Образцы TTFT (с): {samples}")
    print(f"TTFT p50: {statistics.median(samples):.3f} с")
    print(f"TTFT p95: {sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]:.3f} с")


if __name__ == "__main__":
    main()
