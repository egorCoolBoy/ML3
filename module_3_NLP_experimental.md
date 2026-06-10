# Homework 3 — Experimental Track

Conversation Agent with **Memory**: from a single-page transcript to an ML System Design doc and an MVP.

**DEADLINE: --.--.2026**

---

## Why this homework looks different

The original homework (Akkadian translation) was a textbook ML engineering task: dataset, leaderboard, fixed metric, well-known recipe. The reality of a ML engineer's first months is rarely that tidy. This homework simulates that reality.

**You are the new ML Engineer at a small startup called *Mindly*.** You started two weeks ago. The Senior ML Engineer who was supposed to onboard you handed in their resignation on your first Friday and is no longer reachable. The CTO is splitting their time between hiring a replacement and putting out fires.

Senior who left the company prepare only task_3_transcript.md file for you with the last call about current project.

---

## Recommended reading (start here, then go further on your own)

Memory in conversational agents is a **young, unsettled** subfield. There is no canonical recipe — that is the point of this homework. You must do the research yourself and defend the choices you make. These are starting points, not a syllabus:

#### Long-term memory and agent memory architectures
- MemGPT — *"MemGPT: Towards LLMs as Operating Systems"* (Packer et al., 2023): <https://arxiv.org/abs/2310.08560>
- Letta (the productionized successor of MemGPT): <https://docs.letta.com/>
- mem0 — open-source memory layer for AI agents: <https://github.com/mem0ai/mem0>
- LangMem (LangChain's long-term memory primitives): <https://langchain-ai.github.io/langmem/>
- Generative Agents — *"Generative Agents: Interactive Simulacra of Human Behavior"* (Park et al., 2023): <https://arxiv.org/abs/2304.03442>
- A-MEM — *"A-MEM: Agentic Memory for LLM Agents"* (Xu et al., 2024): <https://arxiv.org/abs/2502.12110>

#### Benchmarks for long-term memory
- LongMemEval — *"LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"* (Wu et al., 2024): <https://arxiv.org/abs/2410.10813>
- LoCoMo — *"Evaluating Very Long-Term Conversational Memory of LLM Agents"* (Maharana et al., 2024): <https://arxiv.org/abs/2402.17753>
- MemoryBench / DialSim / PerLTQA — see related work in the LongMemEval paper.

#### General background
- *"A Survey on the Memory Mechanism of Large Language Model based Agents"* (Zhang et al., 2024): <https://arxiv.org/abs/2404.13501>
- OpenAI's ChatGPT memory feature (product write-up): <https://openai.com/index/memory-and-new-controls-for-chatgpt/>
- Anthropic's prompt caching docs (orthogonal but useful for cost): <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>

You are **not required** to use any specific framework. You are required to **understand at least three options**, pick one, and defend the choice in writing.

---

### Part 1 — ML System Design Document

Recommend you use this template — copy the file into your repo and fill it out section by section:

> **Template:** [ML_System_Design_Doc_Template.md](https://github.com/IrinaGoloshchapova/ml_system_design_doc_ru/blob/main/ML_System_Design_Doc_Template.md) (Ирина Голощапова, ods.ai).

**Language: Russian or English — your choice.** Pick one and stay consistent within the document. The rest of your repo (code, README, commit messages) can be in either language too.

**Format:** Markdown in `/docs/ml_system_design_doc.md` of the repo. PDF export is optional. 

### Part 2 — MVP Conversation Agent with Memory

A working system that does **the demo Client described**. Concretely:

1. A **CLI or simple web UI** (your choice) where a user can sign in (any auth — even just a name) and chat with the agent.
2. Persistent memory across sessions: close the process, reopen it, the agent still remembers what was said before.
3. **At least two personas** (e.g., "tough-love" and "wellness friend"). The user can pick / switch. The memory of the client is **shared** across personas.
4. **Tenant isolation**: at least two test users in the system; demonstrate (with a script or README screenshots) that user B cannot see user A's facts.
5. **User-controlled forgetting**: a way for the user to say "forget X" or "delete all my memory" and have it actually take effect.
6. **Streaming output**. Tokens appear progressively, not in one final blob.
7. **Evaluation.** Run your system on **one** memory benchmark (LongMemEval or LoCoMo recommended; alternatives must be justified in the README). Report the number. **There is no quality threshold to hit** — there is a quality threshold to *defend*. You must argue, in writing, why the number you got is acceptable for the demo, and what would have to change to make it better.

#### What "any approach" means

You are explicitly free to use:
- A managed memory framework (mem0, Letta, LangMem, etc.).
- A hand-rolled memory layer over a vector DB + a small relational/KV store.
- Long-context only with periodic summarization.
- Anything else you can defend.

You are explicitly **not allowed** to:
- Use a single ever-growing context window with no retrieval/compression strategy. ("Just stuff everything in the prompt" fails the homework — it's neither a defensible architecture nor will it survive the cost model.)
- Hardcode the demo facts.

---

## Recommended Models & APIs

You can pick **any** LLM you can defend in the design doc's cost model. To keep this homework affordable, here are the paths we recommend.

### Option A — OpenRouter (free tier)

[OpenRouter](https://openrouter.ai/) gives you a single OpenAI-compatible API across hundreds of models. Filter by price to find free endpoints: <https://openrouter.ai/models?order=pricing-low-to-high>

| Provider     | Model                        | Access                                                                                                                                                                                                            |
|--------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OpenRouter   | Any free API                 | <https://openrouter.ai/models?order=pricing-low-to-high>                                                                                                                                                          |
| Local / Free | `Qwen2.5-7B`, `Llama-3.1-8B` | Via [Ollama](https://ollama.com), the [HuggingFace Inference API](https://huggingface.co/inference-api), or the [Unsloth inference & deployment guides](https://unsloth.ai/docs/basics/inference-and-deployment)  |

A few free OpenRouter endpoints worth trying:

- <https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b:free>
- <https://openrouter.ai/qwen/qwen3-next-80b-a3b-instruct:free>
- <https://openrouter.ai/openai/gpt-oss-20b:free>
- <https://openrouter.ai/z-ai/glm-4.5-air:free>

Free tiers have rate limits and can change without notice — pin the exact model ID in your config and document the fallback you'd use if the endpoint disappeared.

### Option B — Self-host a small LLM on Colab / Kaggle

If you'd rather control the serving stack (or your benchmark needs reproducibility), you can run a small open-weights model on a free GPU notebook (Google Colab or Kaggle) and expose it as a remote OpenAI-compatible endpoint via Ollama + a tunnel:

- Walkthrough: [Create a Remote LLM Server using Kaggle Notebooks and Ollama](https://medium.com/data-science-collective/create-a-remote-llm-server-using-kaggle-notebooks-and-ollama-acb299ead1e5)

This is a fine path for the MVP, but note in your design doc that free-notebook GPUs are **not** a production deployment target — sessions time out, IPs rotate, and TOS forbids long-running services. Treat it as a dev/eval harness, not as Mindly's year-end serving plan.

---

## Technical requirements

The same engineering hygiene as Homework 3 (regular track). We do not accept submissions unless **all** of the following are satisfied:

- **Git policy**: code in a public `GitHub` / `GitLab` repo. Two branches: `develop` (or `dev`) and `main` (or `master`). Meaningful commit messages, no last-minute commit rush. Your design doc lives in `/docs`.
- **Dependency management**: [Poetry](https://python-poetry.org/) or [UV](https://github.com/astral-sh/uv). Committed `poetry.lock` / `uv.lock`.
- **`README.md`**:
    - Your full name and group number.
    - Link to the design doc (or doc inlined under `/docs`).
    - "How-to": run the agent, run the eval, swap personas, delete memory, demonstrate tenant isolation.
    - The benchmark you chose, why, and the score you got.
    - Memory architecture summary: one paragraph + a diagram.
    - List of any datasets / models used with their licenses.
    - GIF or short video of two cross-session conversations demonstrating recall.
- **Logging**: full pipeline (memory writes, retrievals, deletions, model calls, errors) logged to `./data/log_file.log`.
- **Secrets**: API keys via `.env` only. `.env` in `.gitignore`.
- **Main software artifact**: a class (you choose the name, e.g. `MindlyAgent`) with at minimum:
    - `chat(user_id: str, persona: str, message: str, stream: bool = True) -> Iterator[str] | str`
    - `forget(user_id: str, query: str | Literal["all"]) -> None`
    - Memory persistence to disk / a local DB so the process can be restarted without losing state.
- **Experiment tracking**: benchmark runs logged to **ClearML** or **Weights & Biases**.
- **Deployment**: a `Dockerfile` (or `docker-compose.yaml` if you have a separate vector DB / UI) that brings up the whole thing in one command. CPU profile is fine.

---

## Project milestones

1. **Read & decide.** Read the transcript. Read at least three of the recommended papers/frameworks. Open a draft of the design doc. Write down every assumption you're making.
2. **Design.** Land on a memory architecture. Cost model. Eval plan. Send the doc to the CTO (your TA / instructor) for a midpoint check.
3. **Doc final.** Submit the design doc.
4. **MVP build.** Memory layer, agent loop, two personas, tenant isolation, deletion, streaming, persistence. Get it end-to-end before polishing any one piece.
5. **Eval & demo.** Run the benchmark. Record the demo video. Write the README. Defend the number you got.

---

## Grading

Total: **100 points.** No bonus tier — see below for optional bonuses.

### Part 1 — Design Document (**40 points**)

Grading is keyed to the [template's sections](https://github.com/IrinaGoloshchapova/ml_system_design_doc_ru/blob/main/ML_System_Design_Doc_Template.md). Each section is scored on **completeness against the task-specific guidance** above:

- **Full** = all guidance points addressed concretely (numbers where numbers are asked for, alternatives where alternatives are asked for, etc.).
- **Partial** = section present and on-topic, but at least one guidance point is missing or handwaved.
- **Empty** = section missing, copy-pasted from the template, or one-liner with no substance.

| Section                                          | Full | Partial | Empty | What "full" means for this task                                                                                                  |
|--------------------------------------------------|:----:|:-------:|:-----:|----------------------------------------------------------------------------------------------------------------------------------|
| 1.1 Зачем идем в разработку продукта?            | 2    | 1       | 0     | Mindly's unit economics + why an AI coach with memory is the lever, framed in the customer's terms.                              |
| 1.2 Бизнес-требования и ограничения              | 3    | 1       | 0     | Numbered list extracted from the transcript, each tagged stated / inferred / assumed, assumptions written out.                   |
| 1.3 Скоуп проекта / итерации                     | 2    | 1       | 0     | Demo scope vs phase 2 explicitly delineated; the "investor demo moment" identified.                                              |
| 1.4 Предпосылки решения                          | 2    | 1       | 0     | Available data, models, vendor.                                                   |
| 2.1 Постановка задачи                            | 2    | 1       | 0     | Inputs / outputs formalised; "memory" and "proactive recall" given operational definitions.                                      |
| 2.2 Блок-схема решения                           | 3    | 1       | 0     | Diagram with memory layer central; trade-off table; defended choice.                    |
| 2.3 Этапы решения задачи                         | 2    | 1       | 0     | Month plan with concrete artifacts at each step.                                                                          |
| 3.1 Способ оценки пилота                         | 3    | 1       | 0     | Benchmark named (LongMemEval / LoCoMo / justified alternative), metric, baseline to compare against, honest scope of the number. |
| 3.2 Что считаем успешным пилотом                 | 2    | 1       | 0     | Two layers: a product success criterion the customer would sign off on + an engineering number from 3.1.                         |
| 3.3 Подготовка пилота                            | 2    | 1       | 0     | Eval dataset, test users, pre-demo checklist, on-stage script for May 30.                                                        |
| 4.1 Архитектура решения                          | 3    | 1       | 0     | Concrete components (LLM, embeddings, vector store, KV/SQL, agent loop, persona layer, streaming gateway), each justified.       |
| 4.2 Инфраструктура и масштабируемость            | 2    | 1       | 0     | Sizing for demo *and* for 10k MAU year-end target; bottlenecks identified.                                                       |
| 4.3 Требования к работе системы                  | 2    | 1       | 0     | Quantified non-functional requirements (latency p50/p95, TTFT, throughput, availability) — numbers, not adjectives.              |
| 4.4 Безопасность системы                         | 2    | 1       | 0     | Auth, rate limiting, **prompt-injection threat model** for an agent with persistent memory, abuse handling.                      |
| 4.5 Безопасность данных                          | 3    | 1       | 0     | Per-tenant isolation, right-to-be-forgotten, encryption, retention,                                       |
| 4.6 Издержки                                     | 2    | 1       | 0     | Dollars/1k MAU at demo and year-end scale with shown math; hosted-API vs self-hosted comparison.                                 |
| 4.7 Integration points                           | 1    | 0       | 0     | Concrete API surface Mindly's existing app must expose / consume (auth, IDs, ingestion, deletion webhook).                       |
| 4.8 Риски                                        | 2    | 1       | 0     | Top 5 ranked, each with a mitigation; ≥ 1 covers tenant leakage.                                                                 |
| **Total**                                        | **40** | —     | —     |                                                                                                                                  |

### Part 2 — MVP (**60 points**)

| Points | Bulletpoint                                          | Description                                                                                                          |
|--------|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 16     | Cross-session recall works end-to-end                | Demo video: two sessions, recall on session 2 of facts from session 1, on a fresh user (not the developer's).        |
| 8      | Memory layer is non-trivial                          | Not a single-context-window hack. Retrieval/summarization/consolidation strategy implemented and visible in code.    |
| 6      | Tenant isolation                                     | At least two users, automated test or scripted demo proving no cross-tenant leakage.                                 |
| 6      | User-controlled forgetting                           | Targeted forget + full account deletion, both implemented, both demonstrated.                                        |
| 4      | At least two personas, shared memory                 | Two personas implemented; switch persona, recall preserved.                                                          |
| 4      | Streaming                                            | Tokens stream to the user; TTFT measured and reported.                                                               |
| 6      | Memory benchmark run + defended                      | One benchmark, one number, written defense of why the number is acceptable for the demo.                             |
| 4      | Logging, secrets hygiene, Docker, CLI                | All technical requirements above satisfied.                                                                          |
| 3      | Experiment tracking (ClearML / W&B)                  | Eval run + hyperparameters + memory config logged as an experiment.                                                  |
| 3      | Git workflow + README quality                        | Public repo, two branches, meaningful commits, README covers the how-to and the demo GIF.                            |

> **Automatic 0** for: hardcoding demo facts, leaking client A's memory into client B's session, committing API keys, fabricating benchmark numbers.

---

## Bonus part

Up to **20 bonus points** for any of:
- **Comparison study.** Implement *two* memory architectures (e.g. mem0 vs hand-rolled fact extraction) behind the same agent interface, run both on the same benchmark, and write up which won and why. The losing implementation must be a real attempt, not a strawman.
- **Self-hosted serving path.** Swap the hosted API for an open-weights model behind vLLM or SGLang. Report TTFT and tokens/sec, and the score delta on your benchmark.
- **Paper review** in the style of [DS Talks Siberia](https://t.me/+fQ07VSVJ2V8yZGYy) on one of: MemGPT, LongMemEval, A-MEM, the Generative Agents paper, or the memory survey by Zhang et al.
---

## A note on the spirit of this homework

There is no leaderboard. There is no public test set. There is no "right answer" we have hidden from you. The grade rewards **engineering judgment under uncertainty**: did you read the transcript carefully, did you research the field honestly, did you make defensible choices, did you write them down clearly, and did you ship something that works for a *user the senior never met* — not just for a script that hardcodes the demo.
