import pytest

from mindly.agent import MindlyAgent


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "log_file.log"))
    return MindlyAgent()


def _add_fact(agent: MindlyAgent, user_id: str, text: str) -> None:
    fact = agent.memory.make_fact(
        user_id=user_id,
        text=text,
        subject="user",
        predicate="mentioned",
        object_value=text,
        recall_policy="active",
        source_quote=text,
    )
    agent.memory.add_fact(fact)


def test_tenant_isolation_retrieval(agent: MindlyAgent):
    user_a = "tenant_a"
    user_b = "tenant_b"
    agent.forget(user_a, "all")
    agent.forget(user_b, "all")

    _add_fact(agent, user_a, "My secret codeword is BLUE ELEPHANT for user A only.")
    _add_fact(agent, user_b, "My favorite color is green.")

    facts_a = agent.memory.list_facts(user_a)
    facts_b = agent.memory.list_facts(user_b)

    assert facts_a
    assert facts_b
    assert all("BLUE ELEPHANT" not in fact.text for fact in facts_b)
    assert any("BLUE ELEPHANT" in fact.text for fact in facts_a)

    retrieved_for_b = agent.memory.retrieve_facts(user_b, "BLUE ELEPHANT secret codeword", top_k=5)
    assert not any("BLUE ELEPHANT" in fact.text for fact in retrieved_for_b)


def test_forget_targeted_and_all(agent: MindlyAgent):
    user_id = "forget_user"
    agent.forget(user_id, "all")
    _add_fact(agent, user_id, "I have a dog named Rex.")
    _add_fact(agent, user_id, "I want to run a marathon in autumn.")

    assert agent.memory.count_facts(user_id) >= 1
    deleted = agent.forget(user_id, "dog Rex")
    assert deleted >= 0

    remaining = agent.memory.list_facts(user_id)
    assert all("rex" not in fact.text.lower() for fact in remaining)

    _add_fact(agent, user_id, "Another fact about work stress.")
    wiped = agent.forget(user_id, "all")
    assert wiped >= 0
    assert agent.memory.count_facts(user_id) == 0


def test_shared_memory_across_personas(agent: MindlyAgent):
    user_id = "persona_user"
    agent.forget(user_id, "all")
    _add_fact(agent, user_id, "My son is in 9th grade.")

    messages = agent._build_messages(user_id, "tough_love", "Hi, rough day.")
    context = messages[0]["content"]
    assert "9th" in context.lower() or "son" in context.lower() or "grade" in context.lower()
