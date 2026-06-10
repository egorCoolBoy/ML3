from __future__ import annotations

import sys
from collections.abc import Iterator

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from mindly.agent import MindlyAgent
from mindly.personas import load_personas

console = Console()


def _print_stream(stream: Iterator[str]) -> str:
    parts: list[str] = []
    for token in stream:
        console.print(token, end="")
        parts.append(token)
    console.print()
    return "".join(parts)


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--user-id", default=None, help="Идентификатор пользователя (тенант)")
@click.option("--persona", default="wellness_friend", help="Идентификатор персоны")
def chat(user_id: str | None, persona: str) -> None:
    personas = load_personas()
    if persona not in personas:
        console.print(f"[red]Неизвестная персона. Доступно: {', '.join(personas.keys())}[/red]")
        sys.exit(1)

    resolved_user = user_id or Prompt.ask("Введите имя пользователя")
    agent = MindlyAgent()
    current_persona = persona

    console.print(
        Panel.fit(
            f"Mindly — AI-коуч\nПользователь: {resolved_user}\nПерсона: {current_persona}\n"
            "Команды: /persona <id>, /forget <текст>, /forget all, /memory, /exit",
            title="Mindly",
        )
    )

    while True:
        try:
            message = Prompt.ask(f"[bold cyan]{resolved_user}[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nДо встречи!")
            break

        if not message.strip():
            continue
        if message.strip() in {"/exit", "/quit", "/выход"}:
            break
        if message.startswith("/persona "):
            new_persona = message.split(" ", 1)[1].strip()
            if new_persona not in personas:
                console.print(f"[red]Неизвестная персона. Доступно: {', '.join(personas.keys())}[/red]")
                continue
            current_persona = new_persona
            console.print(f"[green]Персона переключена на {current_persona}[/green]")
            continue
        if message.startswith("/forget"):
            query = message.replace("/forget", "", 1).strip() or "all"
            deleted = agent.forget(resolved_user, query if query != "all" else "all")
            console.print(f"[yellow]Удалено записей: {deleted}[/yellow]")
            continue
        if message.strip() == "/memory":
            facts = agent.memory.list_facts(resolved_user)
            if not facts:
                console.print("[dim]Память пуста[/dim]")
            else:
                for fact in facts[:20]:
                    policy = "только по запросу" if fact.recall_policy == "passive_only" else "активная"
                    console.print(f"- [{policy}] {fact.text}")
            continue

        console.print(f"[bold magenta]{personas[current_persona]['name']}[/bold magenta]: ", end="")
        try:
            stream = agent.chat(resolved_user, current_persona, message, stream=True)
            if isinstance(stream, str):
                console.print(stream)
            else:
                _print_stream(stream)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        except RuntimeError as exc:
            console.print(f"[red]Ошибка LLM: {exc}[/red]")


@main.command()
def personas() -> None:
    catalog = load_personas()
    for persona_id, meta in catalog.items():
        console.print(f"[bold]{persona_id}[/bold] — {meta['name']}: {meta['description']}")


if __name__ == "__main__":
    main()
