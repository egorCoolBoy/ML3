from __future__ import annotations

from pathlib import Path

import yaml

from mindly.config import PROJECT_ROOT

DEFAULT_PERSONAS_PATH = PROJECT_ROOT / "config" / "personas.yaml"


def load_personas(path: Path | None = None) -> dict[str, dict[str, str]]:
    personas_path = path or DEFAULT_PERSONAS_PATH
    with personas_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_persona_prompt(persona_id: str, personas: dict[str, dict[str, str]] | None = None) -> str:
    catalog = personas or load_personas()
    if persona_id not in catalog:
        available = ", ".join(catalog.keys())
        raise ValueError(f"Неизвестная персона '{persona_id}'. Доступно: {available}")
    return catalog[persona_id]["system_prompt"].strip()
