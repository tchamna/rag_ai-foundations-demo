from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def load_env_files() -> list[Path]:
    loaded: list[Path] = []
    explicit = os.getenv("RAG_ENV_PATH")
    candidates = [
        Path(explicit) if explicit else None,
        ROOT / ".env",
        WORKSPACE / "african-languages-mcp-servers" / ".env",
        WORKSPACE / "maintenance-ai-agent" / ".env",
        WORKSPACE / "Temp" / ".env",
    ]
    for path in candidates:
        if path and path.exists():
            _load_env(path)
            loaded.append(path)
    return loaded


def available_providers() -> dict[str, bool]:
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "claude": bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "ollama": True,
    }


def env_sources_for_status(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path) for path in paths]


def _load_env(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
