from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: Path):
        self.path = path
        self._items: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def get(self, key_parts: dict[str, Any]) -> dict[str, Any] | None:
        self._ensure_loaded()
        key = self.key(key_parts)
        item = self._items.get(key)
        if not item:
            return None
        response = item["response"]
        response["cache"] = {
            "hit": True,
            "created_at": item["created_at"],
            "tokens_saved": item.get("tokens_saved", estimate_response_tokens(response)),
        }
        return response

    def set(self, key_parts: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        key = self.key(key_parts)
        tokens_saved = estimate_response_tokens(response)
        stored = json.loads(json.dumps(response))
        stored["cache"] = {"hit": False, "tokens_saved": 0}
        self._items[key] = {
            "created_at": int(time.time()),
            "tokens_saved": tokens_saved,
            "response": stored,
        }
        self._save()
        response["cache"] = {"hit": False, "tokens_saved": 0}
        return response

    def key(self, parts: dict[str, Any]) -> str:
        encoded = json.dumps(parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            self._items = json.loads(self.path.read_text(encoding="utf-8"))
        self._loaded = True

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, ensure_ascii=True), encoding="utf-8")


def estimate_response_tokens(response: dict[str, Any]) -> int:
    if "results" in response:
        return sum(_tokens_from_result(item) for item in response["results"])
    return _tokens_from_result(response)


def _tokens_from_result(result: dict[str, Any]) -> int:
    tokens = result.get("tokens")
    if isinstance(tokens, dict):
        return int(tokens.get("input") or 0) + int(tokens.get("output") or 0)
    answer = result.get("answer", "")
    return max(1, len(str(answer).split()) * 2)
