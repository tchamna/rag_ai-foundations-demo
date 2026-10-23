from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    path: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    path: str
    page: int | None
    section: str | None
    token_estimate: int


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
