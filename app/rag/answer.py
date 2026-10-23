from __future__ import annotations

import re
from collections import Counter

from .models import RetrievalHit


WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def build_answer(question: str, hits: list[RetrievalHit]) -> dict:
    if not hits:
        return {
            "answer": "I could not find relevant support in the indexed data. Try asking with a specific policy name, process, role, or phrase from the documents.",
            "citations": [],
        }

    query_terms = _terms(question)
    candidate_sentences = []
    for hit in hits:
        for sentence in _sentences(hit.chunk.text):
            terms = _terms(sentence)
            overlap = len(query_terms & terms)
            if overlap == 0:
                continue
            density = overlap / max(8, len(terms))
            score = overlap * 1.7 + density + hit.score
            candidate_sentences.append((score, sentence, hit.rank))

    if not candidate_sentences:
        context = hits[0].chunk.text.strip()
        answer = _clip(context, 850) + f" [{hits[0].rank}]"
    else:
        selected = _select_sentences(candidate_sentences, limit=5)
        answer = " ".join(f"{sentence} [{rank}]" for sentence, rank in selected)

    citations = [
        {
            "rank": hit.rank,
            "source": hit.chunk.source,
            "path": hit.chunk.path,
            "page": hit.chunk.page,
            "section": hit.chunk.section,
            "score": round(hit.score, 4),
            "preview": _clip(hit.chunk.text, 420),
            "text": hit.chunk.text,
        }
        for hit in hits
    ]
    return {"answer": answer, "citations": citations}


def _terms(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
    }
    terms = {word.lower() for word in WORD_RE.findall(text) if len(word) > 2 and word.lower() not in stop}
    normalized = text.lower()
    if "hipaa" in normalized or ("privacy" in normalized and "right" in normalized):
        terms.update(
            {
                "access",
                "inspect",
                "copy",
                "amend",
                "amended",
                "confidential",
                "restriction",
                "restrictions",
                "disclosure",
                "complaint",
                "authorization",
                "phi",
            }
        )
    return terms


def _sentences(text: str) -> list[str]:
    sentences = []
    for sentence in SENTENCE_RE.split(text):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if 45 <= len(sentence) <= 520:
            sentences.append(sentence)
    return sentences


def _select_sentences(candidates: list[tuple[float, str, int]], limit: int) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    seen_terms: Counter[str] = Counter()
    for _, sentence, rank in sorted(candidates, reverse=True):
        signature = " ".join(sorted(_terms(sentence))[:12])
        if seen_terms[signature]:
            continue
        selected.append((sentence, rank))
        seen_terms[signature] += 1
        if len(selected) == limit:
            break
    selected.sort(key=lambda item: item[1])
    return selected


def _clip(text: str, length: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= length:
        return text
    return text[: length - 1].rsplit(" ", 1)[0] + "..."
