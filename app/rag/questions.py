from __future__ import annotations

import re
from collections import Counter

from .models import Chunk


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
HEADING_RE = re.compile(r"\b(Purpose|Practices/Procedures|Employee Responsibilities|Manager/Supervisor Responsibilities|Forms|See Also|MasterControl)\b", re.I)


QUESTION_TEMPLATES = [
    "What should employees know about {topic}?",
    "What are the responsibilities for {topic}?",
    "What is the procedure for {topic}?",
    "Who is responsible for {topic}?",
    "What forms or records are required for {topic}?",
    "What fees, deadlines, or exceptions apply to {topic}?",
    "What should a manager do about {topic}?",
    "What should an employee do if there is an issue with {topic}?",
]


def generate_questions(chunks: list[Chunk], limit: int = 100) -> list[dict]:
    topics = _extract_topics(chunks)
    questions: list[dict] = []
    seen: set[str] = set()

    max_rounds = max(len(_templates_for_topic(topic)) for topic, *_ in topics) if topics else 0
    for round_index in range(max_rounds):
        for topic, score, source, page in topics:
            templates = _templates_for_topic(topic)
            if round_index >= len(templates):
                continue
            template = templates[round_index]
            question = template.format(topic=topic)
            normalized = question.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            questions.append(
                {
                    "question": question,
                    "topic": topic,
                    "source": source,
                    "page": page,
                    "score": round(score, 3),
                }
            )
            if len(questions) >= limit:
                return questions

    return questions


def _extract_topics(chunks: list[Chunk]) -> list[tuple[str, float, str, int | None]]:
    counts: Counter[str] = Counter()
    metadata: dict[str, tuple[str, int | None]] = {}

    for chunk in chunks:
        candidates = _candidate_topics(chunk.text)
        for candidate in candidates:
            counts[candidate] += _topic_weight(candidate, chunk.text)
            metadata.setdefault(candidate, (chunk.source, chunk.page))

    ranked = []
    for topic, score in counts.most_common(80):
        source, page = metadata[topic]
        ranked.append((topic, float(score), source, page))
    return ranked


def _candidate_topics(text: str) -> list[str]:
    candidates: list[str] = []
    lines = [line.strip(" :-•\t") for line in text.splitlines() if line.strip()]
    for line in lines[:18]:
        if 3 <= len(line.split()) <= 8 and not line.endswith("."):
            cleaned = _clean_topic(line)
            if cleaned:
                candidates.append(cleaned)

    for match in re.finditer(r"\b(?:Purpose|Forms|See Also):\s*([^.\n]{8,80})", text, re.I):
        cleaned = _clean_topic(match.group(1))
        if cleaned:
            candidates.append(cleaned)

    noun_phrases = re.findall(
        r"\b(?:Employee|Human Resources|Manager|Supervisor|Network|HIPAA|Health|Identification|Security|Benefits|Leave|Termination|Separation|Performance|Privacy|Complaint|Access)\s+[A-Za-z][A-Za-z\s/-]{2,55}",
        text,
    )
    for phrase in noun_phrases[:12]:
        cleaned = _clean_topic(phrase)
        if cleaned:
            candidates.append(cleaned)

    return candidates


def _clean_topic(value: str) -> str | None:
    value = HEADING_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s/&-]", "", value).strip(" -/")
    words = value.split()
    if not 2 <= len(words) <= 9:
        return None
    lowered = value.lower()
    if lowered in {"human resources standard operating procedure manual", "practices procedures"}:
        return None
    if sum(ch.isalpha() for ch in value) < 8:
        return None
    return lowered


def _topic_weight(topic: str, text: str) -> int:
    weight = 1
    if topic.title() in text[:400] or topic.upper() in text[:400]:
        weight += 2
    if any(term in topic for term in ["badge", "hipaa", "privacy", "benefit", "leave", "termination", "separation", "ishelp"]):
        weight += 3
    return weight


def _templates_for_topic(topic: str) -> list[str]:
    if any(term in topic for term in ["fee", "badge", "benefit", "leave", "termination", "separation"]):
        return QUESTION_TEMPLATES
    return QUESTION_TEMPLATES[:5]


def generate_document_questions(chunks: list[Chunk], limit: int = 100) -> list[dict]:
    """Document-specific prompts without HR assumptions or invented requirements."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk)
    questions = []
    for source, items in grouped.items():
        text = " ".join(item.text for item in items)
        prompts = [f'What are the key takeaways from "{source}"?',
                   f'Summarize "{source}" in plain language.']
        if re.search(r'\b(deadline|due|by \d|january|february|march|april|may|june|july|august|september|october|november|december)\b', text, re.I):
            prompts.append(f'What dates or deadlines are mentioned in "{source}"?')
        if re.search(r'\b(must|required|shall|responsib|action|steps|procedure)\w*', text, re.I):
            prompts.append(f'What actions or requirements are described in "{source}"?')
        if re.search(r'\b(risk|warning|danger|caution|failure|hazard)\w*', text, re.I):
            prompts.append(f'What risks or warnings are described in "{source}"?')
        prompts.append(f'Explain the main topics in "{source}" with supporting citations.')
        for prompt in prompts:
            questions.append({'question': prompt, 'topic': source, 'source': source, 'page': None})
    return questions[:limit]
