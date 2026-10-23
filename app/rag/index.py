from __future__ import annotations

import hashlib
import math
import pickle
import os
import uuid
import faiss
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from .embeddings import encode, MODEL_NAME

from .loaders import discover_files, load_documents
from .models import Chunk, Document, RetrievalHit


WORD_RE = re.compile(r"\w+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
QUERY_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "how",
    "is",
    "me",
    "more",
    "of",
    "the",
    "tell",
    "that",
    "this",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "would",
}
SCOPE_ALLOW_TERMS = {
    "badge",
    "badges",
    "benefits",
    "employee",
    "employees",
    "hipaa",
    "hr",
    "ishelp",
    "leave",
    "network",
    "policy",
    "privacy",
    "procedure",
    "sop",
    "sops",
}
MIN_CHUNK_WORDS = 12


def normalized_phrase(text: str) -> str:
    text = unicodedata.normalize('NFC', text).casefold()
    text = text.replace('’', "'").replace('‘', "'").replace('ʼ', "'")
    return re.sub(r"\s+", ' ', text).strip()


def requested_phrase(query: str) -> str | None:
    quoted = re.search(r'["“«]([^"”»]{2,200})["”»]', query)
    if quoted:
        return normalized_phrase(quoted.group(1))
    match = re.search(r"(?:how (?:do|would|can) (?:you|i|we) say|how to say|translate|comment (?:dit-on|dire)|traduire)\s+(.+)", query, re.I)
    if not match:
        return None
    phrase = re.split(r'\s+(?:in|into|en)\s+', match.group(1), maxsplit=1, flags=re.I)[0]
    return normalized_phrase(phrase.strip(' ?.!"“”«»')) or None


class RagIndex:
    SCHEMA = 3

    def __init__(self, data_dir: Path, storage_path: Path):
        self.data_dir = data_dir
        self.storage_path = storage_path
        self.chunks: list[Chunk] = []
        self.faiss_index = None
        self.vectors = None
        self._fingerprint = ""
        self._vocabulary_terms = set()
        self._embedding_cache = {}

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self.faiss_index is not None

    def load_or_build(self) -> None:
        current = self._data_fingerprint()
        if self.storage_path.exists():
            try:
                with self.storage_path.open('rb') as handle:
                    payload = pickle.load(handle)
                if payload.get('schema') == self.SCHEMA and payload.get('embedding_model') == MODEL_NAME:
                    self._embedding_cache = payload.get('embedding_cache', {})
                    if payload.get('fingerprint') == current:
                        self.chunks = payload['chunks']
                        self.faiss_index = faiss.deserialize_index(payload['faiss']) if payload['faiss'] is not None else None
                        self.vectors = payload['vectors']
                        self._vocabulary_terms = set(payload['vocabulary_terms'])
                        self._fingerprint = current
                        if self.chunks and self.faiss_index.ntotal != len(self.chunks):
                            raise ValueError('Index metadata mismatch')
                        return
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, RuntimeError):
                pass  # Rebuild old or incomplete cached indexes from source documents.
        self.rebuild()

    def rebuild(self, progress=None) -> None:
        documents = []
        for file_path in discover_files(self.data_dir):
            documents.extend(load_documents(file_path, self.data_dir))
        if progress:
            progress('chunking', 'Splitting document text into overlapping passages')
        chunks = chunk_documents(documents)
        vectors, index, retained = None, None, {}
        if chunks:
            if progress:
                progress('embedding', f'Creating semantic embeddings for {len(chunks)} passages')
            keys = [hashlib.sha256(c.text.encode()).hexdigest() for c in chunks]
            missing = dict((key, c.text) for key, c in zip(keys, chunks) if key not in self._embedding_cache)
            cache = dict(self._embedding_cache)
            if missing:
                new_vectors = encode(list(missing.values()))
                cache.update(zip(missing, new_vectors))
            vectors = np.ascontiguousarray([cache[key] for key in keys], dtype='float32')
            faiss.normalize_L2(vectors)
            retained = {key: vector for key, vector in zip(keys, vectors)}
            if progress:
                progress('indexing', 'Building and saving the FAISS vector index')
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        elif progress:
            progress('embedding', 'No readable passages to embed')
            progress('indexing', 'Saving empty workspace index')
        fingerprint = self._data_fingerprint()
        vocabulary = set(WORD_RE.findall(' '.join(c.text for c in chunks).lower()))
        payload = {'schema':self.SCHEMA, 'embedding_model':MODEL_NAME, 'fingerprint':fingerprint,
                   'chunks':chunks, 'faiss':faiss.serialize_index(index) if index is not None else None,
                   'vectors':vectors, 'embedding_cache':retained, 'vocabulary_terms':vocabulary}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.storage_path.with_name(self.storage_path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temp.open('wb') as handle:
                pickle.dump(payload, handle)
            os.replace(temp, self.storage_path)
        finally:
            temp.unlink(missing_ok=True)
        self.chunks, self.faiss_index, self.vectors = chunks, index, vectors
        self._fingerprint, self._vocabulary_terms = fingerprint, vocabulary
        self._embedding_cache = retained

    def search(self, query: str, limit: int = 8, candidate_pool: int = 40, phrase: str | None = None, all_meanings: bool = False, search_phrases: list[str] | None = None) -> list[RetrievalHit]:
        if not self.ready or not query.strip():
            return []
        expansions = list(dict.fromkeys(search_phrases or []))[:6]
        query_vectors = encode([query] + expansions)
        scores, ids = self.faiss_index.search(query_vectors, min(max(candidate_pool, limit), len(self.chunks)))
        relevance = {}
        for row_scores, row_ids in zip(scores,ids):
            for i,score in zip(row_ids,row_scores):
                if i >= 0: relevance[int(i)] = max(relevance.get(int(i), -1.), float(score))
        phrase = normalized_phrase(phrase) if phrase else requested_phrase(query)
        targets = ([phrase] if phrase else []) + [normalized_phrase(p) for p in expansions]
        exact = {}
        target_matches = {}
        if targets:
            patterns = [(target,re.compile(r"(?<!\w)"+re.escape(target)+r"(?!\w)")) for target in dict.fromkeys(targets)]
            for i,chunk in enumerate(self.chunks):
                text = normalized_phrase(chunk.text)
                for target,pattern in patterns:
                    match = pattern.search(text)
                    if match:
                        target_matches.setdefault(target, []).append(i)
                        standalone = bool(re.match(r"\s*[.;!?](?:\s|$)",text[match.end():]))
                        strength = (4 if target == phrase else 2) + int(standalone)
                        exact[i] = max(exact.get(i,0),strength)
                        if i not in relevance: relevance[i] = float(np.max(self.vectors[i] @ query_vectors.T))
        if all_meanings:
            # Scan every passage for the phrase, rather than stopping at top_k.
            # Include adjacent chunks to recover entries split at chunk boundaries.
            seeds = sorted(exact, key=lambda i: (-exact[i], -relevance[i])) if exact else list(relevance)[:limit]
            # Cover each lookup component before spending context on repeated
            # senses of one word. Budget fitting retains this ordered prefix.
            leaders = []
            for target in dict.fromkeys(targets):
                matches = target_matches.get(target, [])
                if matches:
                    best = max(matches, key=lambda i: (exact[i], relevance[i]))
                    if best not in leaders: leaders.append(best)
            seeds = leaders + [i for i in seeds if i not in leaders]
            expanded, seen, used_chars = [], set(), 0
            self.meaning_coverage = {'matching_passages':len(exact), 'included_matches':0, 'limited':False}
            # Direct matches precede neighboring pages, which often contain
            # unrelated dictionary entries.
            for neighbors in (False, True):
                for seed in seeds:
                    for i in ((seed - 1, seed + 1) if neighbors else (seed,)):
                        if i < 0 or i >= len(self.chunks) or i in seen: continue
                        if i != seed and (self.chunks[i].path != self.chunks[seed].path or
                                          abs((self.chunks[i].page or 0) - (self.chunks[seed].page or 0)) > 1): continue
                        if used_chars + len(self.chunks[i].text) > 60000:
                            self.meaning_coverage['limited'] = True
                            continue
                        seen.add(i); expanded.append(i); used_chars += len(self.chunks[i].text)
                        if i in exact: self.meaning_coverage['included_matches'] += 1
                        if i not in relevance: relevance[i] = float(np.max(self.vectors[i] @ query_vectors.T))
            return [RetrievalHit(self.chunks[i], relevance[i], rank) for rank, i in enumerate(expanded,1)]
        candidates = list(relevance)
        selected = []
        # MMR selects relevant evidence while limiting near-duplicate passages.
        while candidates and len(selected) < limit:
            def mmr(i):
                redundancy = max((float(self.vectors[i] @ self.vectors[j]) for j in selected), default=0.)
                return 2 * exact.get(i, 0) + .8 * relevance[i] - .2 * redundancy
            best = max(candidates, key=mmr)
            candidates.remove(best)
            if relevance[best] > 0 or best in exact:
                selected.append(best)
        return [RetrievalHit(self.chunks[i], relevance[i], rank) for rank, i in enumerate(selected, 1)]

    def is_in_scope(self, query: str) -> bool:
        return bool(self.search(query, limit=1))

    def stats(self) -> dict[str, Any]:
        return {'files':len(discover_files(self.data_dir)), 'chunks':len(self.chunks),
                'sources':sorted({c.source for c in self.chunks}), 'fingerprint':self._fingerprint,
                'ready':self.ready, 'retrieval':{'engine':'FAISS', 'index_type':'IndexFlatIP',
                'embedding_model':MODEL_NAME, 'dimensions':self.faiss_index.d if self.faiss_index is not None else None,
                'metric':'cosine', 'vectors':self.faiss_index.ntotal if self.faiss_index is not None else 0}}

    def _data_fingerprint(self) -> str:
        digest = hashlib.sha256(f'{self.SCHEMA}:{MODEL_NAME}'.encode())
        for path in discover_files(self.data_dir):
            stat = path.stat()
            digest.update(str(path.relative_to(self.data_dir)).encode())
            digest.update(f'{stat.st_mtime_ns}:{stat.st_size}'.encode())
        return digest.hexdigest()


def chunk_documents(documents: list[Document], target_words: int = 140, overlap_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = _paragraphs(document.text)
        buffer: list[str] = []
        buffer_words = 0
        chunk_number = 0

        for paragraph in paragraphs:
            words = WORD_RE.findall(paragraph)
            if not words:
                continue
            if buffer and buffer_words + len(words) > target_words:
                chunk_number += 1
                chunk = _make_chunk(document, chunk_number, buffer)
                if len(WORD_RE.findall(chunk.text)) >= MIN_CHUNK_WORDS:
                    chunks.append(chunk)
                buffer = _overlap_tail(buffer, overlap_words)
                buffer_words = sum(len(WORD_RE.findall(item)) for item in buffer)
            buffer.append(paragraph)
            buffer_words += len(words)

        if buffer:
            chunk_number += 1
            chunk = _make_chunk(document, chunk_number, buffer)
            if len(WORD_RE.findall(chunk.text)) >= MIN_CHUNK_WORDS:
                chunks.append(chunk)

    return chunks


def _paragraphs(text: str) -> list[str]:
    parts = []
    for paragraph in re.split(r"\n\s*\n+", text):
        words = paragraph.split()
        parts.extend(' '.join(words[start:start+100]) for start in range(0, len(words), 100))
    return parts


def _overlap_tail(paragraphs: list[str], target_words: int) -> list[str]:
    words = ' '.join(paragraphs).split()
    return [' '.join(words[-target_words:])] if target_words and words else []


def _make_chunk(document: Document, chunk_number: int, paragraphs: list[str]) -> Chunk:
    text = "\n\n".join(paragraphs).strip()
    digest = hashlib.sha1(f"{document.path}:{document.page}:{chunk_number}:{text[:80]}".encode("utf-8")).hexdigest()[:12]
    return Chunk(
        id=digest,
        text=text,
        source=document.source,
        path=document.path,
        page=document.page,
        section=f"chunk {chunk_number}",
        token_estimate=max(1, len(WORD_RE.findall(text)) * 4 // 3),
    )


def expand_query(query: str) -> str:
    normalized = query.lower()
    expansions: list[str] = []
    if "hipaa" in normalized or ("privacy" in normalized and "right" in normalized):
        expansions.append(
            "protected health information PHI access inspect copy amendment amend confidential communication "
            "restrictions disclosure accounting complaint authorization retaliation"
        )
    if "sop" in normalized or "procedure" in normalized or "process" in normalized:
        expansions.append("purpose scope responsibility responsible procedure forms policy steps")
    if "badge" in normalized or "badges" in normalized:
        expansions.append(
            "identification badge employee identification badges electronic access badge lost badge replacement fee "
            "ten dollars human resources security access separation returned displayed shared"
        )
    return " ".join([query, *expansions])
