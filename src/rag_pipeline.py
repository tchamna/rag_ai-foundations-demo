import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle

import numpy as np
import pandas as pd 

import unicodedata

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI

from config import (
    DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, GENERATION_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIM_THRESHOLD
)




# ----------------- Utils -----------------
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12"
}

def normalize_months(query: str) -> str:
    q_lower = query.lower()
    for name, num in MONTH_MAP.items():
        if name in q_lower:
            q_lower = q_lower.replace(name, num)
    return q_lower

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _normalize_text(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clean_phrase(s: str) -> str:
    """Remove leading numbering and surrounding artifacts from phrasebook lines."""
    if not s:
        return s
    out = s.strip()
    # remove leading numeric markers like '520)' or '520) '
    out = re.sub(r"^\s*\d+\)\s*", "", out)
    out = re.sub(r"^\s*\d+[:\-]\s*", "", out)
    # remove stray leading labels if present (e.g., 'English:')
    out = re.sub(r"^English:\s*", "", out, flags=re.I)
    out = out.strip()
    return out

def _is_translation_query(q: str) -> bool:
    ql = q.lower()
    triggers = (
        "translate ", "how do you say", "how to say",
        "say '", "say “", "say \"", "in fe'efe'e", "in feefe'e"
    )
    return any(t in ql for t in triggers)

def _english_target_from_query(q: str) -> str:
    # Extract the English phrase from queries like:
    # "translate where are you", "how do you say 'where are you' in fe'efe'e?"
    t = re.sub(r"(?i)translate\s+", "", q)
    t = re.sub(r"(?i)how\s+do\s+you\s+say\s+", "", t)
    t = re.sub(r"(?i)how\s+to\s+say\s+", "", t)
    t = re.sub(r"(?i)\s+in\s+fe[’'´`]?efe[’'´`]?e\??", "", t)
    t = t.strip(" ?\"'“”‘’")
    return t

def _lexical_search_docs(docs, term: str, limit: int = 50):
    """
    Substring search that mirrors your Vector DB browser behavior,
    but accent/case insensitive. Keeps original corpus order.
    """
    if not term:
        return []
    norm_term = _normalize_text(term)
    out_idxs = []
    for i, d in enumerate(docs):
        norm_txt = _normalize_text(d["text"])
        if norm_term in norm_txt:
            out_idxs.append(i)
            if len(out_idxs) >= limit:
                break
    return out_idxs


# ----------------- File Readers -----------------

def _read_text_file(fp: Path) -> str:
    return fp.read_text(encoding="utf-8")

def _read_csv_faqs(fp: Path) -> List[str]:
    df = pd.read_csv(fp)
    rows = []
    if "question" in df.columns and "answer" in df.columns:
        for _, r in df.iterrows():
            rows.append(f"Q: {r['question']}\nA: {r['answer']}")
    return rows

def _read_xlsx_faqs(fp: Path) -> List[str]:
    df = pd.read_excel(fp)
    rows = []
    if "question" in df.columns and "answer" in df.columns:
        for _, r in df.iterrows():
            rows.append(f"Q: {r['question']}\nA: {r['answer']}")
    return rows

# ----------------- Corpus Loader -----------------

def load_corpus(data_dir: Path) -> List[Dict[str, Any]]:
    corpus = []
    for p in data_dir.glob("**/*"):
        if p.is_dir():
            continue

        if p.suffix.lower() == ".txt":
            lines = p.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                if "|" in line:  # phrasebook style
                    parts = [seg.strip() for seg in line.split("|")]
                    if len(parts) == 3:
                        text = f"English: {parts[0]}\nFe'efe'e: {parts[1]}\nFrench: {parts[2]}"
                    else:
                        text = line.strip()
                else:
                    text = line.strip()
                corpus.append({
                    "id": f"{p}#{i}",
                    "text": text,
                    "meta": {"source": f"{p.name} (line {i+1})"}
                })

        elif p.suffix.lower() == ".csv" and "faq" in p.name.lower():
            df = pd.read_csv(p)
            for idx, (_, r) in enumerate(df.iterrows(), start=1):
                q = str(r.get("question", "")).strip()
                a = str(r.get("answer", "")).strip()
                if q and a:
                    text = f"Q: {q}\nA: {a}"
                    corpus.append({
                        "id": f"{p}#{idx}",
                        "text": text,
                        "meta": {"source": f"{p.name} (row {idx})"}
                    })

        elif p.suffix.lower() == ".xlsx":
            df = pd.read_excel(p)
            for idx, (_, r) in enumerate(df.iterrows(), start=1):
                q = str(r.get("question", "")).strip()
                a = str(r.get("answer", "")).strip()
                if q and a:
                    text = f"Q: {q}\nA: {a}"
                    corpus.append({
                        "id": f"{p}#{idx}",
                        "text": text,
                        "meta": {"source": f"{p.name} (row {idx})"}
                    })

    return corpus

# ----------------- Chunking -----------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
        if i <= 0:
            break
    return chunks

def build_documents(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for item in corpus:
        chunks = chunk_text(item["text"])
        for idx, ch in enumerate(chunks):
            docs.append({
                "id": f"{item['id']}::chunk{idx}",
                "text": ch,
                "meta": item["meta"]
            })
    return docs

# ----------------- Vector Index -----------------

class VectorIndex:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.docs: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def build(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        vectors = self._embed([d["text"] for d in docs]).astype("float32")
        d = vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        q = self._embed([query]).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self, vs_dir: Path):
        vs_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(vs_dir / "faiss.index"))
        with open(vs_dir / "docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, vs_dir: Path):
        self.index = faiss.read_index(str(vs_dir / "faiss.index"))
        with open(vs_dir / "docs.pkl", "rb") as f:
            self.docs = pickle.load(f)

# ----------------- Re-Ranker -----------------

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Pick a safe device for the CrossEncoder. Prefer CUDA/MPS when available,
        # otherwise fall back to CPU. If loading the model fails for any reason
        # (in constrained cloud runtimes), keep model=None so callers can skip
        # reranking instead of crashing the app.
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            device = "cpu"

        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            # Avoid crashing the whole app if the reranker can't be loaded in this environment.
            print(f"Warning: failed to load CrossEncoder reranker on device={device}: {e}")
            self.model = None

    def rerank(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        if not self.model:
            # Reranker not available in this environment; return contexts unchanged.
            return contexts

        pairs = [(query, c["text"]) for c in contexts]
        scores = self.model.predict(pairs)

        for c, s in zip(contexts, scores):
            c["rerank_score"] = float(s)

        return sorted(contexts, key=lambda x: x["rerank_score"], reverse=True)

# ----------------- Generator -----------------

class Generator:
    def __init__(self, model_name: str = GENERATION_MODEL, use_chatgpt: bool = False):
        self.use_chatgpt = use_chatgpt
        if use_chatgpt:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.use_chatgpt:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------- Prompt -----------------

def format_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return f"""You are a Fe'efe'e language tutor.
Question: {query}

No sources were found. Answer: I could not find this in the documents."""

    cited = "\n\n".join(
        [f"[{i+1}] Source: {c['meta']['source']}\n{c['text']}" for i, c in enumerate(contexts)]
    )
    return f"""You are a Fe'efe'e language assistant. 
Always prioritize answers in Fe'efe'e. If relevant, add the French translation after.
Do not invent information not in the sources.

Question: {query}

Sources:
{cited}

Answer (Fe'efe'e first, optional French, with citations like [1], [2]):"""

# ----------------- RAG Pipeline -----------------
from sentence_transformers import CrossEncoder
from openai import OpenAI
import os

def normalize_query(query: str) -> str:
    """
    Normalize user queries to better match stored chunk formats.
    Example: "translate where are you" -> "English: where are you"
    """
    q = query.lower().strip()
    if q.startswith("translate "):
        q = q.replace("translate ", "").strip()
        return f"English: {q}"
    return query

class RAGPipeline:
    
    def __init__(
        self,
        vs_dir: Path = VECTORSTORE_DIR,
        use_chatgpt: bool = False,
        use_reranker: bool = True,
        refine_phrasebook_with_gpt: bool = False, 
    ):
        self.vs_dir = vs_dir
        self.index = VectorIndex()
        self.use_chatgpt = use_chatgpt
        self.refine_phrasebook_with_gpt = refine_phrasebook_with_gpt
        self.generator = Generator(model_name=GENERATION_MODEL, use_chatgpt=use_chatgpt)

        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    def ensure_loaded(self):
        if not (self.vs_dir / "faiss.index").exists():
            raise FileNotFoundError("Vector store not found. Run ingest.py first.")
        self.index.load(self.vs_dir)

    
    # def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    #     # 🔑 Normalize query for months (e.g., "June" → "06")
    #     query = normalize_months(query)

    #     results = self.index.search(query, k)
    #     ctxs = [self.index.docs[i] | {"score": s} for i, s in results]

    #     # ✅ Apply reranker if enabled
    #     if self.use_reranker and ctxs:
    #         pairs = [(query, c["text"]) for c in ctxs]
    #         scores = self.reranker.predict(pairs)
    #         for j, score in enumerate(scores):
    #             ctxs[j]["rerank_score"] = float(score)
    #         ctxs = sorted(ctxs, key=lambda x: x["rerank_score"], reverse=True)

    #     return ctxs


    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: combines FAISS semantic search with lexical matching.
        Ensures numeric/date queries (like '06', '07/05', 'June') return exact matches.
        """
        docs = self.index.docs
        ctxs = []

        # --------- Detect if query looks like a number/date ----------
        def is_date_or_number(q: str) -> bool:
            return bool(re.search(r"\b\d{1,2}(/|-)\d{1,2}\b", q)) or \
                bool(re.search(r"\b\d{2,4}\b", q)) or \
                any(m in q.lower() for m in ["january","february","march","april","may","june",
                                                "july","august","september","october","november","december"])

        def normalize(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))

        lexical_idxs = []
        if is_date_or_number(query):
            norm_term = normalize(query)
            for i, d in enumerate(docs):
                if norm_term in normalize(d["text"]):
                    lexical_idxs.append(i)

        # --------- Semantic FAISS search ----------
        faiss_results = self.index.search(query, top_k=50)
        semantic_idxs = [i for i, _ in faiss_results]

        # --------- Merge ----------
        merged = list(dict.fromkeys(lexical_idxs + semantic_idxs))  # keep order

        # Build a map of semantic scores from faiss_results (if any). faiss_results
        # is a list of (idx, score) tuples; higher is better. Use that score for
        # semantic entries, and give lexical exact matches a boost on top.
        faiss_score_map: Dict[int, float] = {i: float(s) for i, s in faiss_results}

        ctxs = []
        for i in merged:
            # Base score: prefer FAISS-provided score when available, otherwise 0.0
            base_sem_score = faiss_score_map.get(i, 0.0)

            # If this doc was found by lexical matching, give it a fixed boost
            # so exact/lexical matches rank higher than similar semantic hits.
            if i in lexical_idxs:
                score = 1.0 + base_sem_score  # lexical + any semantic relevance
            else:
                    # For semantic-only hits, scale FAISS cosine/dot scores into [0,1].
                    # When embeddings are normalized, FAISS returns cosine (in [-1,1]).
                    # Map that to [0,1] with a linear transform so UI thresholds work
                    # sensibly: score = 0.5 + 0.5 * base_sem_score.
                    raw = float(base_sem_score)
                    scaled = 0.5 + 0.5 * raw
                    # clamp to [0,1]
                    score = max(0.0, min(1.0, scaled))

            ctxs.append(docs[i] | {"score": score})

        # --------- Rerank (if enabled) ----------
        if getattr(self, "use_reranker", False) and ctxs:
            pairs = [(query, c["text"]) for c in ctxs]
            scores = self.reranker.predict(pairs)
            for j, s in enumerate(scores):
                ctxs[j]["rerank_score"] = float(s)
            ctxs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return ctxs[:k]

    def answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not contexts:
            return {
                "answer": "I could not find this in the documents.",
                "contexts": [],
                "prompt": f"Question: {query}\n\nNo strong matches found.",
                "mode": "no_context",
            }

        # Limit to top contexts (to reduce noise)
        top_contexts = contexts[:3]

        # ------------------------------
        # FAQ-style shortcut
        # ------------------------------
        for i, c in enumerate(top_contexts):
            text = c["text"]
            if text.strip().startswith("Q:") and "A:" in text:
                qna_parts = text.split("A:", 1)
                if len(qna_parts) == 2:
                    ans = qna_parts[1].strip()
                    full_prompt = f"""
                    [FAQ Direct Match]

                    Question: {query}

                    Matched Source: {c['meta']['source']}
                    Context: {text}

                    Extracted Answer: {ans}
                    """
                    return {
                        "answer": f"{ans} [{i+1}]",
                        "contexts": top_contexts,
                        "prompt": full_prompt.strip(),
                        "mode": "faq",
                    }

        # ------------------------------
        # Phrasebook style (Fe'efe'e first)
        # ------------------------------
        for i, c in enumerate(top_contexts):
            text = c["text"]
            if "Fe'efe'e:" in text:
                # Try to extract English, Fe'efe'e and French parts where present
                eng = None
                feefee = None
                fr = None
                # The text format is usually like:
                # English: ...\nFe'efe'e: ...\nFrench: ...
                if "English:" in text:
                    eng = text.split("English:", 1)[1].split("Fe'efe'e:", 1)[0].strip()
                if "Fe'efe'e:" in text:
                    feefee = text.split("Fe'efe'e:", 1)[1]
                    if "French:" in feefee:
                        feefee, fr_part = feefee.split("French:", 1)
                        fr = fr_part.strip()
                    feefee = feefee.strip()
                # Build answer including available parts
                # Clean parts and produce a polished sentence
                feefee_clean = _clean_phrase(feefee) if feefee else None
                eng_clean = _clean_phrase(eng) if eng else None
                fr_clean = _clean_phrase(fr) if fr else None

                # Build human-friendly answer: Fe'efe'e sentence followed by
                # parenthetical English and French if available.
                if feefee_clean:
                    if eng_clean or fr_clean:
                        tail = " — ".join([p for p in (eng_clean, fr_clean) if p])
                        answer = f"{feefee_clean} ({tail})"
                    else:
                        answer = feefee_clean
                else:
                    # fallback to combining available translations
                    answer = " — ".join([p for p in (eng_clean, fr_clean) if p])

                full_prompt = f"""
                [Phrasebook Direct Match]

                Question: {query}

                Matched Source: {c['meta']['source']}
                Context: {text}

                Extracted Fe'efe'e Answer: {answer}
                """

                # If ChatGPT is enabled, ask it to rewrite the phrasebook
                # content into a polished, natural sentence (Fe'efe'e first,
                # then English and French in parentheses). Keep faithful to
                # the original content and remove numbering/artifacts.
                if self.use_chatgpt:
                    rp = f"""
You are a helpful assistant. Rewrite the following phrasebook entry into a polished, natural-sounding sentence.
Requirements:
- Start with the Fe'efe'e sentence (exact or slight normalization).
- Then include the English and French translations in parentheses, separated by ' — ' if both are present.
- Remove any numbering or chunk artifacts. Do NOT add or invent content.

Input: {answer}

Output:
"""
                    try:
                        polished = self.generator.generate(rp, max_new_tokens=120).strip()
                        if polished:
                            answer = polished
                            full_prompt += "\n\n[Refined with GPT]"
                    except Exception:
                        pass

                return {
                    "answer": f"{answer} [{i+1}]",
                    "contexts": top_contexts,
                    "prompt": full_prompt.strip(),
                    "mode": "phrasebook",
                }

        # ------------------------------
        # GPT branch (if enabled)
        # ------------------------------
        if self.use_chatgpt:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # When using ChatGPT, explicitly ask for an elaborated answer that
            # re-uses full sentences from the provided contexts. Request clear
            # citations like [1], [2] and a short summary sentence at the top.
            cited = "\n\n".join(
                [f"[{i+1}] {c['text']} (Source: {c['meta']['source']})" for i, c in enumerate(top_contexts)]
            )

            prompt = f"""
You are a helpful, precise assistant. Use ONLY the context below; do NOT invent facts.

Task:
- Produce a concise answer (1-3 sentences) at the top that directly responds to the question.
- Then, elaborate using full sentences drawn from the contexts. When you reuse text from a context, quote or paraphrase it faithfully and include its citation like [1] after the sentence.
- Keep the tone factual and neutral. If the documents do not contain the answer, say: "I could not find this in the documents." Do not hallucinate.

Question: {query}

Contexts:
{cited}

Answer:
1) Short direct answer:
2) Elaboration (use full sentences from the sources with citations):
"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700
            )
            return {
                "answer": resp.choices[0].message.content.strip(),
                "contexts": top_contexts,
                "prompt": prompt,
                "mode": "gpt",
            }

        # ------------------------------
        # Fallback: local FLAN-T5
        # ------------------------------
        cited = "\n\n".join(
            [f"[{i+1}] {c['text']} (Source: {c['meta']['source']})" for i, c in enumerate(top_contexts)]
        )
        prompt = f"""
    You are a precise assistant.
    Use ONLY the context below. Do not invent information.
    Return a clear sentence that answers the question AND include citations like [1], [2].
    If the answer is not explicit, say: "I could not find this in the documents."

    Question: {query}

    Context:
    {cited}

    Answer:"""

        completion = self.generator.generate(prompt).strip()

        # Guard against citation-only answers like "[1] [2]"
        import re
        if not completion or re.fullmatch(r"(?:\s*\[\d+\]\s*)+", completion):
            completion = f"{top_contexts[0]['text']} [1]"
            mode = "fallback"
        else:
            mode = "flan"

        return {
            "answer": completion,
            "contexts": top_contexts,
            "prompt": prompt,
            "mode": mode,
        }

    
        if not contexts:
            return {
                "answer": "I could not find this in the documents.",
                "contexts": [],
                "prompt": f"Question: {query}\n\nNo strong matches found."
            }

        # ------------------------------
        # FAQ-style shortcut
        # ------------------------------
        for i, c in enumerate(contexts):
            text = c["text"]
            if text.strip().startswith("Q:") and "A:" in text:
                qna_parts = text.split("A:", 1)
                if len(qna_parts) == 2:
                    ans = qna_parts[1].strip()
                    full_prompt = f"""
                    [FAQ Direct Match]

                    Question: {query}

                    Matched Source: {c['meta']['source']}
                    Context: {text}

                    Answer: {ans}
                    """
                    return {
                        "answer": f"{ans} [{i+1}]",
                        "contexts": contexts,
                        "prompt": full_prompt.strip()
                    }


        # ------------------------------
        # Phrasebook style (Fe'efe'e first)
        # ------------------------------
        for i, c in enumerate(contexts):
            text = c["text"]
            if "Fe'efe'e:" in text:
                eng = None
                feefee = None
                fr = None
                if "English:" in text:
                    eng = text.split("English:", 1)[1].split("Fe'efe'e:", 1)[0].strip()
                if "Fe'efe'e:" in text:
                    feefee = text.split("Fe'efe'e:", 1)[1]
                    if "French:" in feefee:
                        feefee, fr_part = feefee.split("French:", 1)
                        fr = fr_part.strip()
                    feefee = feefee.strip()

                feefee_clean = _clean_phrase(feefee) if feefee else None
                eng_clean = _clean_phrase(eng) if eng else None
                fr_clean = _clean_phrase(fr) if fr else None

                if feefee_clean:
                    if eng_clean or fr_clean:
                        tail = " — ".join([p for p in (eng_clean, fr_clean) if p])
                        answer = f"{feefee_clean} ({tail})"
                    else:
                        answer = feefee_clean
                else:
                    answer = " — ".join([p for p in (eng_clean, fr_clean) if p])

                full_prompt = f"""
                [Phrasebook Direct Match]

                Question: {query}

                Matched Source: {c['meta']['source']}
                Context: {text}

                Extracted Answer: {answer}
                """

                # If ChatGPT is enabled here, refine the answer similarly
                if self.use_chatgpt:
                    rp = f"""
You are a helpful assistant. Rewrite the following phrasebook entry into a polished, natural-sounding sentence.
Requirements:
- Start with the Fe'efe'e sentence (exact or slight normalization).
- Then include the English and French translations in parentheses, separated by ' — ' if both are present.
- Remove any numbering or chunk artifacts. Do NOT add or invent content.

Input: {answer}

Output:
"""
                    try:
                        polished = self.generator.generate(rp, max_new_tokens=120).strip()
                        if polished:
                            answer = polished
                    except Exception:
                        pass

                return {
                    "answer": f"{answer} [{i+1}]",
                    "contexts": contexts,
                    "prompt": full_prompt.strip()
                }

        # ------------------------------
        # GPT branch (if enabled)
        # ------------------------------
        if self.use_chatgpt:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            context_text = "\n\n".join(
                [f"{c['text']} (Source: {c['meta']['source']})" for c in contexts]
            )

            prompt = f"""
You are a precise assistant.
Use ONLY the context below. Do not invent information.
Return a clear sentence that answers the question AND include citations like [1], [2].
If the answer is not explicit, say: "I could not find this in the documents."

Question: {query}

Context:
{context_text}

Answer:
"""

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return {
                "answer": resp.choices[0].message.content.strip(),
                "contexts": contexts,
                "prompt": prompt
            }

        # ------------------------------
        # Default fallback (no GPT)
        # ------------------------------
        top_context = contexts[0]
        return {
            "answer": f"{top_context['text']} [1]",
            "contexts": contexts,
            "prompt": f"Fallback: returning top chunk.\n\nQuestion: {query}"
        }

        if not contexts:
            return {
                "answer": "I could not find this in the documents.",
                "contexts": [],
                "prompt": f"Question: {query}\n\nNo strong matches found.",
                "mode": "no_context",
            }

        top_contexts = contexts[:3]

        # 1) FAQ shortcut
        for i, c in enumerate(top_contexts):
            txt = c["text"]
            if "Q:" in txt and "A:" in txt:
                ans = txt.split("A:", 1)[1].strip()
                return {
                    "answer": f"{ans} [{i+1}]",
                    "contexts": top_contexts,
                    "prompt": "Direct FAQ match",
                    "mode": "faq",
                }

        # 2) Phrasebook shortcut
        for i, c in enumerate(top_contexts):
            txt = c["text"]
            if "Fe'efe'e:" in txt:
                eng = None
                feefee = None
                fr = None
                if "English:" in txt:
                    eng = txt.split("English:", 1)[1].split("Fe'efe'e:", 1)[0].strip()
                if "Fe'efe'e:" in txt:
                    feefee = txt.split("Fe'efe'e:", 1)[1]
                    if "French:" in feefee:
                        feefee, fr_part = feefee.split("French:", 1)
                        fr = fr_part.strip()
                    feefee = feefee.strip()

                feefee_clean = _clean_phrase(feefee) if feefee else None
                eng_clean = _clean_phrase(eng) if eng else None
                fr_clean = _clean_phrase(fr) if fr else None

                if feefee_clean:
                    if eng_clean or fr_clean:
                        tail = " — ".join([p for p in (eng_clean, fr_clean) if p])
                        answer = f"{feefee_clean} ({tail})"
                    else:
                        answer = feefee_clean
                else:
                    answer = " — ".join([p for p in (eng_clean, fr_clean) if p])

                mode = "phrasebook"
                # Optional: let GPT lightly polish (disabled by default)
                if self.use_chatgpt and self.refine_phrasebook_with_gpt:
                    rp = f"""You are a strict Fe'efe'e assistant.
Do NOT translate or invent; keep the same Fe'efe'e content.
Only normalize spacing/diacritics if needed. Output Fe'efe'e only.

Draft: {answer}
Final:"""
                    polished = self.generator.generate(rp).strip()
                    # keep safe if GPT returns something weird/empty
                    if polished and len(polished) >= len(answer) - 2:
                        answer = polished
                        mode = "phrasebook+gpt"

                return {
                    "answer": f"{answer} [{i+1}]",
                    "contexts": top_contexts,
                    "prompt": "Direct phrasebook match",
                    "mode": mode,
                }

        # 3) Generator path (GPT or FLAN)
        cited = "\n\n".join(
            [f"[{i+1}] {c['text']} (Source: {c['meta']['source']})" for i, c in enumerate(top_contexts)]
        )
        prompt = f"""
You are a precise assistant.
Use ONLY the context below. Do not invent information.
Return a clear sentence that answers the question AND include citations like [1], [2].
If the answer is not explicit, say: "I could not find this in the documents."

Question: {query}

Context:
{cited}

Answer:"""

        completion = self.generator.generate(prompt).strip()

        # Guard against citation-only output like "[1] [2]"
        import re
        if not completion or re.fullmatch(r"(?:\s*\[\d+\]\s*)+", completion):
            completion = f"{top_contexts[0]['text']} [1]"
            mode = "fallback"
        else:
            mode = "gpt" if self.use_chatgpt else "flan"

        return {
            "answer": completion,
            "contexts": top_contexts,
            "prompt": prompt,
            "mode": mode,
        }