import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle

import numpy as np
import pandas as pd 

import unicodedata
import faiss

# transformers and sentence-transformers are heavy; import lazily when needed
try:
    # keep typing names available for static analysis only; real imports happen in Generator
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore

# LangChain imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None
    Document = None

from src.config import (
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

        try:
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

            elif p.suffix.lower() in [".csv", ".xlsx"]:
                if p.suffix.lower() == ".csv":
                    df = pd.read_csv(p)
                else:
                    df = pd.read_excel(p)
                
                for idx, (_, r) in enumerate(df.iterrows(), start=1):
                    # Create a descriptive text from the row
                    if "question" in df.columns and "answer" in df.columns:
                        # FAQ format
                        q = str(r.get("question", "")).strip()
                        a = str(r.get("answer", "")).strip()
                        if q and a:
                            text = f"Q: {q}\nA: {a}"
                        else:
                            continue
                    else:
                        # General row: concatenate all values
                        values = [str(v).strip() for v in r.values if str(v).strip()]
                        text = " ".join(values)
                        if not text:
                            continue
                    
                    corpus.append({
                        "id": f"{p}#{idx}",
                        "text": text,
                        "meta": {"source": f"{p.name} (row {idx})"}
                    })
        except Exception as e:
            # Skip unreadable files
            print(f"Warning: could not load {p}: {e}")
            continue

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
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.docs: List[Dict[str, Any]] = []

    def _get_embeddings(self):
        if self.embeddings is None:
            if HuggingFaceEmbeddings is None:
                raise ImportError("LangChain HuggingFaceEmbeddings not available")
            device = "cpu"
            try:
                import torch
                if getattr(torch, "cuda", None) and torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                device = "cpu"
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True})
        return self.embeddings

    def build(self, docs: List[Dict[str, Any]]):
        if FAISS is None or Document is None:
            raise ImportError("LangChain FAISS and Document not available")
        self.docs = docs
        for i, d in enumerate(docs):
            d["index"] = i
        langchain_docs = [Document(page_content=d["text"], metadata={"index": d["index"], **d["meta"]}) for d in docs]
        embeddings = self._get_embeddings()
        self.vectorstore = FAISS.from_documents(langchain_docs, embeddings)

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        if self.vectorstore is None:
            # Lexical-overlap fallback (no embedding required)
            def normalize(s: str) -> str:
                return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))

            q_norm = normalize(query)
            q_tokens = set(q_norm.split())
            scored = []
            for idx, d in enumerate(self.docs):
                d_norm = normalize(d.get("text", ""))
                d_tokens = set(d_norm.split())
                if not q_tokens or not d_tokens:
                    continue
                # Use OR logic: score is fraction of query tokens present in doc
                matched = sum(1 for q in q_tokens if q in d_tokens)
                score = float(matched) / max(1, len(q_tokens)) * 0.5  # Scale to 0-0.5
                if score > 0:
                    scored.append((idx, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        # Use LangChain FAISS
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            return [(doc.metadata["index"], float(score)) for doc, score in results]
        except Exception as e:
            print(f"Error in semantic search: {e}. Falling back to lexical.")
            # Fallback to lexical
            def normalize(s: str) -> str:
                return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))

            q_norm = normalize(query)
            q_tokens = set(q_norm.split())
            scored = []
            for idx, d in enumerate(self.docs):
                d_norm = normalize(d.get("text", ""))
                d_tokens = set(d_norm.split())
                if not q_tokens or not d_tokens:
                    continue
                matched = sum(1 for q in q_tokens if q in d_tokens)
                score = float(matched) / max(1, len(q_tokens)) * 0.5  # Scale to 0-0.5
                if score > 0:
                    scored.append((idx, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def save(self, vs_dir: Path):
        vs_dir.mkdir(parents=True, exist_ok=True)
        if self.vectorstore:
            self.vectorstore.save_local(str(vs_dir))
        # Also save docs for compatibility
        with open(vs_dir / "docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, vs_dir: Path):
        try:
            embeddings = self._get_embeddings()
            self.vectorstore = FAISS.load_local(str(vs_dir), embeddings, allow_dangerous_deserialization=True)
            # Extract docs from docstore
            self.docs = []
            for doc_id, doc in self.vectorstore.docstore._dict.items():
                meta = doc.metadata.copy()
                index = meta.pop("index", len(self.docs))
                self.docs.append({
                    "id": doc_id,
                    "text": doc.page_content,
                    "meta": meta,
                    "index": index
                })
            self.docs.sort(key=lambda x: x["index"])
        except Exception as e:
            print(f"Failed to load LangChain vectorstore: {e}. Loading docs from pickle.")
            try:
                with open(vs_dir / "docs.pkl", "rb") as f:
                    self.docs = pickle.load(f)
                self.vectorstore = None
            except Exception as e2:
                raise FileNotFoundError(f"Could not load vectorstore or docs: {e2}") from e2

# ----------------- Re-Ranker -----------------

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Do not load reranker model to avoid downloads
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

        return sorted(contexts, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

# ----------------- Generator -----------------

class Generator:
    def __init__(self, model_name: str = GENERATION_MODEL, use_chatgpt: bool = False):
        self.use_chatgpt = use_chatgpt
        # Delay heavy transformer loads until generation is actually needed.
        self.client = None
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        if use_chatgpt:
            try:
                # Import OpenAI lazily so environments that don't include the
                # `openai` package (runtime-minimal) can still import the module.
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = "gpt-4o-mini"
            except Exception:
                # Don't fail import time; disable chatgpt mode and leave
                # generator available for local (transformers) generation.
                print("Warning: 'openai' package not installed; ChatGPT mode disabled.")
                self.client = None
                self.use_chatgpt = False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.use_chatgpt:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        else:
            # Do not load local model to avoid downloads
            raise RuntimeError("Local generation not available")

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
        # Defer instantiation of heavy components until needed
        self.generator = None
        self.use_reranker = use_reranker
        self.reranker = None
        # Only create the simple objects here; heavy models load lazily


    def ensure_loaded(self):
        if not (self.vs_dir / "docs.pkl").exists():
            raise FileNotFoundError("Vector store not found. Run ingest.py first.")
        self.index.load(self.vs_dir)

    def get_generator(self):
        if self.generator is None:
            self.generator = Generator(model_name=GENERATION_MODEL, use_chatgpt=self.use_chatgpt)
        return self.generator

    def get_reranker(self):
        if not self.use_reranker:
            return None
        if self.reranker is None:
            self.reranker = ReRanker()
        return self.reranker

    

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
        else:
            # For non-date queries, also do lexical search to boost substring matches
            lexical_idxs = _lexical_search_docs(docs, query, limit=50)

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
                # Use raw similarity score
                score = max(0.0, float(base_sem_score))

            ctxs.append(docs[i] | {"score": score})

        # Ensure contexts are ordered by descending score by default so the
        # highest-scoring chunk is shown first in the UI. If a reranker is
        # enabled it will re-order these contexts as needed.
        try:
            ctxs.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        except Exception:
            pass

        # --------- Rerank (if enabled) ----------
        reranker = self.get_reranker()
        if reranker and ctxs:
            ctxs = reranker.rerank(query, ctxs)

        # Ensure contexts are ordered by descending score by default so the
        # highest-scoring chunk is shown first in the UI. If a reranker is
        # enabled it will re-order these contexts as needed.
        try:
            ctxs.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        except Exception:
            pass

        # --------- Rerank (if enabled) ----------
        reranker = self.get_reranker()
        if reranker and ctxs:
            ctxs = reranker.rerank(query, ctxs)

        # Final sort by the chosen 'final metric' so the UI always shows the
        # highest-scoring context first. Prefer `rerank_score` if available
        # (set by the reranker), otherwise fall back to the semantic/lexical
        # `score` computed above.
        try:
            ctxs.sort(key=lambda c: (c.get("rerank_score", float("-inf")), c.get("score", 0.0)), reverse=True)
        except Exception:
            pass

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

        # For direct extraction (FAQ or phrasebook cases) prefer a single
        # primary context chosen according to the strongest available
        # signal. If reranker scores exist for the returned contexts,
        # choose the context with the largest absolute `rerank_score`
        # (magnitude indicates strength). Otherwise fall back to the
        # highest semantic `score`.
        #
        # Keep `top_contexts` for generation prompts and display; use the
        # `primary_order` list for direct-match shortcuts below.
        if any(c.get("rerank_score") is not None for c in top_contexts):
            # sort by absolute magnitude of rerank_score (larger magnitude first)
            primary_order = sorted(
                top_contexts,
                key=lambda c: abs(float(c.get("rerank_score", 0.0))),
                reverse=True,
            )
        else:
            primary_order = sorted(top_contexts, key=lambda c: c.get("score", 0.0), reverse=True)

        # ------------------------------
        # FAQ-style shortcut
        # ------------------------------
        for i, c in enumerate(primary_order):
            text = c["text"]
            if text.strip().startswith("Q:") and "A:" in text:
                qna_parts = text.split("A:", 1)
                if len(qna_parts) == 2:
                    q = qna_parts[0].replace("Q:", "").strip()
                    ans = qna_parts[1].strip()
                    full_prompt = f"""
                    [FAQ Direct Match]

                    Question: {query}

                    Matched Source: {c['meta']['source']}
                    Context: {text}

                    Extracted Answer: {ans}
                    """
                    return {
                        "answer": f"Q: {q}\nA: {ans} [{i+1}]",
                        "contexts": top_contexts,
                        "prompt": full_prompt.strip(),
                        "mode": "faq",
                    }

            # ------------------------------
            # Phrasebook style (Fe'efe'e first)
            # ------------------------------
            for j, c in enumerate(primary_order):
                text = c.get("text", "")
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
                        "answer": f"{answer} [{j+1}]",
                        "contexts": top_contexts,
                        "prompt": full_prompt.strip(),
                        "mode": "phrasebook",
                    }

        # ------------------------------
        # GPT branch (if enabled)
        # ------------------------------
        if self.use_chatgpt:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                raise RuntimeError(
                    "OpenAI package required for ChatGPT mode. Install the 'openai' package or disable ChatGPT usage."
                ) from e
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
        # Fallback: return top context without generation
        # ------------------------------
        completion = top_contexts[0]['text']
        prompt = f"Question: {query}\n\nTop context: {completion}"
        mode = "lexical"

        return {
            "answer": completion,
            "contexts": top_contexts,
            "prompt": prompt,
            "mode": mode,
        }

    
       