# RAG Demo

This repository contains two retrieval augmented chat surfaces:

- **NJ Sharing Chat** at `/njsharing`, backed by documents under `data/`.
- **Generic RAG Studio** at `/` (also `/generic`), backed by uploads under `data_generic/`.

Both apps let users drag documents into the sidebar, rebuild the index, refresh question ideas, and ask natural-language questions over the indexed content with citations.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

Open http://127.0.0.1:8000/generic for the generic upload-and-query version. Its APIs live under `/api/generic/*`, and uploaded files are stored under `data_generic/uploads/` so they do not mix with the NJ Sharing documents.

Open http://127.0.0.1:8000/generic/compare to compare two LLM answers over the generic uploaded-document index.

Use http://127.0.0.1:8000/njsharing/compare to compare OpenAI, Claude, and Ollama answers side by side. The compare API is `/api/compare` and runs provider calls in parallel against the same retrieved context.

Use http://127.0.0.1:8000/njsharing/freemodels to compare three local free Ollama models. The API is `/api/freemodels`. Defaults are `llama3:8b`, `mistral:latest`, and `gouranshitera/bloomz-1b7`; override with `RAG_FREE_MODELS`.

Compare results include elapsed seconds, input/output token counts, and configured price per million tokens. Defaults are the cheapest wired model for each provider: OpenAI `gpt-5-nano` at `$0.05/$0.40` per million input/output tokens, Groq `llama-3.3-70b-versatile` at `$0/$0` by default, Claude `claude-haiku-4-5-20251001` at `$1/$5`, and local Ollama `llama3.2:1b` at `$0/$0`. Override models with `RAG_OPENAI_MODEL`, `RAG_GROQ_MODEL`, `RAG_CLAUDE_MODEL`, and `RAG_OLLAMA_MODEL`; override prices with `OPENAI_INPUT_PER_MILLION`, `OPENAI_OUTPUT_PER_MILLION`, `GROQ_INPUT_PER_MILLION`, `GROQ_OUTPUT_PER_MILLION`, `CLAUDE_INPUT_PER_MILLION`, `CLAUDE_OUTPUT_PER_MILLION`, `OLLAMA_INPUT_PER_MILLION`, and `OLLAMA_OUTPUT_PER_MILLION`.

Answers are synthesized after retrieval with this provider order:

1. OpenAI
2. Groq
3. Claude
4. Ollama
5. Local extractive fallback

The app loads environment variables from `rag_demo/.env` when present. If it is absent, it also checks related workspace `.env` files, including `african-languages-mcp-servers/.env`.

Useful overrides:

```powershell
$env:RAG_ENV_PATH="C:\path\to\.env"
$env:RAG_PROVIDER_ORDER="openai,groq,claude,ollama"
$env:RAG_OPENAI_MODEL="gpt-5-nano"
$env:RAG_CLAUDE_MODEL="claude-haiku-4-5-20251001"
$env:GROQ_API_KEY="your-groq-key"
$env:RAG_GROQ_MODEL="llama-3.3-70b-versatile"
$env:OLLAMA_URL="http://127.0.0.1:11434"
$env:RAG_OLLAMA_MODEL="llama3.2:1b"
```

If all synthesis providers fail, the app falls back to local extractive answers with citations.

## Supported Files

PDF, TXT, Markdown, CSV, JSON, JSONL, DOCX, ODT, RTF, XLSX, HTML.

The app automatically rebuilds its cached indexes when files change. Use the refresh button in either UI to force a rebuild.


## Generic Studio workspace behavior

The modern `/generic` interface gives each browser a separate workspace using a random,
HttpOnly, SameSite cookie. Files and answer caches are stored under
`storage/workspaces/<random-token>/`. Existing demo documents under `data_generic/`
are not exposed to visitors. The generic comparison page uses the same workspace.
Clearing cookies loses access to that workspace; these are browser sessions, not accounts.

Uploads accept up to 10 files per request, 20 MB per file, and 100 MB stored per
workspace. Invalid documents and files with no extractable text are rejected individually.
Scanned PDFs need OCR before uploading. Office archive expansion is limited to 100 MB.
Document removal rebuilds the index and removes the workspace answer cache.
The UI clears the visible conversation after a document is removed.

### Before public deployment

This remains a single-process local application. Use one Uvicorn worker: workspace
operations are serialized within that process. A public service still needs account
authentication, durable workspace ownership, distributed locking and background parsing,
request-body limits at the reverse proxy (multipart parsing precedes file validation),
rate limiting, parser sandboxing, retention and cleanup policies, monitoring, and backups.
Serve through HTTPS and set `RAG_SECURE_COOKIES=1`. Same-origin writes are enforced;
configure the reverse proxy to preserve the public host and scheme.
Do not expose the legacy `/api/*` NJ Sharing demo endpoints in a public deployment.
They remain a shared local demo. Only `/api/generic/*` uses browser workspaces.
Relevant document excerpts are sent to configured model providers to generate answers.
Retrieval uses local sentence embeddings and FAISS semantic search. Scanned-document OCR
and full conversational follow-up resolution are not implemented.


## Battery Education

Open `/battery` from the Battery Education card in Generic Studio. This read-only
section uses a separate curated library in `data_battery/` and never searches visitor
uploads. `/api/battery/status`, `/questions`, and `/chat` provide its APIs. Guides cover
battery-health basics, consumer lithium-ion care, warning signs, and lead-acid differences,
with Apple, U.S. Fire Administration, and Trojan reference links (reviewed 2026-10-23).
It uses the configured synthesis providers and falls back to reference extracts.
Questions mentioning smoke, fire or hissing receive immediate local safety guidance.
This is a starter reference library, not exhaustive coverage or a remote diagnostic tool.
Extend and review the guides to expand coverage; model-specific settings require the
manufacturer's documentation. There is no live web lookup inside the education chat.


## Semantic retrieval: embeddings + FAISS

All document indexes now use `sentence-transformers/all-MiniLM-L6-v2` on CPU, matching
the original Foundations demo's embedding model. It creates 384-dimensional normalized
vectors; FAISS `IndexFlatIP` searches by cosine similarity. MMR reduces repeated evidence.
There is no silent TF-IDF fallback. Embedding failures are reported as processing failures.
The LLM receives the retrieved passages and generates cited answers as before.

The first run downloads the public embedding model. Later runs use its local cache.
Document and question embedding stays local; answer generation still uses configured
model providers. Change `RAG_EMBEDDING_MODEL` before starting the server to choose another
SentenceTransformer model. Existing keyword indexes or indexes using another model are
rebuilt automatically. Cached vectors for unchanged passage text are reused on rebuild.
Each workspace persists its FAISS index, vectors, and citation metadata together in an
atomically replaced versioned cache; deleting a document removes its vectors on rebuild.
The status API exposes engine, model, dimensions, vector count, and similarity metric.

To run regression and real-embedding semantic retrieval tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```


### Question intent interpretation

Generic Studio asks the configured LLM to interpret typos, informal phrasing and mixed-language
questions before FAISS retrieval. It rewrites the question wrapper while preserving an exact
translation-target substring from the original input. That substring is also searched directly
across document passages, so exact dictionary examples can outrank broad semantic definitions.
The UI displays an “Understood as” note when a question is rewritten. Ambiguous interpretations
request clarification; if providers are unavailable, the original query is retained. Interpretations
are cached per browser workspace. This adds a model call for each new question; it is best-effort
intent recognition, not a guarantee that every typo or ambiguous request can be understood.


### Dictionary meaning coverage

For dictionary and translation questions, the answer lists the distinct senses of the relevant
entry, preserving numbered senses, grammar labels, figurative uses and source examples.
Exact-expression retrieval scans all indexed passages and includes adjacent context to recover
split entries, instead of limiting evidence to the first top-k match. Evidence is bounded to
60,000 characters; the response exposes `meaning_coverage` and the UI warns when limited.
This does not guarantee exhaustive recovery of OCR errors, synonyms or meanings absent from
the extracted text. Longer source examples retain their full gloss and are labeled contextual.


### Unicode and PDF fidelity

PDF extraction uses PyMuPDF to retain embedded-font character mappings, followed by NFC
normalization without removing accents or tone marks. Index schema 3 rebuilds earlier text
extractions automatically. Answer and citation text use locally hosted Charis fonts from SIL
(version 7.000); the bundled `static/fonts/OFL.txt` contains the redistribution license.
Numeric character references are decoded as text before HTML escaping. The frontend does
not reinterpret document text as markup. Model instructions require copying source spellings;
they cannot guarantee perfect transcription, so citations remain available for verification.

PyMuPDF is distributed under AGPL/commercial licensing; review the upstream license when
choosing terms for a hosted distribution.


Generic chat includes up to ten prior user/assistant turns when interpreting a follow-up.
Clarification replies resolve against the previous request, rather than becoming new dictionary
lookups. This history lives in the current page session and clears on New conversation.
History participates in both interpretation and answer cache keys. The interpreter does not
invent a translation target from the language in which the question was asked.


For cross-language queries, intent interpretation also produces up to six search equivalents
in the document's reference/gloss language. FAISS searches the original question and these
variants, and exact-phrase lookup includes the variants across the corpus. The original
translation target remains intact. Generated search variants are not evidence; final answers
must cite actual retrieved document text. This improves English questions over French glosses
without claiming the English MiniLM encoder itself is a multilingual embedding model.


### Low-cost model comparison

`/generic/compare` defaults to GPT-5 nano versus GPT-6 Luna, with an optional GPT-6.1 Sol
comparison. It keeps the same retrieved evidence and prompt for both answer models; this
isolates answer quality and does not benchmark different retrieval or intent models. Main chat
continues to use its configured model. Calls use standard service tier, no tools, no automatic
retry, and a maximum 1,200 completion tokens (including reasoning). Shared intent normalization
still uses the normal configured provider. The preflight estimate conservatively counts UTF-8
bytes plus framing headroom as input tokens, reserves $0.002 for interpretation, and blocks
comparisons above an estimated $0.05 total request budget. This is a per-request estimate,
not an account-wide spending limit. Rate constants reflect official standard pricing checked
2026-10-23 and should be kept current. Results show token-based estimated costs and latency;
failed model access is displayed explicitly. Identical requests use the workspace response cache.


## Enforced guardrails

- Paid calls share a SQLite allowance across all visitors and routes: `$0.50` per UTC day
  by default (`RAG_DAILY_BUDGET_USD`), with a `$0.05` per-call estimate ceiling
  (`RAG_CALL_BUDGET_USD`). Conservative reservations are atomic and persist through restart.
  Reservations remain charged against the allowance after failures because provider billing
  can be uncertain. This is a token-cost allowance, not an exact provider billing ledger;
  keep the approved model price table in `app/rag/spend.py` current.
- Every model call has an output cap of 1,600 tokens or less. Comparison calls keep their
  tighter 1,200-token cap and shared `$0.05` preflight. Unknown paid models, unapproved model
  destinations and tool/function calls are rejected before HTTP calls. Raw provider errors
  are not sent to the browser. Text-only document chat cannot execute tools or browse links.
- API requests enforce same-origin and Fetch Metadata checks, 90 requests/minute/client IP
  (`RAG_REQUESTS_PER_MINUTE`), at most four active jobs globally, and one active workspace job.
  Job protection spans streamed response completion. Trusted hosts default to localhost,
  127.0.0.1 and the local test host. Set `RAG_ALLOWED_HOSTS` to deployment hostnames.
- Bodies are spooled and bounded before parsing, including chunked transfers: 40 MB/upload
  batch and 64 KB/chat request. Files retain the 20 MB/file and 100 MB/workspace limits;
  additional limits are 100 documents/workspace and 2 GB total workspace document bytes.
  Office archive expansion, reserved Windows names, PDF page/text limits (2,000 pages,
  6 million characters) and spreadsheet cell limits (500,000) are checked.
- CSP blocks external scripts, connections, framing and embedded objects. Source and answer
  text is escaped before rendering. Browser cookies are HttpOnly/SameSite; enable secure
  cookies behind HTTPS. Private storage is excluded from Git and is not served statically.

The app still uses browser capability cookies rather than authenticated user accounts.
The local request/job limits are single-process; use one worker. Before public hosting,
add authenticated ownership, a reverse proxy with trusted forwarding and TLS, isolated parser
workers, malware scanning appropriate to the files accepted, and retention/deletion policy.
These controls reduce risk; prompt instructions alone cannot guarantee that an LLM will never
follow malicious text or make a factual error. Always retain source verification.


Comparisons now automatically fit retrieved evidence into the existing $0.05 preflight
estimate: lower-priority passages are removed first, then an exceptionally long remaining
passage may be shortened at a word boundary. All selected models receive exactly the same
resulting prompt and retain the same output cap. The UI discloses reduced context and the
citations contain only text actually sent to the models. This budgeted comparison can omit
meanings from the full document; the ordinary document chat remains separate. A question
that cannot fit even with minimal evidence is still rejected rather than raising the budget.
