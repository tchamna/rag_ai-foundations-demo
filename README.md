# AI-Powered Banking Assistant (RAG Demo)
**Goal:** Demonstrate hands-on ability to build an LLM/RAG experience similar to Capital One’s AI Foundations use-cases:  
- Customer-facing Q&A over banking FAQs, disclosures, and *uploaded statements*  
- Retrieval-Augmented Generation (RAG) with **FAISS** + **SentenceTransformer embeddings**  
- Local, no vendor keys required (uses **Transformers: `google/flan-t5-base`** for answer synthesis)  
- Deployed as a **Streamlit app**

> This repo shows practical skills with LLMs, NLP, vector databases, and app delivery.

## Quickstart
### 0) Create and activate a Python environment
```bash
python -m venv .venv
source .venv/bin/activate             # macOS/Linux
# or
.venv\Scripts\activate              # Windows
```

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

> CPU-only works. If you have a GPU and PyTorch with CUDA, generation will be faster.

### 2) Build (or rebuild) the vector store
```bash
python src/ingest.py --data_dir data/docs --vs_dir vectorstore


```

### 3) Run the Streamlit app
```bash
streamlit run src/app_streamlit.py
```
Note: if that doesnt work, use 
```
python -m streamlit run src/app_streamlit.py

```
Open the local URL that Streamlit prints (usually http://localhost:8501).

## What’s Inside
- **RAG pipeline** with FAISS + `all-MiniLM-L6-v2` embeddings (SentenceTransformers)
- **Answer synthesis** with `google/flan-t5-base` (no API keys needed)
- **Upload statements** and ask questions like “What subscription charges did I have in May?”
- **Grounding citations**: we show which chunks were used to answer

## Repo Structure
```
.
├── README.md
├── requirements.txt
├── data/
│   └── docs/
│       ├── faqs_bank.csv
│       ├── product_disclosures.txt
│       ├── privacy_notice.txt
│       └── sample_statements/
│           └── sample_statement_001.txt
├── vectorstore/                 # created by ingest.py
└── src/
    ├── app_streamlit.py
    ├── ingest.py
    ├── rag_pipeline.py
    └── config.py
```

## Example Prompts
- *What are overdraft fees and how can I avoid them?*
- *Summarize monthly recurring charges in the uploaded statement.*
- *Do you store my data? (privacy policy)*
- *What is the process to dispute a transaction?*

## Notes
- This demo runs **entirely local** using open models; no external keys required.
- For production: replace the local generator with your preferred LLM (e.g., Bedrock, OpenAI, Azure OpenAI) and move artifacts to S3/ECR/ECS/SageMaker.
- Add auth, telemetry, prompt logging, and guardrails before real users.
