FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/opt/models
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY app/ ./app/
COPY static/ ./static/
COPY data_battery/ ./data_battery/
RUN mkdir -p data data_generic storage && useradd --uid 10001 --create-home studio && chown -R studio:studio /app /opt/models
USER studio
ENV PORT=8080 RAG_ALLOWED_HOSTS=rag.tchamna.com,localhost,127.0.0.1 RAG_SECURE_COOKIES=1 RAG_PROVIDER_ORDER=openai RAG_DAILY_BUDGET_USD=0.50
EXPOSE 8080
HEALTHCHECK --interval=30s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=10)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers", "--forwarded-allow-ips", "*"]
