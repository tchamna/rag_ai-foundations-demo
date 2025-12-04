FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements for better caching
COPY light_requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r light_requirements.txt

# Copy application code and precomputed vectorstore
COPY src/ ./src/
COPY vectorstore/ ./vectorstore/
COPY data/ ./data/

ENV PORT=8000
EXPOSE ${PORT}

CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port", "8000", "--server.headless", "true"]
