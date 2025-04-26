FROM python:3.12-slim

RUN set -x \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir \
        chromadb \
        langchain \
        langchain-community \
        sentence-transformers \
        google-generativeai \
        python-dotenv \
        streamlit \
        streamlit-javascript \
        sentencepiece

WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
