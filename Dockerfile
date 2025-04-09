FROM python:alpine

RUN set -x \
    && apk update \
    && apk upgrade \
    && apk add --no-cache --virtual .build-deps \
        g++ \
        cmake \
        build-base \
        apache-arrow-dev \
        python3-dev \
        linux-headers \
        rust \
        cargo
RUN pip install --no-cache-dir streamlit
RUN pip install --no-cache-dir google-generativeai
RUN pip install --no-cache-dir python-dotenv
RUN pip install --no-cache-dir langchain==0.0.350 chromadb==0.4.18 sentence-transformers==2.2.2
RUN apk del .build-deps

WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
