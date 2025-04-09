# try-lag

RAG (Retrieval-Augmented Generation) のデモアプリケーション

## 構成

```
.
├── app/               # UIアプリケーション
│   ├── app.py         # Streamlitアプリケーション
│   └── requirements.txt
├── Dockerfile         # UIアプリケーション用
└── compose.yaml       # Docker Compose設定
```

## 使い方

### UIアプリケーションの起動（ローカル）

```bash
docker-compose up --build
```

ブラウザで `http://localhost:8501` にアクセス

## 機能

- テキスト生成
  - Gemini APIを使用
  - API Keyの設定
  - プロンプト入力
  - 最大長と温度の調整
- データ投入
  - TODO
