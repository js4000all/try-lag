# try-lag

RAG (Retrieval-Augmented Generation) のデモアプリケーション

## 構成

```
.
├── llm/                # LLMサーバー（Colab上で実行）
│   ├── main.py        # FastAPIサーバー
│   └── requirements.txt
├── app/               # ローカルUI
│   ├── app.py         # Streamlitアプリケーション
│   └── requirements.txt
├── Dockerfile         # ローカルUIサーバイメージ作成用
└── compose.yaml       # ローカルUIサーバ起動用
```

## 使い方

### 1. LLMサーバーの起動（Colab上）

```bash
cd llm
pip install -r requirements.txt
python main.py
```

### 2. ローカルUIの起動（ローカル）

```bash
docker-compose up --build
```

ブラウザで `http://localhost:8501` にアクセス

## 機能

- テキスト生成
  - プロンプト入力
  - 最大長と温度の調整
  - LLMサーバーURLの設定
- データ投入
  - テキストファイルのアップロード
