import os
import typing as ty
import uuid

import chromadb
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

from validation import validate


DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PERSIST_DIR = "/tmp/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="wide"
)

# セッション状態の初期化
if 'model' not in st.session_state:
    st.session_state.model = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY
if 'embedding' not in st.session_state:
    st.session_state.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 利用可能なモデル
AVAILABLE_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash"
]

st.title("📚 RAG Demo")

# サイドバー
with st.sidebar:
    st.header("設定")
    st.write(f"セッションID: {st.session_state.session_id}")

    api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
    st.write("https://aistudio.google.com/app/apikey")
    
    selected_model = st.selectbox(
        "モデルを選択",
        options=list(AVAILABLE_MODELS),
        format_func=lambda x: x
    )
    
    if st.button("モデルを準備"):
        if api_key:
            try:
                genai.configure(api_key=api_key)
                st.session_state.model = genai.GenerativeModel(selected_model)
                st.success("モデルの準備が完了しました")
            except Exception as e:
                st.error(f"モデルの準備に失敗しました: {str(e)}")
        else:
            st.warning("API Keyを入力してください")
    
    if st.session_state.model is not None:
        st.write(f"モデル: {st.session_state.model.model_name}")
    else:
        st.write("モデルを選択してください")

    max_length = st.slider("最大長", 64, 512, 256)
    temperature = st.slider("温度", 0.0, 1.0, 0.7)

def _get_call_model_function() -> ty.Optional[ty.Callable[[str], str]]:
    if st.session_state.model is None:
        return None
    def _call_model(prompt: str) -> str:
        return st.session_state.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_length,
            temperature=temperature
            )
        ).text
    return _call_model

# メインコンテンツ
tab1, tab2, tab3 = st.tabs(["質問", "データ投入", "test"])

with tab1:
    st.header("質問")
    question = st.text_area("質問を入力してください", height=100)
    
    if st.button("回答を生成"):
        if question and st.session_state.model:
            try:
                call_model = _get_call_model_function()
                # プロンプトの検証
                is_valid, error_message = validate(question, call_model)
                if not is_valid:
                    st.error(f"プロンプトが不適切です: {error_message}")
                    st.stop()
                
                context_text = ""
                if 'vector_db' in st.session_state:
                    docs = st.session_state.vector_db.similarity_search(question, k=3)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                else:
                    st.warning("検索対象データがありません。データ投入を先に行ってください。")

                prompt = f"""
以下の文書に基づいて、質問に答えてください。
--- 文書情報 ---
{context_text}
--- 質問 ---
{question}
"""

                response = call_model(prompt)

                # 回答の検証
                is_valid, error_message = validate(response, call_model)
                if not is_valid:
                    st.error(f"回答に不適切な内容が含まれています: {error_message}")
                    st.stop()

                st.write("回答:")
                st.write(response)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
        else:
            if not st.session_state.model:
                st.warning("モデルを初期化してください")
            if not question:
                st.warning("質問を入力してください")

with tab2:
    st.header("データ投入")
    uploaded_file = st.file_uploader("テキストファイルをアップロード", type=["txt"])
    
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("アップロードされたテキスト", text, height=200, disabled=True)

        call_model = _get_call_model_function()
        # チャンク分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_text(text)
        st.write(f"チャンク数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            st.text_area(f"チャンク {i}", chunk, height=100, disabled=True)
            is_valid, error_message = validate(chunk, call_model)
            if not is_valid:
                st.error(f"チャンクに不適切な内容が含まれています: {error_message}")
                st.stop()

        db = Chroma(
            client=chroma_client,
            collection_name=st.session_state.session_id,
            embedding_function=st.session_state.embedding
        )
        db.add_texts(
            texts=chunks,
            metadatas=[{"source": uploaded_file.name} for _ in range(len(chunks))],
        )
        st.success("データベースに文書を登録しました！")
        # 保存されたDBをセッションに保持
        st.session_state.vector_db = db

with tab3:
    st.header("test")
    target = st.text_area("target", height=100)
    if st.button("validate"):
        is_valid, error_message = validate(target, _get_call_model_function())
        st.write(is_valid)
        st.write(error_message)
