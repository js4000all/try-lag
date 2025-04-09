import streamlit as st
import requests
import json

st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Demo")

# サイドバー
with st.sidebar:
    st.header("設定")
    llm_url = st.text_input("LLMサーバーURL", "http://localhost:8000")
    max_length = st.slider("最大長", 50, 500, 100)
    temperature = st.slider("温度", 0.0, 1.0, 0.7)

# メインコンテンツ
tab1, tab2 = st.tabs(["質問", "データ投入"])

with tab1:
    st.header("質問")
    question = st.text_area("質問を入力してください", height=100)
    
    if st.button("回答を生成"):
        if question:
            try:
                response = requests.post(
                    f"{llm_url}/generate",
                    json={
                        "prompt": question,
                        "max_length": max_length,
                        "temperature": temperature
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    st.write("回答:")
                    st.write(result["generated_text"])
                else:
                    st.error(f"エラーが発生しました: {response.text}")
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
        else:
            st.warning("質問を入力してください")

with tab2:
    st.header("データ投入")
    uploaded_file = st.file_uploader("テキストファイルをアップロード", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("アップロードされたテキスト", text, height=200) 
        