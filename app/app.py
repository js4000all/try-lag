import streamlit as st
import google.generativeai as genai
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
    api_key = st.text_input("Gemini API Key", type="password")
    max_length = st.slider("最大長", 50, 500, 100)
    temperature = st.slider("温度", 0.0, 1.0, 0.7)

# メインコンテンツ
tab1, tab2 = st.tabs(["質問", "データ投入"])

with tab1:
    st.header("質問")
    question = st.text_area("質問を入力してください", height=100)
    
    if st.button("回答を生成"):
        if question and api_key:
            try:
                # Geminiの設定
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                
                # プロンプトの生成
                prompt = f"""
                以下の質問に回答してください。
                質問: {question}
                """
                
                # テキスト生成
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_length,
                        temperature=temperature
                    )
                )
                
                st.write("回答:")
                st.write(response.text)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
        else:
            if not api_key:
                st.warning("API Keyを入力してください")
            if not question:
                st.warning("質問を入力してください")

with tab2:
    st.header("データ投入")
    uploaded_file = st.file_uploader("テキストファイルをアップロード", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("アップロードされたテキスト", text, height=200) 
        