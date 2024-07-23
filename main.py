import os
import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Settings
# from llama_index.indices.list import ListIndex

import tempfile
import os
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader,ListIndex

# Streamlitページ設定
st.set_page_config(
    page_title="",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

import os
from dotenv import load_dotenv

load_dotenv() # .envファイルを読み込む

api_key = os.getenv('API_KEY') # 環境変数からAPIキーを取得する


# Temperature slider
temperature = st.slider(
    "Temperature (Creativity)", min_value=0.0, max_value=2.0, value=0.8, step=0.1
)

if api_key:
    st.session_state.api_key = api_key
    openai.api_key = api_key

import streamlit as st
import openai

# APIキーを保存するためのセッション状態の変数
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# APIキー入力欄
st.title("OpenAI API キー入力")
st.text_input("APIキーを入力してください:", key="api_key", value=st.session_state.api_key)

# 入力されたAPIキーをopenaiに設定
if st.button("APIキーを設定"):
    openai.api_key = st.session_state.api_key
    st.success("APIキーが設定されました")


# PDFファイルが格納されているフォルダ
pdf_folder = r"C:\Users\MI1935\Downloads\■■■2024-03-21_PROJECT_FILE\RAG\data"


# --- ベクトルデータベースの構築 (事前構築) ---
@st.cache_resource(show_spinner=False)
def build_vector_database():
    # データの読み込み中にスピナーを表示
    with st.spinner(text="Wait minites."):

        # ドキュメントの読み込みとインデックス作成
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        for document in docs:
            document.metadata["filename"] = document.source  # メタデータにファイル名を追加
        embed_model = OpenAIEmbedding(openai_api_key=openai.api_key)

        # Create a vector store index from the documents
        index = VectorStoreIndex.from_documents(docs)
        return index

# ベクトルデータベースを構築
index = build_vector_database()

# --- チャットUI ---

# チャット履歴の初期化
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "こんにちは！"}
    ]

# LLMとquery engineの設定 (APIキーが有効な場合のみ)
if openai.api_key:
    llm = OpenAI(
        model="gpt-4o",
        temperature=temperature,
    )

# --- チャットエンジンの初期化 (変更点) ---
if "chat_engine" not in st.session_state.keys() and openai.api_key is not None:
    similarity_top_k = 5  # 類似するものを5つ検索
    postprocessor = SimilarityPostprocessor(similarity_top_k=similarity_top_k)
    system_prompt ="""
    ・あなたはナレッジベースに提供されている書類に関する情報を提供するチャットボットです。
    ・利用者の質問に、正確かつなるべく詳細に、参考資料を引用しながら答えることがあなたの役割です。
    ・情報は800文字以上、4000文字以内に収めるようにしてください。
    ・詳細な情報が必要な場合は、利用者から追加の質問を促します。
    ・マークダウン形式で見やすく出力してください。
    ・情報源を明記して回答するように努めます。
    ・複数の解釈がある場合は、それぞれを提示します。
    ・与えられた情報だけでは判断できない場合には、判断できない旨を伝えます。
    ・判断に不足している情報があれば、追加で情報を求めます。
    ・必要に応じて、関連する情報源へのリンクを提供します。
    ・中立的な立場を保ち、偏った情報提供は行いません。
    """  # システムプロンプトを設定
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        postprocessor=postprocessor,
        llm=llm,  # llm変数も定義されていることを前提としています
        system_prompt=system_prompt  # システムプロンプトを追加
    )


# チャット処理
if openai.api_key  is not None:
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)


                # for source in response.source_nodes:
                #     st.write(f"**参考箇所:**")
                #     st.markdown(source.node.get_content())  # Display node content directly

                for source in response.source_nodes:
                    st.write(f"**参考箇所:** {source.node.metadata['filename']}") # ファイル名をここに表示
                    st.markdown(source.node.get_content())  # Display node content directly
