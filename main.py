
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import getpass
from dotenv import load_dotenv
import tempfile


from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = st.secrets.get("GEMINI_API_KEY")
if api_key == None:
    st.error("Gemini api key not found. please add it in steamlit secrets")
    st.stop
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF Question Answering Assistant")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files and api_key:
    with st.spinner("Loading and indexing your documents..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = PyMuPDFLoader(tmp_file.name)
                all_docs.extend(loader.load())



model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0,
    max_tokens = None

)
