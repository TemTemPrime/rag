
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.environ.get("GEMINI_API_KEY")
if api_key == None:
    raise Exception("api_key not found")
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF Question Answering Assistant")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0,
    max_token = None

)