
import os
import streamlit as st
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import getpass
from dotenv import load_dotenv
import tempfile
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key == None:
    st.error("Gemini api key not found. please add it in steamlit secrets")
    st.stop()
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
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
        split_docs = splitter.split_documents(all_docs)

        model = ChatGoogleGenerativeAI(
               model = 'gemini-2.5-flash',
               temperature = 0,
               max_tokens = None,
               google_api_key=api_key
        )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=api_key)
        embedding_dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
)
        vector_store.add_documents(split_docs)

        retriever = vector_store.as_retriever(search_kwargs={"k":2})

    st.success("Documents ready. Ask your question below.")

query = st.text_input("Ask a question about your documents")

if query:
    with st.spinner("Generating answer..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = model.invoke(prompt)

        st.markdown("### Answer")
        st.write(response.content)

        st.markdown("### Source Snippets")
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content[:400]
            st.write(f"Source {i}: {snippet}...")
      
            

            