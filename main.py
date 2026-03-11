from key import key
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

api_key = os.environ.get("GEMINI_API_KEY")
if api_key == None:
    raise Exception("api_key not found")

client = genai.Client(api_key=key)
model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0,
    max_token = None

)