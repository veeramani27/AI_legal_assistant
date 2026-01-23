#
# 2. CONFIGURATION & INITIALIZATION
#

import os

# Environment & Models
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
# Langfuse (Optional, keeping as per original code)
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()

#  LLM & Embeddings
llm = ChatOpenAI(
    model="gpt-oss-20b",
    temperature=0.7,
    openai_api_base="https://Fyra.im/v1",
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Vector Store (Main Knowledge Base)
db = Chroma(
    persist_directory="./chroma",
    embedding_function=embeddings,
    collection_name="legal",
)
print(f"Chroma DB initialized with {db._collection.count()} documents.")

#  Callbacks
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)
langfuse_handler = CallbackHandler()

# Configuration
config = {
    "callbacks": [langfuse_handler],
    "configurable": {"thread_id": "test_thread_01"},
}
