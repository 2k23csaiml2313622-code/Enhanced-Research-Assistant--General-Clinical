from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# LOAD PRECOMPUTED CLINICAL DB
# =========================

@st.cache_resource
def get_clinical_db():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

# =========================
# WEB VECTOR DB
# =========================

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_text(text[:3000])

    return Chroma.from_texts(docs, embedding=embedding_model)

# =========================
# RETRIEVAL (MODE-BASED 🔥)
# =========================

def retrieve_context(vector_db, query, mode):

    # 🔎 Web retrieval
    web_docs = vector_db.as_retriever(
        search_kwargs={"k": 5}
    ).invoke(query)

    # 🧠 Clinical retrieval (ONLY if needed)
    if mode == "Clinical Diagnostic Mode":

        clinical_db = get_clinical_db()

        clinical_docs = clinical_db.as_retriever(
            search_kwargs={"k": 3}
        ).invoke(query)

        # 🔥 Clinical priority
        combined_docs = clinical_docs + web_docs[:1]

    else:
        # 🚀 General mode = NO clinical noise
        combined_docs = web_docs

    # =========================
    # FINAL CONTEXT
    # =========================

    context = "\n".join([
        doc.page_content for doc in combined_docs
    ])

    return context