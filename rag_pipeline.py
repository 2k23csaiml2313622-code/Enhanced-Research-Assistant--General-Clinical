from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

# 🔥 RERANKING IMPORTS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# 🔥 LOAD RERANKER MODEL
# =========================

@st.cache_resource
def load_reranker():
    return SentenceTransformer("all-MiniLM-L6-v2")

reranker = load_reranker()

# =========================
# LOAD PRECOMPUTED CLINICAL DB
# =========================

@st.cache_resource
def get_clinical_db():
    return Chroma(
        persist_directory="chroma_db_1",
        embedding_function=embedding_model
    )

# =========================
# WEB VECTOR DB
# =========================

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )

    docs = splitter.split_text(text[:4000])

    return Chroma.from_texts(docs, embedding=embedding_model)

# =========================
# 🔥 RERANK FUNCTION
# =========================

def rerank_documents(query, docs, top_k=6):

    if not docs:
        return docs

    try:
        query_emb = reranker.encode([query])
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = reranker.encode(doc_texts)

        scores = cosine_similarity(query_emb, doc_embs)[0]

        ranked_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked_docs[:top_k]]

    except Exception as e:
        print("Reranking error:", e)
        return docs[:top_k]

# =========================
# 🔥 RETRIEVAL (ABLATION ENABLED)
# =========================

def retrieve_context(vector_db, query, mode):

    # 🔥 CHANGE THIS VALUE FOR EXPERIMENTS
    retrieval_mode = "hybrid"
    # options: "hybrid", "clinical", "web"

    # =========================
    # 🔎 RETRIEVAL MODES
    # =========================

    if retrieval_mode == "clinical":

        clinical_db = get_clinical_db()

        combined_docs = clinical_db.similarity_search(
            query,
            k=10
        )

    elif retrieval_mode == "web":

        combined_docs = vector_db.as_retriever(
            search_kwargs={"k": 10}
        ).invoke(query)

    else:  # HYBRID

        # web retrieval
        web_docs = vector_db.as_retriever(
            search_kwargs={"k": 8}
        ).invoke(query)

        # clinical retrieval (only in clinical mode UI)
        if mode == "Clinical Diagnostic Mode":

            clinical_db = get_clinical_db()

            clinical_docs = clinical_db.similarity_search(
                query,
                k=12
            )

            combined_docs = clinical_docs[:8] + web_docs[:3]

        else:
            combined_docs = web_docs

    # =========================
    # 🔥 SMART FILTERING
    # =========================

    filtered_docs = []

    for doc in combined_docs:
        text = doc.page_content.strip()

        if (
            len(text) > 120 and
            any(word in text.lower() for word in query.lower().split())
        ):
            filtered_docs.append(doc)

    # fallback
    if len(filtered_docs) < 4:
        filtered_docs = combined_docs

    # =========================
    # 🔥 RERANKING
    # =========================

    final_docs = rerank_documents(query, filtered_docs, top_k=6)

    # =========================
    # FINAL CONTEXT
    # =========================

    context = "\n".join([
        doc.page_content for doc in final_docs
    ])

    return context
