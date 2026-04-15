import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from utils import web_search, scrape_website, read_pdf
from rag_pipeline import create_vector_store, retrieve_context
from report_generator import generate_report
from evaluation import evaluate_rag

# =========================
# ENV + LLM SETUP
# =========================

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Student Deep Research Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Deep Research Assistant")
st.markdown("⚡ RAG System using **Web + Clinical (MIMIC-IV) Knowledge**")

query = st.text_input("Enter your research topic")
pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"])

mode = st.radio(
    "Select Mode",
    ["General Research", "Clinical Diagnostic Mode"]
)

run_eval = st.checkbox("Enable Performance Evaluation")

# =========================
# MAIN
# =========================

if st.button("Start Research"):

    progress = st.progress(0)
    text_data = ""

    # =========================
    # WEB SEARCH
    # =========================

    st.write("🔎 Searching web sources...")
    urls = web_search(query)[:2]

    progress.progress(20)

    # =========================
    # SCRAPING
    # =========================

    for url in urls:
        try:
            content = scrape_website(url)
            if content:
                text_data += content[:1500]
        except:
            pass

    progress.progress(40)

    # =========================
    # PDF
    # =========================

    if pdf:
        st.write("📄 Reading PDF...")
        pdf_text = read_pdf(pdf)
        text_data += pdf_text[:2000]

    progress.progress(60)

    # =========================
    # LIMIT TEXT
    # =========================

    text_data = text_data[:6000]

    st.write(f"📊 Processed text size: {len(text_data)} characters")

    # =========================
    # SAFETY CHECK
    # =========================

    if not text_data.strip():
        st.error("❌ No data found. Try another query.")
        st.stop()

    # =========================
    # VECTOR DB
    # =========================

    st.write("🧠 Creating web knowledge base...")
    vector_db = create_vector_store(text_data)

    progress.progress(80)

    # =========================
    # 🔥 MODE-BASED RETRIEVAL
    # =========================

    context = retrieve_context(vector_db, query, mode)

    # =========================
    # LLM
    # =========================

    st.write("🤖 Generating answer...")

    if mode == "Clinical Diagnostic Mode":
        prompt = f"""
You are a clinical AI assistant.

STRICTLY prioritize clinical evidence.
If any conflict occurs, trust clinical data.

Context:
{context}

Query:
{query}
"""
    else:
        prompt = f"""
You are a research assistant.

Answer clearly and concisely using only relevant information.

Context:
{context}

Query:
{query}
"""

    answer = llm.invoke(prompt).content

    progress.progress(100)

    # =========================
    # OUTPUT
    # =========================

    st.subheader("📚 Research Answer")
    st.write(answer)

    # =========================
    # REPORT
    # =========================

    report = generate_report(llm, context, query)

    st.subheader("📑 Structured Research Report")
    st.write(report)

    # =========================
    # DOWNLOAD
    # =========================

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="research_report.txt",
        mime="text/plain"
    )

    # =========================
    # 🔥 EVALUATION (MODE AWARE)
    # =========================

    if run_eval:
        st.subheader("📊 Performance Metrics")

        try:
            contexts_list = [c for c in context.split("\n") if c.strip()]

            metrics = evaluate_rag(query, answer, contexts_list, mode)

            # 🔥 Show metrics
            for key, value in metrics.items():

                # ❌ Hide clinical score in general mode
                if mode == "General Research" and "Clinical" in key:
                    continue

                st.metric(label=key, value=value)

        except Exception as e:
            st.warning(f"⚠️ Evaluation failed: {e}")

    # =========================
    # SOURCES
    # =========================

    st.subheader("🔗 Sources")

    for u in urls:
        st.write(u)