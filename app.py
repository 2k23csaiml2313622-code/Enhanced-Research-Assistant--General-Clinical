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

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ API key not found. Check your .env file.")
    st.stop()

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.2
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
st.markdown("⚡ RAG System using **Web + Clinical Knowledge**")

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

    if not query.strip():
        st.warning("⚠️ Please enter a query.")
        st.stop()

    progress = st.progress(0)
    text_data = ""

    # =========================
    # WEB SEARCH
    # =========================

    st.write("🔎 Searching web sources...")
    urls = web_search(query)[:3]

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
    # 🔥 FIX: FALLBACK (CRITICAL)
    # =========================

    if not text_data.strip():

        st.warning("⚠️ Web data not found. Using fallback knowledge...")

        text_data = f"""
        Topic: {query}

        Provide a clear explanation, key concepts, causes, and important insights.
        """

    # =========================
    # VECTOR DB
    # =========================

    st.write("🧠 Creating web knowledge base...")
    vector_db = create_vector_store(text_data)

    progress.progress(80)

    # =========================
    # RETRIEVAL
    # =========================

    context = retrieve_context(vector_db, query, mode)

    # =========================
    # LLM GENERATION
    # =========================

    st.write("🤖 Generating answer...")

    if mode == "Clinical Diagnostic Mode":
        prompt = f"""
You are a clinical expert AI.

STRICT RULES:
- Use ONLY clinical evidence from context
- Do NOT use general knowledge
- Be medically precise
- If insufficient data → say "Insufficient clinical evidence"

Context:
{context}

Query:
{query}
"""
    else:
        prompt = f"""
You are a research assistant.

STRICT RULES:
- Use ONLY the provided context
- Keep answer clear, structured, and factual
- Avoid unnecessary details

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
    # EVALUATION
    # =========================

    if run_eval:
        st.subheader("📊 Performance Metrics")

        try:
            contexts_list = [c for c in context.split("\n") if c.strip()]

            metrics = evaluate_rag(query, answer, contexts_list, mode)

            for key, value in metrics.items():

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
