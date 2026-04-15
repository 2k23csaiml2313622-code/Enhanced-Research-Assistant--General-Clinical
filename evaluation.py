from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# LOAD MODEL (ONCE)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# LOAD CLINICAL DATA (LIMITED)
# =========================

@st.cache_resource
def load_clinical_data():
    df = pd.read_csv("data/mimic_final.csv", nrows=500)
    df = df.dropna(subset=["text", "summary"])
    return df

clinical_df = load_clinical_data()

# =========================
# PRECOMPUTE EMBEDDINGS
# =========================

@st.cache_resource
def compute_embeddings(texts):
    return model.encode(texts)

clinical_texts = clinical_df["text"].tolist()
clinical_summaries = clinical_df["summary"].tolist()
clinical_embeddings = compute_embeddings(clinical_texts)

# =========================
# GET GROUND TRUTH
# =========================

def get_ground_truth(query):
    try:
        query_emb = model.encode([query])
        similarities = cosine_similarity(query_emb, clinical_embeddings)[0]
        idx = similarities.argmax()
        return clinical_summaries[idx]
    except:
        return ""

# =========================
# MAIN EVALUATION FUNCTION
# =========================

def evaluate_rag(query, answer, contexts, mode):

    try:
        # Clean contexts
        contexts = [c.strip() for c in contexts if c.strip()]

        if len(contexts) == 0:
            return {
                "Relevance Score": 0,
                "Faithfulness Score": 0,
                "Precision Score": 0,
                "Recall Score": 0,
                "Groundedness Score": 0
            }

        # =========================
        # EMBEDDINGS
        # =========================

        query_emb = model.encode([query])
        answer_emb = model.encode([answer])
        context_emb = model.encode(contexts)

        # =========================
        # METRICS
        # =========================

        relevance = cosine_similarity(query_emb, answer_emb)[0][0]

        faithfulness = np.mean(
            cosine_similarity(answer_emb, context_emb)
        )

        precision = np.max(
            cosine_similarity(answer_emb, context_emb)
        )

        recall = np.mean(
            cosine_similarity(context_emb, answer_emb)
        )

        # 🔥 FIXED GROUNDEDNESS (more stable)
        groundedness = np.percentile(
            cosine_similarity(answer_emb, context_emb), 25
        )

        # =========================
        # NORMALIZATION
        # =========================

        def normalize(x):
            return (x + 1) / 2

        results = {
            "Relevance Score": round(float(normalize(relevance)), 3),
            "Faithfulness Score": round(float(normalize(faithfulness)), 3),
            "Precision Score": round(float(normalize(precision)), 3),
            "Recall Score": round(float(normalize(recall)), 3),
            "Groundedness Score": round(float(normalize(groundedness)), 3),
        }

        # =========================
        # 🔥 CLINICAL ALIGNMENT (ONLY IN CLINICAL MODE)
        # =========================

        if mode == "Clinical Diagnostic Mode":

            ground_truth = get_ground_truth(query)

            if ground_truth:
                gt_emb = model.encode([ground_truth])
                clinical_score = cosine_similarity(answer_emb, gt_emb)[0][0]
            else:
                clinical_score = 0

            results["Clinical Alignment Score"] = round(
                float(normalize(clinical_score)), 3
            )

        return results

    except Exception as e:
        return {
            "Relevance Score": 0,
            "Faithfulness Score": 0,
            "Precision Score": 0,
            "Recall Score": 0,
            "Groundedness Score": 0,
            "Error": str(e)
        }