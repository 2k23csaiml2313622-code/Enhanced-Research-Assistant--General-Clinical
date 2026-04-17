from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 🔥 BETTER MODEL (MATCH RAG)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")   # 🔥 upgraded

model = load_model()

# =========================
# LOAD CLINICAL DATA (INCREASE SIZE)
# =========================

@st.cache_resource
def load_clinical_data():
    df = pd.read_csv("data/mimic_final.csv", nrows=2000)  # 🔥 more coverage
    df = df.dropna(subset=["text", "summary"])
    return df

clinical_df = load_clinical_data()

# =========================
# PRECOMPUTE EMBEDDINGS
# =========================

@st.cache_resource
def compute_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)

clinical_texts = clinical_df["text"].tolist()
clinical_summaries = clinical_df["summary"].tolist()
clinical_embeddings = compute_embeddings(clinical_texts)

# =========================
# 🔥 BETTER GROUND TRUTH (TOP-K AVG)
# =========================

def get_ground_truth(query):

    try:
        query_emb = model.encode([query], convert_to_numpy=True)

        similarities = cosine_similarity(query_emb, clinical_embeddings)[0]

        # 🔥 take top 3 instead of 1
        top_k_idx = similarities.argsort()[-3:]

        combined_text = " ".join([
            clinical_summaries[i] for i in top_k_idx
        ])

        return combined_text

    except:
        return ""

# =========================
# MAIN EVALUATION FUNCTION
# =========================

def evaluate_rag(query, answer, contexts, mode):

    try:
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

        query_emb = model.encode([query], convert_to_numpy=True)
        answer_emb = model.encode([answer], convert_to_numpy=True)
        context_emb = model.encode(contexts, convert_to_numpy=True)

        # =========================
        # METRICS (IMPROVED)
        # =========================

        relevance = cosine_similarity(query_emb, answer_emb)[0][0]

        sim_matrix = cosine_similarity(answer_emb, context_emb)

        faithfulness = np.mean(sim_matrix)

        precision = np.max(sim_matrix)

        recall = np.mean(sim_matrix)   # 🔥 more stable

        # 🔥 improved groundedness (avoid harsh min)
        groundedness = np.percentile(sim_matrix, 30)

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
        # 🔥 CLINICAL ALIGNMENT (FIXED)
        # =========================

        if mode == "Clinical Diagnostic Mode":

            ground_truth = get_ground_truth(query)

            if ground_truth:

                gt_emb = model.encode([ground_truth], convert_to_numpy=True)

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
