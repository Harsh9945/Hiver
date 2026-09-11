# src/retrieve.py
import os
import re
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retrieval_index.pkl")

class Retriever:
    """
    FR-2 Retriever: Semantic and lexical retrieval over 15,000+ historical AppleSupport resolved threads.
    Combines BM25 and TF-IDF vector space similarity to retrieve top-k cases and compute max similarity.
    Cold retrieval is flagged when max similarity < 0.25.
    """
    def __init__(self, index_path: str = INDEX_PATH, k1: float = 1.5, b: float = 0.75):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Retrieval index not found at {index_path}. Run scripts/generate_datasets.py first.")
            
        with open(index_path, "rb") as f:
            bundle = pickle.load(f)
            
        self.case_ids = bundle["case_ids"]
        self.corpus_texts = bundle["corpus_texts"]
        self.agent_replies = bundle["agent_replies"]
        self.doc_lengths = bundle["doc_lengths"]
        self.avgdl = bundle["avgdl"]
        self.idf = bundle["idf"]
        self.tfidf_vectorizer = bundle["tfidf_vectorizer"]
        self.tfidf_matrix = bundle["tfidf_matrix"]
        self.metadata = bundle.get("metadata", {})
        
        self.k1 = k1
        self.b = b
        self.cold_threshold = 0.25

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top_k similar past cases for a customer query.
        Returns retrieved cases, max similarity score, and cold retrieval flag.
        """
        clean_q = str(query).strip().lower()
        if not clean_q:
            return {
                "top_k_cases": [],
                "max_similarity": 0.0,
                "is_cold_retrieval": True
            }

        # Vector space cosine similarity
        q_vec = self.tfidf_vectorizer.transform([clean_q])
        cos_scores = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # BM25 Lexical scoring
        tokens = re.findall(r"\w+", clean_q)
        bm25_scores = np.zeros(len(self.corpus_texts))
        
        for term in tokens:
            if term in self.idf:
                term_idf = self.idf[term]
                # Fast vector approximation using TF-IDF term indices
                if term in self.tfidf_vectorizer.vocabulary_:
                    term_idx = self.tfidf_vectorizer.vocabulary_[term]
                    term_col = self.tfidf_matrix.getcol(term_idx).toarray().flatten()
                    tf = term_col > 0  # binary or scaled indicator
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / self.avgdl))
                    bm25_scores += term_idf * ((tf * (self.k1 + 1.0)) / (denom + 1e-9))

        # Composite hybrid score: 0.6 * cosine_sim + 0.4 * normalized_bm25
        max_bm25 = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1.0
        normalized_bm25 = bm25_scores / max_bm25
        hybrid_scores = 0.65 * cos_scores + 0.35 * normalized_bm25

        # Top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        cases = []
        for idx in top_indices:
            cases.append({
                "case_id": self.case_ids[idx],
                "customer_text": self.corpus_texts[idx],
                "agent_reply": self.agent_replies[idx],
                "similarity": round(float(hybrid_scores[idx]), 4),
                "cosine_sim": round(float(cos_scores[idx]), 4)
            })

        max_sim = float(hybrid_scores[top_indices[0]]) if len(top_indices) > 0 else 0.0
        is_cold = max_sim < self.cold_threshold

        return {
            "top_k_cases": cases,
            "max_similarity": round(max_sim, 4),
            "is_cold_retrieval": is_cold
        }
