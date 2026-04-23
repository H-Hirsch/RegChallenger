"""
embeddings.py
Sentence embeddings and hybrid similarity scoring for case retrieval.

Hybrid similarity combines:
  1. Cosine similarity on text embeddings (semantic content of rule)
  2. Categorical boosting (agency match, doctrine era match, administration match)

Embedding text includes: FR title + abstract + agency + CFR references + Claude reasoning
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd

CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "embeddings_cache.json"
MODEL_NAME = "all-mpnet-base-v2"

# Categorical similarity weights
AGENCY_MATCH_BOOST   = 0.08  # same agency → boost similarity score
DOCTRINE_MATCH_BOOST = 0.05  # same doctrine era → boost
ADMIN_MATCH_BOOST    = 0.05  # same administration → boost


_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(MODEL_NAME)
    return _EMBEDDER


def embed_query_rule(rule: dict) -> np.ndarray:
    """Embed a single rule dict (e.g., from FR API search) for similarity scoring."""
    text = build_rule_text(pd.Series(rule))
    model = get_embedder()
    return model.encode([text], show_progress_bar=False)[0]


def find_similar_to_query(
    query_embedding: np.ndarray,
    query_row: dict,
    rulemakings_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar historical cases to an external query rule
    (one not in the dataset). Same hybrid scoring as find_similar_cases.
    """
    q_series = pd.Series(query_row)
    similarities = []
    for i in range(len(rulemakings_df)):
        text_score = cosine_similarity(query_embedding, embeddings[i])
        cat_boost  = categorical_boost(q_series, rulemakings_df.iloc[i])
        similarities.append((i, min(1.0, text_score + cat_boost)))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def build_rule_text(row: pd.Series) -> str:
    """
    Build text representation of a rulemaking case for embedding.
    Includes abstract, explanation, topics, and CFR topics if available.
    """
    parts = []

    title = str(row.get('fr_title', '') or '')
    if title and title != 'nan':
        parts.append(title)

    abstract = str(row.get('fr_abstract', '') or '')
    if abstract and abstract != 'nan':
        parts.append(abstract[:500])

    explanation = str(row.get('fr_explanation', '') or '')
    if explanation and explanation != 'nan':
        parts.append(explanation[:300])

    agency = str(row.get('fr_agency_name', '') or '')
    if agency and agency != 'nan':
        parts.append(f"Agency: {agency}")

    cfr = str(row.get('cfr_references', '') or '')
    if cfr and cfr != 'nan':
        parts.append(f"CFR: {cfr}")

    cfr_topics = str(row.get('fr_cfr_topics', '') or '')
    if cfr_topics and cfr_topics != 'nan':
        parts.append(f"CFR Topics: {cfr_topics}")

    topics = str(row.get('fr_topics', '') or '')
    if topics and topics != 'nan':
        parts.append(f"Topics: {topics}")

    case_name = str(row.get('cl_case_name', '') or '')
    if case_name and case_name != 'nan':
        parts.append(f"Case: {case_name}")

    reasoning = str(row.get('claude_reasoning', '') or '')
    if reasoning and reasoning != 'nan':
        parts.append(reasoning[:400])

    return " | ".join(parts)


def compute_embeddings(rulemakings_df: pd.DataFrame) -> np.ndarray:
    """Compute or load cached embeddings."""
    ids = sorted(rulemakings_df.index.astype(str).tolist())
    has_abstract = 'fr_abstract' in rulemakings_df.columns
    cache_key = "_".join(ids) + f"_abstract={has_abstract}_model={MODEL_NAME}"

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            if cache.get('key') == cache_key:
                return np.array(cache['embeddings'])
        except Exception:
            pass

    model = get_embedder()
    texts = [build_rule_text(row) for _, row in rulemakings_df.iterrows()]
    embeddings = model.encode(texts, show_progress_bar=False)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump({'key': cache_key, 'embeddings': embeddings.tolist()}, f)

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def categorical_boost(query_row: pd.Series, candidate_row: pd.Series) -> float:
    """Compute categorical similarity boost based on shared attributes."""
    boost = 0.0

    # Agency match — same regulatory domain. Joint rulemakings have multiple
    # agencies joined by "; " in fr_agency_name; match if ANY agency overlaps.
    def _agency_set(val):
        return {a.strip() for a in str(val or '').split(';') if a.strip()}
    q_agencies = _agency_set(query_row.get('fr_agency_name', ''))
    c_agencies = _agency_set(candidate_row.get('fr_agency_name', ''))
    if q_agencies and c_agencies and (q_agencies & c_agencies):
        boost += AGENCY_MATCH_BOOST

    # Doctrine era match — same legal framework
    q_era = str(query_row.get('doctrine_era', '') or '').strip()
    c_era = str(candidate_row.get('doctrine_era', '') or '').strip()
    if q_era and c_era and q_era == c_era:
        boost += DOCTRINE_MATCH_BOOST

    # Administration match — use our derived field (correctly distinguishes Trump45/Trump47)
    # fr_president from FR API doesn't distinguish between terms
    q_admin = str(query_row.get('administration_rule', '') or
                  query_row.get('administration_case', '') or '').strip()
    c_admin = str(candidate_row.get('administration_rule', '') or
                  candidate_row.get('administration_case', '') or '').strip()
    if q_admin and c_admin and q_admin == c_admin:
        boost += ADMIN_MATCH_BOOST

    return boost


def find_similar_cases(
    query_idx: int,
    rulemakings_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar rulemaking cases using hybrid scoring.
    Score = cosine_similarity + categorical_boost (agency, doctrine era, administration).
    Dataset is deduplicated by opinion_file at load time.
    """
    query_embedding = embeddings[query_idx]
    query_row = rulemakings_df.iloc[query_idx]

    similarities = []
    for i in range(len(rulemakings_df)):
        if i == query_idx:
            continue
        text_score  = cosine_similarity(query_embedding, embeddings[i])
        cat_boost   = categorical_boost(query_row, rulemakings_df.iloc[i])
        final_score = min(1.0, text_score + cat_boost)
        similarities.append((i, final_score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
