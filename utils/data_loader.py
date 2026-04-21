"""
data_loader.py
Loads and merges full pipeline outputs into a unified dataset for the dashboard.

Joins step1 → step2 → step3 on cl_cluster_id.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache
from datetime import datetime

ADMINISTRATIONS = [
    ("2001-01-20", "2009-01-20", "Bush (43rd)"),
    ("2009-01-20", "2017-01-20", "Obama"),
    ("2017-01-20", "2021-01-20", "Trump (1st term)"),
    ("2021-01-20", "2025-01-20", "Biden"),
    ("2025-01-20", "2099-01-01", "Trump (2nd term)"),
]

OUTCOME_LABELS = {
    1: "Rule Upheld (Affirmed)",
    2: "Rule Struck Down (Reversed/Vacated)",
    3: "Mixed (Affirmed in Part)",
    5: "Dismissed",
    6: "Remanded",
    7: "Other",
}

CIRCUIT_NAMES = {
    0: "D.C. Circuit", 1: "1st Circuit",  2: "2nd Circuit",
    3: "3rd Circuit",  4: "4th Circuit",  5: "5th Circuit",
    6: "6th Circuit",  7: "7th Circuit",  8: "8th Circuit",
    9: "9th Circuit",  10: "10th Circuit", 11: "11th Circuit",
}


def _parse_date(date_str):
    if not date_str or pd.isna(date_str):
        return None
    for fmt in ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return None


def get_administration(date_str):
    dt = _parse_date(date_str)
    if not dt:
        return "Unknown"
    d = dt.strftime('%Y-%m-%d')
    for start, end, name in ADMINISTRATIONS:
        if start <= d < end:
            return name
    return "Unknown"


def get_doctrine_era(date_str):
    dt = _parse_date(date_str)
    if not dt:
        return "Unknown"
    d = dt.strftime('%Y-%m-%d')
    if d < "1984-06-25":
        return "Pre-Chevron (pre-1984)"
    elif d < "2019-06-26":
        return "Chevron Era (1984–2024)"
    elif d < "2022-06-30":
        return "Post-Kisor (2019–2022)"
    elif d < "2024-06-28":
        return "Post-WV v. EPA (2022–2024)"
    else:
        return "Post-Loper Bright (2024–)"


def get_outcome_label(outcome):
    try:
        return OUTCOME_LABELS.get(int(float(outcome)), f"Code {outcome}")
    except Exception:
        return str(outcome)


def get_outcome_category(outcome):
    try:
        o = int(float(outcome))
        if o == 1:
            return "win"
        elif o in [2, 3]:
            return "loss"
        else:
            return "other"
    except Exception:
        return "other"


def _clean_cluster_id(val):
    """Normalize cluster_id to integer string (strip .0 suffix)."""
    try:
        return str(int(float(val)))
    except Exception:
        return str(val).strip()


@lru_cache(maxsize=1)
def load_dataset(
    step1_path: str,
    step2_path: str,
    step3_path: str,
) -> tuple:
    """
    Load and merge full pipeline outputs.
    Join key: cl_cluster_id across all three steps.

    Returns:
        (all_cases_df, rulemakings_df)
    """
    # ── Step 1 ─────────────────────────────────────────────────────────────────
    s1 = pd.read_csv(step1_path, dtype=str)
    s1 = s1[s1['cl_cluster_id'].notna() & (s1['cl_cluster_id'] != '')].copy()
    s1['cl_cluster_id'] = s1['cl_cluster_id'].apply(_clean_cluster_id)
    s1['outcome'] = pd.to_numeric(s1['outcome'], errors='coerce')
    s1['circuit'] = pd.to_numeric(s1['circuit'], errors='coerce')

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    if step2_path.endswith('.xlsx'):
        s2 = pd.read_excel(step2_path, dtype=str)
    else:
        s2 = pd.read_csv(step2_path, dtype=str)
    s2['cl_cluster_id'] = s2['cl_cluster_id'].apply(_clean_cluster_id)

    for col in ['claude_case_type', 'claude_reasoning', 'claude_challenged_fr']:
        if col not in s2.columns:
            s2[col] = None

    # Deduplicate step2 by cluster_id before merging
    s2_dedup = s2.drop_duplicates(subset=['cl_cluster_id'], keep='first')

    # ── Step 3 ─────────────────────────────────────────────────────────────────
    s3 = pd.read_csv(step3_path, dtype=str)
    s3['cl_cluster_id'] = s3['cl_cluster_id'].apply(_clean_cluster_id)

    # ── Merge step1 + step2 ────────────────────────────────────────────────────
    merged = s1.merge(
        s2_dedup[['cl_cluster_id', 'claude_case_type', 'claude_challenged_fr',
                  'claude_reasoning']],
        on='cl_cluster_id', how='left'
    )

    merged['outcome_label']       = merged['outcome'].apply(get_outcome_label)
    merged['outcome_category']    = merged['outcome'].apply(get_outcome_category)
    merged['circuit_name']        = merged['circuit'].apply(
        lambda x: CIRCUIT_NAMES.get(int(x), f"Circuit {x}") if pd.notna(x) else "Unknown"
    )
    merged['administration_case'] = merged['date_filed'].apply(get_administration)
    merged['doctrine_era']        = merged['date_filed'].apply(get_doctrine_era)

    # ── Rulemakings subset ─────────────────────────────────────────────────────
    rulemakings = merged[merged['claude_case_type'] == 'RULEMAKING'].copy()

    # Build step3 column list — include enriched fields if present
    s3_base_cols = [
        'cl_cluster_id', 'fr_document_number', 'fr_citation_official',
        'fr_title', 'fr_publication_date', 'fr_agency_name',
        'fr_parent_department', 'fr_sub_agency', 'rin_numbers',
        'cfr_references', 'fr_html_url', 'fr_significant',
        'fr_lookup_status', 'fr_type', 'fr_action',
    ]
    s3_enriched_cols = [
        'fr_abstract', 'fr_explanation', 'fr_cfr_topics', 'fr_topics',
        'fr_page_length', 'fr_president', 'fr_effective_on',
    ]
    s3_cols = s3_base_cols + [c for c in s3_enriched_cols if c in s3.columns]

    rulemakings = rulemakings.merge(
        s3[[c for c in s3_cols if c in s3.columns]],
        on='cl_cluster_id', how='inner'
    )

    rulemakings = rulemakings[
        rulemakings['fr_lookup_status'].str.startswith('FOUND', na=False)
    ].copy()

    rulemakings = rulemakings.drop_duplicates(
        subset=['cl_cluster_id'], keep='first'
    ).copy()
    rulemakings = rulemakings.reset_index(drop=True)

    rulemakings['administration_rule'] = rulemakings['fr_publication_date'].apply(
        get_administration
    )
    rulemakings['doctrine_era_rule'] = rulemakings['fr_publication_date'].apply(
        get_doctrine_era
    )

    return merged, rulemakings


def load_opinion_text(opinions_dir: str, cluster_id: str) -> str:
    """
    Load opinion text for a cluster from the opinions directory.
    Full pipeline stores opinions as cluster_{id}.txt
    Validation dataset uses cluster_{id}_CAXX.txt format.
    """
    if not cluster_id or pd.isna(cluster_id):
        return ""
    clean_id = _clean_cluster_id(cluster_id)
    opinions_path = Path(opinions_dir)

    # Full pipeline format
    path = opinions_path / f"cluster_{clean_id}.txt"
    if not path.exists():
        # Validation format fallback
        matches = list(opinions_path.glob(f"cluster_{clean_id}_*.txt"))
        if matches:
            path = matches[0]
        else:
            return ""

    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        if '=' * 20 in text:
            text = text.split('=' * 20)[-1]
        import re
        text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
        text = re.sub(r'^(\s*)\. ', r'\1', text, flags=re.MULTILINE)
        return text.strip()
    except Exception:
        return ""


def lookup_previously_challenged(
    rulemakings_df: pd.DataFrame,
    fr_document_number: str
) -> pd.DataFrame:
    """
    Check if a specific rule has been previously challenged in our dataset.
    Returns matching rows if found, empty DataFrame otherwise.
    Used by the FR search feature to flag rules with known challenges.
    """
    if not fr_document_number or pd.isna(fr_document_number):
        return pd.DataFrame()
    clean = str(fr_document_number).strip()
    return rulemakings_df[
        rulemakings_df['fr_document_number'].fillna('') == clean
    ]
