"""
app.py
Regulatory Litigation Risk Dashboard

Predicts the likelihood and outcome of legal challenges to federal rules
using RAG-based similarity search over historical cases and Claude AI.

Usage:
    streamlit run app.py

Environment variables:
    ANTHROPIC_API_KEY  — Anthropic API key for Claude predictions

Data files (place in dashboard/data/):
    step1_idb_to_cl_validation_full.csv
    step2_validation_with_claude.xlsx  (or .csv)
    step3_fr_lookup.csv
    validation_opinions/               (directory of opinion TXT files)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import numpy as np
import pandas as pd
import streamlit as st

# Secrets fallback for Streamlit Cloud: if .env wasn't found (production deploy),
# read ANTHROPIC_API_KEY from st.secrets and expose it via os.environ so the
# prediction module (which reads os.environ at import) picks it up.
if not os.environ.get('ANTHROPIC_API_KEY'):
    try:
        key = st.secrets.get('ANTHROPIC_API_KEY')
        if key:
            os.environ['ANTHROPIC_API_KEY'] = key
    except Exception:
        pass

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import (
    load_dataset, load_opinion_text, lookup_previously_challenged,
    get_administration, get_doctrine_era,
)
from utils.embeddings import (
    compute_embeddings, find_similar_cases,
    embed_query_rule, find_similar_to_query,
)
from utils.prediction import predict_outcome, explain_historical_outcome
from utils.fr_api import search_documents, get_agencies, normalize_fr_result, lookup_by_citation
from datetime import date
import random

EXAMPLE_KEYWORDS = [
    "power plant emissions", "menthol cigarettes", "student loan forgiveness",
    "independent contractor", "climate disclosure", "methane emissions",
    "vehicle fuel economy", "drug pricing", "title IX", "net neutrality",
    "overtime pay", "waters of the United States", "noncompete agreements",
    "firearms stabilizing brace", "PFAS", "pipeline safety",
    "medicare advantage", "endangered species", "nuclear waste",
    "ESG investing",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RegChallenger",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

    /* Pull the main content up so the title aligns with the sidebar header */
    .main .block-container,
    [data-testid="stMain"] .block-container { padding-top: 1.5rem !important; }
    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    .main-header {
        background: linear-gradient(135deg, #033C5A 0%, #033C5A 100%);
        padding: 2rem 2.5rem; border-radius: 8px; margin-bottom: 2rem;
        border-left: 5px solid #D6BF91;
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700;
        color: #F6F1E8;
        font-size: 2.6rem;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
        letter-spacing: 0.01em;
    }
    .main-header h1 .suffix {
        font-weight: 400;
        color: #D6BF91;
        font-size: 0.7em;
        margin-left: 0.6rem;
    }
    .main-header p {
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 300;
        color: #D6BF91;
        font-size: 1rem;
        margin: 0;
        letter-spacing: 0.02em;
    }

    .metric-card {
        background: #f8f6f1; border: 1px solid #e8e2d5; border-radius: 6px;
        padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-card .value { font-family: 'Playfair Display', serif; font-size: 2rem;
        font-weight: 700; color: #033C5A; }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.05em; margin-top: 0.2rem; }

    .prediction-box { border-radius: 8px; padding: 1.5rem 2rem; margin: 1rem 0; }
    .prediction-win { background: #f0fdf4; border: 2px solid #16a34a; }
    .prediction-loss { background: #fef2f2; border: 2px solid #dc2626; }
    .prediction-mixed { background: #fffbeb; border: 2px solid #d97706; }
    .prediction-uncertain { background: #f1f5f9; border: 2px solid #64748b; }

    .outcome-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-win { background: #dcfce7; color: #166534; }
    .badge-loss { background: #fee2e2; color: #991b1b; }
    .badge-mixed { background: #fef3c7; color: #92400e; }
    .badge-other { background: #f1f5f9; color: #475569; }

    .case-card { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1.2rem 1.5rem; margin: 0.75rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }

    .similarity-bar { height: 6px; background: #e5e7eb; border-radius: 3px; margin: 0.5rem 0; }
    .similarity-fill { height: 100%; background: linear-gradient(90deg, #CCE3F4, #0075C8);
        border-radius: 3px; }

    .doctrine-tag { display: inline-block; background: #033C5A; color: #D6BF91;
        font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 3px;
        text-transform: uppercase; letter-spacing: 0.05em; }

    .section-header { font-family: 'Playfair Display', serif; font-size: 1.3rem;
        color: #033C5A; border-bottom: 2px solid #D6BF91;
        padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }

    .fr-link { color: #D6BF91; text-decoration: none; font-weight: 600; }

    .sidebar-info { background: #f8f6f1; border-left: 3px solid #D6BF91;
        padding: 0.8rem 1rem; border-radius: 0 4px 4px 0; font-size: 0.85rem;
        color: #4b5563; margin: 0.5rem 0; }

    .sidebar-section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        color: #033C5A;
        border-bottom: 2px solid #D6BF91;
        padding-bottom: 0.4rem;
        margin: 0.5rem 0 0.8rem 0;
    }

    /* Make sidebar blend with the editorial style */
    section[data-testid="stSidebar"] {
        background: #faf8f3;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 460px !important;
        min-width: 460px !important;
        max-width: 460px !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        font-family: 'Source Sans 3', sans-serif;
        color: #033C5A;
    }

    /* Keep the sidebar collapse button visible without requiring hover */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] button[aria-label*="ollapse"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Match the native collapse button's hover background to the custom bottom button */
    [data-testid="stSidebarCollapseButton"] button:hover,
    section[data-testid="stSidebar"] button[kind="header"]:hover,
    section[data-testid="stSidebar"] button[aria-label*="ollapse"]:hover {
        background-color: #ede6d4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
# Full pipeline: opinions stored as cluster_{id}.txt in opinions/ subfolder
# Validation: opinions stored as cluster_{id}_CAXX.txt in validation_opinions/
OPINIONS_DIR = DATA_DIR / "opinions"
if not OPINIONS_DIR.exists():
    OPINIONS_DIR = DATA_DIR / "validation_opinions"  # fallback for dev
STEP1_PATH = str(DATA_DIR / "step1_output.csv")
STEP2_PATH = str(DATA_DIR / "step2_output.csv")
STEP3_PATH = str(DATA_DIR / "step3_output.csv")


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_dataset(STEP1_PATH, STEP2_PATH, STEP3_PATH)


@st.cache_data(show_spinner="Computing embeddings...")
def get_embeddings(_rulemakings_df):
    return compute_embeddings(_rulemakings_df)


@st.cache_data(show_spinner="Loading agency list...", ttl=86400)
def get_agency_list():
    return get_agencies()


@st.cache_data(show_spinner="Searching Federal Register...", ttl=600)
def cached_fr_search(term, agency_slugs, date_gte, date_lte, cfr_title,
                     cfr_part, significant_only, doc_types, per_page, page):
    return search_documents(
        term=term, agency_slugs=list(agency_slugs) if agency_slugs else None,
        date_gte=date_gte, date_lte=date_lte,
        cfr_title=cfr_title, cfr_part=cfr_part,
        significant_only=significant_only,
        doc_types=list(doc_types) if doc_types else None,
        per_page=per_page, page=page,
    )


def _run_search(criteria, page):
    """Fetch search results and store in session_state."""
    fr_citation = criteria.get("fr_citation", "").strip()
    if fr_citation:
        result = lookup_by_citation(fr_citation)
    else:
        result = cached_fr_search(
            term=criteria["term"],
            agency_slugs=criteria["agency_slugs"],
            date_gte=criteria["date_gte"],
            date_lte=criteria["date_lte"],
            cfr_title=criteria["cfr_title"],
            cfr_part=criteria["cfr_part"],
            significant_only=criteria["significant_only"],
            doc_types=criteria["doc_types"],
            per_page=criteria["per_page"],
            page=page,
        )
    if result["error"]:
        st.error(f"FR API error: {result['error']}")
        return False
    st.session_state["search_results"] = [
        normalize_fr_result(d) for d in result["results"]
    ]
    st.session_state["search_count"] = result["count"]
    st.session_state["search_total_pages"] = result["total_pages"]
    st.session_state["search_page"] = page
    st.session_state["search_criteria"] = criteria
    return True


def enrich_query_rule(rule: dict) -> dict:
    """Add derived administration/doctrine fields to an FR search result."""
    pub = rule.get("fr_publication_date", "")
    rule["administration_rule"] = get_administration(pub)
    rule["doctrine_era_rule"]   = get_doctrine_era(pub)
    # For boost matching, treat the query's "doctrine_era" as today's
    # (a hypothetical challenge filed today would be governed by current doctrine)
    rule["doctrine_era"]        = get_doctrine_era(date.today().strftime("%Y-%m-%d"))
    rule["administration_case"] = get_administration(date.today().strftime("%Y-%m-%d"))
    return rule


# ── Helpers ────────────────────────────────────────────────────────────────────
def outcome_badge(cat, label):
    cls = {'win': 'badge-win', 'loss': 'badge-loss', 'mixed': 'badge-mixed'}.get(cat, 'badge-other')
    return f'<span class="outcome-badge {cls}">{label}</span>'


def vulnerability_box_class(vuln):
    if not vuln:
        return "prediction-uncertain"
    v = vuln.lower()
    if "high" in v:
        return "prediction-loss"       # red
    elif "moderate" in v or "medium" in v:
        return "prediction-mixed"      # orange
    elif "low" in v:
        return "prediction-win"        # green
    return "prediction-uncertain"


def cl_slugged_url(cl_url, case_name):
    """Append a CourtListener case-name slug if the stored URL is slugless.
    CL no longer reliably redirects /opinion/<id>/ without a slug."""
    import re as _re
    s = str(cl_url or '').strip()
    if not s or s == 'nan':
        return ''
    m = _re.match(r'^(https?://(?:www\.)?courtlistener\.com/opinion/\d+)/?$', s)
    if not m:
        return s
    name = str(case_name or '').strip()
    slug = _re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'case'
    slug = _re.sub(r'-+', '-', slug)[:80].rstrip('-')
    return f"{m.group(1)}/{slug}/"


def vulnerability_from_outcome(outcome_category):
    """Map historical outcome to vulnerability indicator for similar cases."""
    if outcome_category == 'loss':
        return ('High', 'badge-loss')
    elif outcome_category == 'mixed':
        return ('Moderate', 'badge-mixed')
    elif outcome_category == 'win':
        return ('Low', 'badge-win')
    return ('Unknown', 'badge-other')


def historical_outcome_label(outcome_category):
    """Factual outcome label for decided cases (not a prediction)."""
    return {
        'loss':  'Struck Down',
        'mixed': 'Mixed Outcome',
        'win':   'Upheld',
    }.get(outcome_category, 'Outcome Unknown')


# ── Display helpers ────────────────────────────────────────────────────────────
def _h(value, default='—'):
    """Stringify and HTML-escape a field value (handles None/NaN)."""
    import html as _html
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    s = str(value).strip()
    return _html.escape(s) if s else default


def render_rule_card(rule: dict):
    """Render the FR rule details card."""
    fr_url = str(rule.get('fr_html_url', '') or '')
    abstract = (rule.get('fr_abstract') or '').strip()
    import html as _html
    abstract_safe = _html.escape(abstract[:600]) + ('…' if len(abstract) > 600 else '')
    abstract_html = (
        f"<div style='margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #f0ece2;"
        f"font-size:0.85rem;color:#4b5563;line-height:1.5;'>{abstract_safe}</div>"
    ) if abstract else ""
    sig_badge = (
        '<span class="doctrine-tag" style="background:#7a1f1f;">Significant</span>'
        if str(rule.get('fr_significant', '')) == '1' else ''
    )
    link_html = (
        f'<br><a href="{_html.escape(fr_url, quote=True)}" target="_blank" '
        f'style="color:#033C5A;text-decoration:underline;font-weight:600;">'
        f'→ View rule on FederalRegister.gov</a>'
    ) if fr_url else ""
    html = (
        f'<div class="case-card">'
        f'<div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;color:#033C5A;margin-bottom:0.8rem;">'
        f'{_h(rule.get("fr_title"), "Unknown")}</div>'
        f'<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.8rem;">'
        f'<span class="doctrine-tag">{_h(rule.get("administration_rule"), "")}</span>'
        f'<span class="doctrine-tag">{_h(rule.get("doctrine_era_rule"), "")}</span>'
        f'{sig_badge}</div>'
        f'<table style="width:100%;font-size:0.85rem;border-collapse:collapse;">'
        f'<tr><td style="color:#6b7280;width:130px;padding:3px 0;vertical-align:top;">Agency</td>'
        f'<td style="color:#033C5A;font-weight:600;">'
        f'{"<br>".join(_h(a) for a in str(rule.get("fr_agency_name") or "").split(";") if a.strip()) or "—"}'
        f'</td></tr>'
        f'<tr><td style="color:#6b7280;padding:3px 0;">Published</td>'
        f'<td style="color:#033C5A;">{_h(rule.get("fr_publication_date"))}</td></tr>'
        f'<tr><td style="color:#6b7280;padding:3px 0;">Citation</td>'
        f'<td style="color:#033C5A;">{_h(rule.get("fr_citation_official"))}</td></tr>'
        f'<tr><td style="color:#6b7280;padding:3px 0;">CFR</td>'
        f'<td style="color:#033C5A;">{_h(rule.get("cfr_references"))}</td></tr>'
        f'<tr><td style="color:#6b7280;padding:3px 0;">Type / Action</td>'
        f'<td style="color:#033C5A;">{_h(rule.get("fr_type"))} · {_h(rule.get("fr_action"))}</td></tr>'
        f'</table>'
        f'{abstract_html}{link_html}'
        f'</div>'
    )
    st.html(html)


def render_court_case_card(case_row):
    """Render court case card for a previously-challenged rule (a row from rulemakings_df)."""
    cl_url = cl_slugged_url(case_row.get('cl_opinion_url', ''),
                             case_row.get('cl_case_name', ''))
    vuln = vulnerability_from_outcome(case_row.get('outcome_category', 'other'))
    outcome_label = historical_outcome_label(case_row.get('outcome_category', 'other'))
    badge = outcome_badge(vuln[1].replace('badge-', ''), outcome_label)
    st.markdown(f"""
    <div class="case-card">
        <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">Court Case (in our dataset)</div>
        <div style="font-size:1rem;color:#033C5A;font-weight:600;margin-bottom:0.5rem;">{case_row.get('cl_case_name','Unknown')}</div>
        <div style="margin-bottom:0.5rem;">{badge}</div>
        <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
            <tr><td style="color:#6b7280;width:130px;padding:3px 0;">Outcome</td>
                <td style="color:#033C5A;font-weight:600;">{case_row.get('outcome_label','—')}</td></tr>
            <tr><td style="color:#6b7280;padding:3px 0;">Circuit</td>
                <td style="color:#033C5A;">{case_row.get('circuit_name','—')}</td></tr>
            <tr><td style="color:#6b7280;padding:3px 0;">Filed</td>
                <td style="color:#033C5A;">{case_row.get('date_filed','—')}</td></tr>
            <tr><td style="color:#6b7280;padding:3px 0;">Doctrine Era</td>
                <td style="color:#033C5A;">{case_row.get('doctrine_era','—')}</td></tr>
        </table>
        {"<br><a href='" + str(cl_url) + "' target='_blank' class='fr-link'>→ CourtListener.com</a>" if cl_url and str(cl_url) != 'nan' else ""}
    </div>
    """, unsafe_allow_html=True)


def render_prediction_box(prediction: dict):
    vuln = prediction.get('vulnerability') or 'Unknown'
    box_cls = vulnerability_box_class(vuln)
    v_lower = vuln.lower()
    if 'high' in v_lower:
        vuln_label = 'Highly Vulnerable'
    elif 'moderate' in v_lower or 'medium' in v_lower:
        vuln_label = 'Moderately Vulnerable'
    elif 'low' in v_lower:
        vuln_label = 'Minimally Vulnerable'
    else:
        vuln_label = 'Assessment Unavailable'

    st.markdown(f"""
    <div class="prediction-box {box_cls}">
        <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.05em;margin-bottom:0.4rem;">Rule Vulnerability Assessment</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:#033C5A;margin-bottom:0.6rem;">{vuln_label}</div>
        <div style="font-size:0.85rem;color:#4b5563;">
            <strong>Confidence:</strong> {prediction.get('confidence','—')}<br>
            <strong>Most Likely Circuit:</strong> {prediction.get('most_likely_circuit','—')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    vulns = prediction.get('key_vulnerabilities', [])
    if vulns:
        st.markdown("**Key Legal Vulnerabilities:**")
        for v in vulns:
            st.markdown(f"• {v}")
    favoring = prediction.get('favoring_factors', [])
    if favoring:
        st.markdown("**Factors Favoring the Rule:**")
        for f in favoring:
            st.markdown(f"• {f}")
    reasoning = prediction.get('reasoning', '')
    if reasoning:
        st.markdown("**Reasoning:**")
        st.markdown(reasoning)


def render_similar_cases(similar_cases_data):
    """Render the expandable similar-cases panels."""
    # Detect cases that share the same underlying rule (by fr_document_number)
    rule_groups = {}
    for i, case in enumerate(similar_cases_data):
        doc_num = str(case.get('fr_document_number', '') or '').strip()
        if doc_num:
            rule_groups.setdefault(doc_num, []).append(i)

    for i, case in enumerate(similar_cases_data):
        sim_score = float(case.get('similarity_score', 0) or 0)
        sim_pct = int(sim_score * 100)
        if sim_score >= 0.75:
            sim_bucket, sim_color = "Strong rule-level similarity", "#1a7f37"
        elif sim_score >= 0.60:
            sim_bucket, sim_color = "Moderate rule-level similarity", "#8a6d00"
        else:
            sim_bucket, sim_color = "Weak rule-level similarity", "#8a3a1f"
        cat = case.get('outcome_category', 'other')
        vuln_text, vuln_cls = vulnerability_from_outcome(cat)
        outcome_full = historical_outcome_label(cat)

        with st.expander(
            f"#{i+1}  {case.get('cl_case_name','Unknown')}  ·  {outcome_full}  ·  {sim_bucket}",
            expanded=(i == 0)
        ):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Rule Challenged**")
                fr_title = case.get('fr_title', 'Unknown')
                case_fr_url = case.get('fr_html_url', '')
                st.markdown(f"""
                <div style="font-size:0.9rem;">
                    <div style="font-weight:600;color:#033C5A;margin-bottom:0.3rem;">
                        {fr_title[:120]}{"..." if len(str(fr_title)) > 120 else ""}
                    </div>
                    <div style="color:#6b7280;font-size:0.8rem;">
                        {case.get('fr_agency_name','—')} · {case.get('fr_publication_date','—')}<br>
                        <span class="doctrine-tag" style="font-size:0.65rem;">
                            {case.get('administration_rule','')}
                        </span>
                    </div>
                    {"<br><a href='" + str(case_fr_url) + "' target='_blank' class='fr-link' style='font-size:0.8rem;'>→ FederalRegister.gov</a>" if case_fr_url and str(case_fr_url) != 'nan' else ""}
                </div>
                """, unsafe_allow_html=True)

            with cb:
                st.markdown("**Court Case**")
                case_cl_url = cl_slugged_url(case.get('cl_opinion_url', ''),
                                              case.get('cl_case_name', ''))
                st.markdown(f"""
                <div style="font-size:0.9rem;">
                    <div style="font-weight:600;color:#033C5A;margin-bottom:0.3rem;">
                        {case.get('cl_case_name','Unknown')}
                    </div>
                    <div style="margin-bottom:0.3rem;">{outcome_badge(vuln_cls.replace('badge-',''), outcome_full)}</div>
                    <div style="color:#6b7280;font-size:0.8rem;">
                        {case.get('circuit_name','—')} · Filed {case.get('date_filed','—')}<br>
                        <span class="doctrine-tag" style="font-size:0.65rem;">
                            {case.get('doctrine_era','')}
                        </span>
                    </div>
                    {"<br><a href='" + str(case_cl_url) + "' target='_blank' class='fr-link' style='font-size:0.8rem;'>→ CourtListener.com</a>" if case_cl_url and str(case_cl_url) != 'nan' else ""}
                </div>
                """, unsafe_allow_html=True)

            # Note when multiple cases in the list challenge the same rule
            doc_num = str(case.get('fr_document_number', '') or '').strip()
            if doc_num and len(rule_groups.get(doc_num, [])) > 1:
                siblings = rule_groups[doc_num]
                other_nums = [f"#{s+1}" for s in siblings if s != i]
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#7a1f1f;margin:0.3rem 0;">'
                    f'Note: This case challenges the same rule as {", ".join(other_nums)}. '
                    f'Different outcomes may reflect different stages of litigation or '
                    f'shifts in legal doctrine between filing dates.</div>',
                    unsafe_allow_html=True,
                )

            rerank_reason = case.get('rerank_reason', '')
            if rerank_reason:
                if isinstance(rerank_reason, dict):
                    labels = [
                        ('rule_analogy', 'Why this rule is analogous'),
                        ('case_relevance', 'How this case informs the prediction'),
                    ]
                    parts = []
                    for key, label in labels:
                        val = rerank_reason.get(key)
                        if val:
                            parts.append(
                                f'<div style="margin:0.6rem 0;">'
                                f'<div style="font-weight:600;color:#033C5A;'
                                f'font-size:0.9rem;margin-bottom:0.15rem;">{label}</div>'
                                f'<div style="font-size:0.9rem;line-height:1.45;">{val}</div>'
                                f'</div>'
                            )
                    if parts:
                        st.markdown(''.join(parts), unsafe_allow_html=True)
                else:
                    st.markdown("**Why this case is relevant:**")
                    st.markdown(rerank_reason)

            cluster_id = str(case.get('cl_cluster_id', '') or f'idx{i}')
            if st.checkbox("Show full opinion text",
                           key=f"show_op_{i}_{cluster_id}"):
                op_text = load_opinion_text(str(OPINIONS_DIR), case.get('cl_cluster_id', ''))
                if op_text:
                    st.markdown("**Full Opinion Text:**")
                    import re, html as html_mod

                    def _reflow_opinion(text):
                        """Reflow hard-wrapped prose paragraphs while
                        preserving short-line formatting (captions, parties,
                        headings).  Works paragraph-by-paragraph."""
                        # Some opinions have a blank line after every single
                        # line (PDF artifact).  Detect and collapse: if >60%
                        # of lines are blank, remove single blank lines but
                        # keep double blanks as paragraph breaks.
                        raw_lines = text.split('\n')
                        non_empty = sum(1 for l in raw_lines if l.strip())
                        if non_empty and len(raw_lines) > 10:
                            blank_ratio = 1 - non_empty / len(raw_lines)
                            if blank_ratio > 0.4:
                                # Collapse single blanks; keep 2+ as one break
                                collapsed = []
                                prev_blank = False
                                for ln in raw_lines:
                                    if not ln.strip():
                                        if prev_blank:
                                            collapsed.append('')
                                        prev_blank = True
                                    else:
                                        collapsed.append(ln)
                                        prev_blank = False
                                text = '\n'.join(collapsed)

                        paragraphs = re.split(r'\n\s*\n', text)
                        out = []
                        for para in paragraphs:
                            lines = [l for l in para.split('\n') if l.strip()]
                            if not lines:
                                continue
                            lengths = [len(l.strip()) for l in lines]
                            median_len = sorted(lengths)[len(lengths) // 2]
                            max_len = max(lengths)

                            # If many lines are individually long (>120),
                            # each is its own paragraph (no blank-line
                            # separators in the source file).
                            if len(lines) >= 2 and median_len > 120:
                                for ln in lines:
                                    s = ln.strip()
                                    if len(s) > 120:
                                        out.append(('prose', s))
                                    elif s:
                                        out.append(('pre', s))
                            # Hard-wrapped prose or single long line
                            elif (len(lines) >= 2 and median_len > 45) \
                                    or max_len > 120:
                                joined = ' '.join(l.strip() for l in lines)
                                joined = re.sub(r' {2,}', ' ', joined)
                                out.append(('prose', joined))
                            else:
                                out.append(('pre', '\n'.join(lines)))
                        return out

                    blocks = _reflow_opinion(op_text)
                    parts_html = []
                    for kind, content in blocks:
                        escaped = html_mod.escape(content)
                        if kind == 'prose':
                            parts_html.append(
                                f'<p style="margin:0 0 0.8rem 0;">{escaped}</p>')
                        else:
                            parts_html.append(
                                f'<pre style="white-space:pre-wrap;'
                                f'font-family:inherit;margin:0 0 0.8rem 0;'
                                f'font-size:0.85rem;line-height:1.4;">'
                                f'{escaped}</pre>')

                    st.markdown(
                        f'<div style="max-height:400px;overflow-y:auto;'
                        f'padding:1rem;background:#f9f9f9;'
                        f'border:1px solid #e5e7eb;border-radius:6px;'
                        f'font-size:0.85rem;line-height:1.6;">'
                        f'{"".join(parts_html)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Opinion text not available.")

def render_circuit_distribution(similar_cases_data):
    if not similar_cases_data:
        return
    st.markdown('<div class="section-header">Circuit Distribution of Analogs</div>',
                unsafe_allow_html=True)
    circuit_counts = {}
    vuln_by_circuit = {}
    for case in similar_cases_data:
        c = case.get('circuit_name', 'Unknown')
        circuit_counts[c] = circuit_counts.get(c, 0) + 1
        if c not in vuln_by_circuit:
            vuln_by_circuit[c] = {'high': 0, 'low': 0, 'other': 0}
        cat = case.get('outcome_category', 'other')
        if cat == 'loss':
            vuln_by_circuit[c]['high'] += 1
        elif cat == 'win':
            vuln_by_circuit[c]['low'] += 1
        else:
            vuln_by_circuit[c]['other'] += 1

    cols = st.columns(max(len(circuit_counts), 1))
    for i, (circuit, count) in enumerate(
        sorted(circuit_counts.items(), key=lambda x: x[1], reverse=True)
    ):
        vulns = vuln_by_circuit.get(circuit, {})
        breakdown = (
            f"{vulns.get('high',0)} Struck Down / "
            f"{vulns.get('low',0)} Upheld"
        )
        if vulns.get('other', 0):
            breakdown += f" / {vulns['other']} Unknown"
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{count}</div>
                <div class="label">{circuit}</div>
                <div style="font-size:0.75rem;color:#6b7280;margin-top:0.3rem;">
                    {breakdown}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar: FR search form ───────────────────────────────────────────────────
def render_search_sidebar(rulemakings_count: int, all_cases_count: int) -> dict:
    """Render the search form. Returns dict of submitted criteria, or None."""
    with st.sidebar:
        st.markdown('<div class="sidebar-section-header">Search Federal Register</div>',
                    unsafe_allow_html=True)

        agencies = get_agency_list()
        agency_lookup = {a['name']: a['slug'] for a in agencies
                         if (a.get('name') or '').strip().upper() != 'ACTION'}

        search_mode = st.radio(
            "Search mode",
            options=["Advanced search", "FR citation lookup"],
            horizontal=True,
            key="fr_search_mode",
            label_visibility="collapsed",
        )

        with st.form("fr_search_form", clear_on_submit=False):
            if search_mode == "Advanced search":
                term = st.text_input(
                    "Keyword(s)",
                    placeholder=f"e.g. {random.choice(EXAMPLE_KEYWORDS)}",
                    key="fr_keyword_input",
                )

                agency_names = st.multiselect(
                    "Agency", options=list(agency_lookup.keys()),
                    help="Filter by issuing agency (multiple allowed)",
                )

                c1, c2 = st.columns(2)
                with c1:
                    date_gte = st.date_input(
                        "Published from", value=None,
                        min_value=date(2008, 1, 1), max_value=date.today(),
                        key="fr_date_gte",
                    )
                with c2:
                    date_lte = st.date_input(
                        "Published to", value=None,
                        min_value=date(2008, 1, 1), max_value=date.today(),
                        key="fr_date_lte",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    cfr_title = st.text_input("CFR Title", placeholder="e.g. 40")
                with c4:
                    cfr_part = st.text_input("CFR Part", placeholder="e.g. 60")

                significant_only = st.checkbox(
                    "Significant only", value=False,
                )
                per_page = st.select_slider(
                    "Results per page", options=[10, 20, 50, 100], value=20,
                )
                fr_citation = ""
            else:
                fr_citation = st.text_input(
                    "FR Citation",
                    placeholder="e.g. 89 FR 31488",
                    key="fr_citation_input",
                )
                term = ""
                agency_names = []
                date_gte = None
                date_lte = None
                cfr_title = ""
                cfr_part = ""
                significant_only = False
                per_page = 20

            submitted = st.form_submit_button("Search", use_container_width=True)

        # Override the default "YYYY/MM/DD" placeholder on the date inputs with
        # faint "e.g. …" hints to match the keyword input style.
        import streamlit.components.v1 as _components_dates
        _components_dates.html("""
        <script>
        (function() {
            const setPH = () => {
                try {
                    const doc = window.parent.document;
                    const inputs = doc.querySelectorAll(
                        'section[data-testid="stSidebar"] input[aria-label]'
                    );
                    for (const inp of inputs) {
                        const label = inp.getAttribute('aria-label') || '';
                        if (label === 'Published from')
                            inp.setAttribute('placeholder', 'e.g. 2024-01-01');
                        else if (label === 'Published to')
                            inp.setAttribute('placeholder', 'e.g. ' + new Date().toISOString().slice(0,10));
                    }
                } catch (e) {}
            };
            setPH();
            setInterval(setPH, 500);
        })();
        </script>
        """, height=0)

        # Cycle the keyword placeholder client-side every ~2.5s while the input is empty.
        import streamlit.components.v1 as components
        import json as _json
        components.html(f"""
        <script>
        (function() {{
            const examples = {_json.dumps(EXAMPLE_KEYWORDS)};
            const TYPE_MS = 90, DELETE_MS = 45, HOLD_MS = 1400, GAP_MS = 350;
            let idx = Math.floor(Math.random() * examples.length);
            let pos = 0;             // current chars shown
            let phase = "typing";    // typing | holding | deleting | gap
            let lastSwitch = Date.now();

            const find = () => {{
                try {{
                    return window.parent.document.querySelector(
                        'section[data-testid="stSidebar"] input[aria-label="Keyword(s)"]'
                    );
                }} catch (e) {{ return null; }}
            }};

            const tick = () => {{
                const inp = find();
                if (!inp || inp.value) return;  // pause if user is typing
                const word = examples[idx % examples.length];
                const now = Date.now();

                if (phase === "typing") {{
                    pos++;
                    inp.setAttribute('placeholder', 'e.g. ' + word.slice(0, pos));
                    if (pos >= word.length) {{ phase = "holding"; lastSwitch = now; }}
                }} else if (phase === "holding") {{
                    if (now - lastSwitch >= HOLD_MS) {{ phase = "deleting"; }}
                }} else if (phase === "deleting") {{
                    pos--;
                    inp.setAttribute('placeholder', 'e.g. ' + word.slice(0, Math.max(0, pos)));
                    if (pos <= 0) {{ phase = "gap"; lastSwitch = now; idx++; }}
                }} else if (phase === "gap") {{
                    if (now - lastSwitch >= GAP_MS) {{ phase = "typing"; pos = 0; }}
                }}
            }};

            // Use a fast base interval; each phase decides whether to act.
            let last = 0;
            setInterval(() => {{
                const now = Date.now();
                const interval = phase === "deleting" ? DELETE_MS : TYPE_MS;
                if (now - last >= interval || phase === "holding" || phase === "gap") {{
                    tick();
                    last = now;
                }}
            }}, 30);
        }})();
        </script>
        """, height=0)

        st.markdown("---")
        st.markdown('<div class="sidebar-section-header">Reference Dataset</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-info">
            <strong>{all_cases_count:,}</strong> matched appellate cases<br>
            <strong>{rulemakings_count:,}</strong> rulemaking challenges<br>
            <strong>FY2008–2026</strong> coverage
        </div>
        """, unsafe_allow_html=True)

        # Collapse-sidebar button at the bottom (mirrors the native one at the top)
        components.html("""
        <div style="display:flex;justify-content:flex-end;padding:1rem 0 0 0;">
          <button id="rc-collapse-bottom" title="Collapse sidebar"
            style="background:transparent;border:none;cursor:pointer;
                   color:rgba(49,51,63,0.6);padding:0.25rem;
                   display:inline-flex;align-items:center;justify-content:center;
                   border-radius:0.5rem;line-height:1;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="11 17 6 12 11 7"></polyline>
              <polyline points="18 17 13 12 18 7"></polyline>
            </svg>
          </button>
        </div>
        <script>
        (function() {
            const btn = document.getElementById('rc-collapse-bottom');
            if (!btn) return;
            btn.addEventListener('mouseenter', () => {
                btn.style.backgroundColor = '#ede6d4';
            });
            btn.addEventListener('mouseleave', () => {
                btn.style.backgroundColor = 'transparent';
            });
            btn.addEventListener('click', () => {
                try {
                    const doc = window.parent.document;
                    const selectors = [
                        '[data-testid="stSidebarCollapseButton"] button',
                        '[data-testid="stSidebarCollapseButton"]',
                        'button[data-testid="stSidebarCollapseButton"]',
                        '[data-testid="stSidebarHeader"] button',
                        'section[data-testid="stSidebar"] button[kind="header"]',
                        'section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"]',
                        'section[data-testid="stSidebar"] header button',
                        'section[data-testid="stSidebar"] button[aria-label*="ollapse"]',
                        'button[aria-label*="ollapse sidebar"]',
                    ];
                    let native = null;
                    for (const sel of selectors) {
                        native = doc.querySelector(sel);
                        if (native) break;
                    }
                    if (!native) {
                        // Last resort: any button inside the sidebar whose svg has a chevron-left-ish path
                        const btns = doc.querySelectorAll('section[data-testid="stSidebar"] button');
                        for (const b of btns) { native = b; break; }
                    }
                    if (native) native.click();
                    else console.warn('rc-collapse-bottom: no native collapse button found');
                } catch (e) { console.warn(e); }
            });
        })();
        </script>
        """, height=60)

    return {
        "submitted": submitted,
        "term": term.strip(),
        "fr_citation": fr_citation.strip(),
        "agency_slugs": tuple(agency_lookup[n] for n in agency_names),
        "date_gte": date_gte.isoformat() if date_gte else None,
        "date_lte": date_lte.isoformat() if date_lte else None,
        "cfr_title": cfr_title.strip() or None,
        "cfr_part": cfr_part.strip() or None,
        "significant_only": significant_only,
        "doc_types": ("RULE",),
        "per_page": per_page,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>RegChallenger</h1>
        <p>AI-powered assessment of legal vulnerability for federal rules</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        all_cases, rulemakings = get_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    if rulemakings.empty:
        st.warning("No confirmed rulemaking cases found.")
        return

    embeddings = get_embeddings(rulemakings)
    criteria = render_search_sidebar(len(rulemakings), len(all_cases))

    # ── Run search on submit ──────────────────────────────────────────────────
    if criteria["submitted"]:
        if not _run_search(criteria, page=1):
            return
        st.session_state["selected_doc_number"] = None
        st.session_state["assess_for"] = None

    results = st.session_state.get("search_results")
    if results is None:
        st.info("← Use the sidebar to search the Federal Register, then select a rule "
                "to view its details and assess its legal vulnerability.")
        return

    if not results:
        st.warning("No matching rules. Adjust your search criteria.")
        return

    total = st.session_state.get("search_count", len(results))
    cur_page = st.session_state.get("search_page", 1)
    total_pages = st.session_state.get("search_total_pages", 1)
    start_idx = (cur_page - 1) * criteria["per_page"] + 1
    end_idx = start_idx + len(results) - 1

    st.markdown(
        f'<div class="section-header">Search Results '
        f'<span style="font-size:0.85rem;color:#6b7280;font-weight:400;">'
        f'(showing {start_idx:,}–{end_idx:,} of {total:,} · '
        f'page {cur_page} of {total_pages})</span></div>',
        unsafe_allow_html=True,
    )

    # ── Pagination controls (top) ─────────────────────────────────────────────
    if total_pages > 1:
        pc1, pc2, pc3, pc4 = st.columns([1, 1, 4, 2])
        stored_criteria = st.session_state.get("search_criteria", criteria)
        with pc1:
            if st.button("◀ Prev", disabled=(cur_page <= 1),
                         use_container_width=True, key="prev_top"):
                _run_search(stored_criteria, page=cur_page - 1)
                st.rerun()
        with pc2:
            if st.button("Next ▶", disabled=(cur_page >= total_pages),
                         use_container_width=True, key="next_top"):
                _run_search(stored_criteria, page=cur_page + 1)
                st.rerun()
        with pc4:
            jump = st.number_input(
                "Jump to page", min_value=1, max_value=total_pages,
                value=cur_page, step=1, label_visibility="collapsed",
                key="page_jump",
            )
            if jump != cur_page:
                _run_search(stored_criteria, page=int(jump))
                st.rerun()

    # ── Result selection (scrollable table with row selection) ────────────────
    table_rows = []
    for r in results:
        challenged = not lookup_previously_challenged(
            rulemakings, r.get("fr_document_number", "")
        ).empty
        table_rows.append({
            "⚖︎": "⚖︎" if challenged else "",
            "Title": (r.get("fr_title") or "Untitled"),
            "Agency": r.get("fr_agency_name", ""),
            "Published": r.get("fr_publication_date", ""),
            "Significant": "Yes" if str(r.get("fr_significant", "")) == "1" else "",
            "Citation": r.get("fr_citation_official", ""),
            "_doc": r.get("fr_document_number", ""),
        })
    results_df = pd.DataFrame(table_rows)

    st.caption("Select a row to analyze that rule. ⚖︎ = previously challenged in our dataset.")
    event = st.dataframe(
        results_df.drop(columns=["_doc"]),
        hide_index=True,
        use_container_width=True,
        height=min(420, 80 + 35 * len(results_df)),
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "⚖︎": st.column_config.TextColumn(width="small"),
            "Title": st.column_config.TextColumn(width="large"),
            "Agency": st.column_config.TextColumn(width="medium"),
            "Published": st.column_config.TextColumn(width="small"),
            "Significant": st.column_config.TextColumn(width="small"),
            "Citation": st.column_config.TextColumn(width="small"),
        },
    )

    sel_rows = event.selection.rows if hasattr(event, "selection") else []
    if not sel_rows:
        st.info("Select a rule to view its details.")
        return
    chosen_doc = results_df.iloc[sel_rows[0]]["_doc"]
    selected = next(r for r in results if r["fr_document_number"] == chosen_doc)
    selected = enrich_query_rule(dict(selected))

    # Check if previously challenged
    challenged_rows = lookup_previously_challenged(rulemakings, chosen_doc)
    is_challenged = not challenged_rows.empty

    # Prediction uses a FIXED evidence set (PREDICTION_N) so the vulnerability
    # assessment is a property of the rule, not the display slider.
    PREDICTION_N = 5
    # Cache is keyed on the document only: toggling display widgets never re-runs the AI.
    analysis_key = (chosen_doc,)
    cached = st.session_state.get("analysis_cache", {}).get(analysis_key)

    # Two-step flow: selecting a row shows rule details immediately; the
    # expensive embedding+rerank+LLM pipeline only runs when the user clicks
    # "Generate Vulnerability Assessment" (or is served from cache).
    assess_requested = (
        cached is not None
        or st.session_state.get("assess_for") == chosen_doc
    )

    # ── Layout: rule details in col1, assessment (or CTA) in col2 ────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Selected Rule</div>',
                    unsafe_allow_html=True)
        render_rule_card(selected)
        if is_challenged:
            st.markdown(
                f'<div style="font-size:0.85rem;color:#7a1f1f;font-weight:600;'
                f'margin:0.5rem 0;">⚖︎ This rule has been challenged in '
                f'{len(challenged_rows)} case(s) in our dataset.</div>',
                unsafe_allow_html=True,
            )
            for _, case_row in challenged_rows.iterrows():
                render_court_case_card(case_row)

    # Both render paths (CTA and assessment) write to col2 with the SAME
    # structure: a header, then a single empty() slot for body content.
    # This keeps Streamlit's widget diff stable so the CTA button is
    # cleanly removed the moment Step 2 begins.
    with col2:
        st.markdown(
            f'<div class="section-header">'
            f'{"Historical Outcome" if is_challenged else "Vulnerability Assessment"}'
            f'</div>',
            unsafe_allow_html=True,
        )
        col2_body = st.empty()

    # ── Step 1: if user hasn't requested analysis, show CTA and stop ─────────
    if not assess_requested:
        with col2_body.container():
            if is_challenged:
                blurb = (
                    "This rule has been challenged in our dataset. "
                    "Click below to see the historical outcome and an AI "
                    "analysis of the court's reasoning."
                )
                btn_label = "Show Historical Analysis"
            else:
                blurb = (
                    "Retrieve historical court challenges to analogous rules "
                    "and use them to assess the legal vulnerability of this rule."
                )
                btn_label = "Generate Vulnerability Assessment"
            st.info(blurb)
            def _request_assess(doc=chosen_doc):
                st.session_state["assess_for"] = doc
            st.button(btn_label, type="primary", use_container_width=True,
                      key=f"assess_btn_{chosen_doc}",
                      on_click=_request_assess)
        return

    # ── Step 2: user has requested — run pipeline and render into col2_body ──
    # Streamlit defers widget removals until script end, so the CTA button
    # from the previous render lingers visibly during the long reranking
    # spinner. Force-hide it with CSS targeting the widget's stable key class.
    st.markdown(
        '<style>[class*="st-key-assess_btn_"]{display:none !important;}</style>',
        unsafe_allow_html=True,
    )
    col2_body.empty()

    with col2_body.container():

        # Compute similar_cases_data (cached, trivial for challenged, or full pipeline)
        if cached is not None:
            similar_cases_data = cached["similar_cases_data"]
        elif is_challenged:
            similar_cases_data = []
            for _, row in challenged_rows.iterrows():
                d = row.to_dict()
                d['similarity_score'] = 1.0
                similar_cases_data.append(d)
        else:
            # Fast local step (embed + cosine search): runs silently in ~1s,
            # which gives Streamlit time to reconcile the DOM and clear the
            # CTA button from the previous render before the spinner appears.
            q_emb = embed_query_rule(selected)
            pool_k = max(15, PREDICTION_N * 3)
            sim = find_similar_to_query(
                q_emb, selected, rulemakings, embeddings, top_k=pool_k,
            )
            pool = []
            for case_idx, sim_score in sim:
                d = rulemakings.iloc[case_idx].to_dict()
                d['similarity_score'] = sim_score
                pool.append(d)

            if os.environ.get('ANTHROPIC_API_KEY') and len(pool) > PREDICTION_N:
                with st.spinner("Retrieving analogous rules and ranking legal challenges..."):
                    from utils.prediction import rerank_candidates
                    pool_opinions = {
                        str(c.get('cl_cluster_id', '')): load_opinion_text(
                            str(OPINIONS_DIR), c.get('cl_cluster_id', '')
                        )
                        for c in pool
                    }
                    order, reasons = rerank_candidates(selected, pool, pool_opinions,
                                                        top_k=PREDICTION_N)
                    similar_cases_data = []
                    for i in order:
                        d = pool[i]
                        if i in reasons:
                            d['rerank_reason'] = reasons[i]
                        similar_cases_data.append(d)
            else:
                similar_cases_data = pool[:PREDICTION_N]

        # Render prediction / historical outcome in col2
        if is_challenged:
            prediction = None
            for _, case_row in challenged_rows.iterrows():
                vuln = vulnerability_from_outcome(case_row.get('outcome_category', 'other'))
                vuln_label = historical_outcome_label(case_row.get('outcome_category', 'other'))
                box_cls = vulnerability_box_class(vuln[0])
                st.markdown(f"""
                <div class="prediction-box {box_cls}">
                    <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;
                                letter-spacing:0.05em;margin-bottom:0.4rem;">Actual Outcome</div>
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                                font-weight:700;color:#033C5A;margin-bottom:0.6rem;">{vuln_label}</div>
                    <div style="font-size:0.85rem;color:#4b5563;">
                        <strong>Court:</strong> {case_row.get('circuit_name','—')}<br>
                        <strong>Disposition:</strong> {case_row.get('outcome_label','—')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── AI narrative explanation ──────────────────────────────────
            if not os.environ.get('ANTHROPIC_API_KEY'):
                st.warning("⚠️ ANTHROPIC_API_KEY is not configured — the AI explanation is disabled.")
            else:
                case_dicts = [r.to_dict() for _, r in challenged_rows.iterrows()]
                opinions = {
                    str(c.get('cl_cluster_id', '')): load_opinion_text(
                        str(OPINIONS_DIR), c.get('cl_cluster_id', '')
                    )
                    for c in case_dicts
                }
                if cached and cached.get("explanation") is not None:
                    explanation = cached["explanation"]
                else:
                    with st.spinner("Generating historical analysis..."):
                        explanation = explain_historical_outcome(selected, case_dicts, opinions)

                if explanation.get('what_happened'):
                    st.markdown("**What Happened**")
                    st.markdown(explanation['what_happened'])
                if explanation.get('legal_reasoning'):
                    st.markdown("**Legal Reasoning**")
                    st.markdown(explanation['legal_reasoning'])
                if explanation.get('key_factors'):
                    st.markdown("**Key Factors**")
                    for f in explanation['key_factors']:
                        st.markdown(f"• {f}")
                if explanation.get('implications'):
                    st.markdown("**Implications**")
                    st.markdown(explanation['implications'])
                if not any([explanation.get('what_happened'),
                            explanation.get('legal_reasoning')]):
                    st.caption("AI analysis unavailable.")
                    with st.expander("Raw response"):
                        st.text(explanation.get('raw_response', ''))

                op_count = sum(1 for v in opinions.values() if v)
                if op_count:
                    st.caption(f"Analysis informed by {op_count}/{len(opinions)} opinion text(s) on disk.")
                else:
                    st.caption("Note: no opinion text on disk for this case — analysis is based on metadata only.")
        else:
            if not os.environ.get('ANTHROPIC_API_KEY'):
                st.warning("⚠️ ANTHROPIC_API_KEY is not configured — AI predictions are disabled.")
                prediction = None
            elif cached and cached.get("prediction") is not None:
                prediction = cached["prediction"]
                if prediction:
                    render_prediction_box(prediction)
            else:
                with st.spinner("Analyzing..."):
                    prediction = predict_outcome(selected, similar_cases_data, "")
                if prediction:
                    render_prediction_box(prediction)

    # ── Similar cases (full width, below prediction so opinion text has room) ─
    if similar_cases_data:
        if is_challenged:
            st.markdown('<div class="section-header">Court Cases for This Rule</div>',
                        unsafe_allow_html=True)
            # Show all historical challenges — no cap
            cases_to_show = similar_cases_data
        else:
            st.markdown('<div class="section-header">Closest Historical Analogs</div>',
                        unsafe_allow_html=True)
            st.caption(
                "Ranked by legal relevance to the selected rule — which may differ "
                "from the rule-level similarity shown per case. The closest analog is "
                "not necessarily the strongest predictor. See the Vulnerability "
                "Assessment reasoning for how each case informs the prediction."
            )
            cases_to_show = similar_cases_data
        render_similar_cases(cases_to_show)

    # ── Circuit distribution (full width, only for non-challenged path) ──────
    if similar_cases_data and not is_challenged:
        render_circuit_distribution(similar_cases_data)

    # ── Persist analysis results so toggling display-only widgets doesn't re-run AI ──
    if cached is None:
        st.session_state.setdefault("analysis_cache", {})[analysis_key] = {
            "similar_cases_data": similar_cases_data,
            "prediction": locals().get("prediction"),
            "explanation": locals().get("explanation"),
        }


if __name__ == "__main__":
    main()
