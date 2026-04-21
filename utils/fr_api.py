"""
fr_api.py
Live Federal Register API client for searching documents and fetching agency list.

Endpoint: https://www.federalregister.gov/api/v1/
No auth required. Used by the dashboard's FR search interface.
"""

from functools import lru_cache
from typing import Optional, List, Dict
import re
import requests
import streamlit as st

BASE = "https://www.federalregister.gov/api/v1"

DEFAULT_FIELDS = [
    "document_number", "citation", "title", "publication_date",
    "agencies", "cfr_references", "html_url", "significant",
    "abstract", "type", "action", "page_length", "president",
    "effective_on", "topics",
]


@lru_cache(maxsize=1)
def get_agencies() -> List[Dict]:
    """
    Fetch full agency list (cached per process).
    Returns list of dicts with at least 'slug' and 'name'.
    """
    try:
        r = requests.get(f"{BASE}/agencies", timeout=15)
        r.raise_for_status()
        data = r.json()
        agencies = [
            {"slug": a.get("slug"), "name": a.get("name")}
            for a in data
            if a.get("slug") and a.get("name")
        ]
        agencies.sort(key=lambda a: a["name"])
        return agencies
    except Exception:
        return []


def search_documents(
    term: str = "",
    agency_slugs: Optional[List[str]] = None,
    date_gte: Optional[str] = None,
    date_lte: Optional[str] = None,
    cfr_title: Optional[str] = None,
    cfr_part: Optional[str] = None,
    significant_only: bool = False,
    doc_types: Optional[List[str]] = None,
    per_page: int = 50,
    page: int = 1,
) -> Dict:
    """
    Search the Federal Register.

    Returns dict with keys: 'results' (list), 'count' (int total matches),
    'total_pages' (int), 'error' (str or None).
    """
    params: List = []

    if term:
        params.append(("conditions[term]", term))
    for slug in (agency_slugs or []):
        params.append(("conditions[agencies][]", slug))
    if date_gte:
        params.append(("conditions[publication_date][gte]", date_gte))
    if date_lte:
        params.append(("conditions[publication_date][lte]", date_lte))
    if cfr_title:
        params.append(("conditions[cfr][title]", str(cfr_title)))
    if cfr_part:
        params.append(("conditions[cfr][part]", str(cfr_part)))
    if significant_only:
        params.append(("conditions[significant]", "1"))
    for t in (doc_types or ["RULE"]):
        params.append(("conditions[type][]", t))

    for f in DEFAULT_FIELDS:
        params.append(("fields[]", f))

    params.append(("per_page", str(min(per_page, 1000))))
    params.append(("page", str(page)))
    params.append(("order", "newest"))

    try:
        r = requests.get(f"{BASE}/documents.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return {
            "results": data.get("results", []),
            "count": data.get("count", 0),
            "total_pages": data.get("total_pages", 0),
            "error": None,
        }
    except requests.HTTPError as e:
        # FR API returns 400 with empty conditions; try to surface body
        try:
            body = r.json()
            msg = body.get("errors") or body.get("message") or str(e)
        except Exception:
            msg = str(e)
        return {"results": [], "count": 0, "total_pages": 0, "error": str(msg)}
    except Exception as e:
        return {"results": [], "count": 0, "total_pages": 0, "error": str(e)}


@st.cache_data(show_spinner=False)
def lookup_by_citation(citation: str) -> Dict:
    """
    Look up a document by FR citation (e.g. "89 FR 31488").

    Parses volume + page, filters by volume and an estimated date window,
    then matches the exact page number client-side.

    Cached via @st.cache_data: citations are stable identifiers, so a hit
    avoids up to 4 FR API round-trips. Cache resets on app redeploy.
    """
    from datetime import date, timedelta

    m = re.match(r'(\d+)\s+FR\s+([\d,]+)', citation.strip(), re.IGNORECASE)
    if not m:
        return {"results": [], "count": 0, "total_pages": 0,
                "error": "Invalid citation format. Expected e.g. '89 FR 31488'."}

    volume = int(m.group(1))
    target_page = int(m.group(2).replace(',', ''))
    base_year = 2008 + (volume - 73)  # vol 73 = 2008

    est_day = target_page / 300  # ~300 FR pages per calendar day
    est_date = date(base_year, 1, 1) + timedelta(days=int(est_day))
    window_start = max(est_date - timedelta(days=5), date(base_year, 1, 1))
    window_end = min(est_date + timedelta(days=5), date(base_year, 12, 31))

    def _fetch_window(d_start, d_end):
        params = [
            ("conditions[volume]", str(volume)),
            ("conditions[publication_date][gte]", d_start.isoformat()),
            ("conditions[publication_date][lte]", d_end.isoformat()),
        ]
        for f in DEFAULT_FIELDS:
            params.append(("fields[]", f))
        params += [("per_page", "1000"), ("order", "oldest")]
        r = requests.get(f"{BASE}/documents.json", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])

    def _page_num(doc):
        cm = re.match(r'(\d+)\s+FR\s+([\d,]+)', doc.get("citation", ""))
        return int(cm.group(2).replace(',', '')) if cm else None

    try:
        for _ in range(4):
            results = _fetch_window(window_start, window_end)
            for doc in results:
                if _page_num(doc) == target_page:
                    return {"results": [doc], "count": 1,
                            "total_pages": 1, "error": None}
            if not results:
                break
            pages = [p for p in (_page_num(d) for d in results) if p]
            if not pages:
                break
            if target_page > max(pages):
                window_start = window_end + timedelta(days=1)
                window_end = min(window_start + timedelta(days=10),
                                 date(base_year, 12, 31))
            elif target_page < min(pages):
                window_end = window_start - timedelta(days=1)
                window_start = max(window_end - timedelta(days=10),
                                   date(base_year, 1, 1))
            else:
                break  # target in range but no exact match — page doesn't exist
        return {"results": [], "count": 0, "total_pages": 0, "error": None}
    except Exception as e:
        return {"results": [], "count": 0, "total_pages": 0, "error": str(e)}


def normalize_fr_result(doc: Dict) -> Dict:
    """
    Convert an FR API document into the field schema used by the rest of the
    dashboard (matching step3 enriched columns).
    """
    agencies = doc.get("agencies") or []
    primary = agencies[0] if agencies else {}
    parent = primary.get("parent_id")  # not always populated
    # Capture ALL agencies for joint rulemakings, not just the first.
    agency_names = [a.get("name", "").strip() for a in agencies
                    if a.get("name", "").strip()]
    all_agencies_str = "; ".join(agency_names)
    cfr_refs = doc.get("cfr_references") or []
    cfr_str = "; ".join(
        f"{c.get('title','?')} CFR {c.get('part','?')}" for c in cfr_refs
    ) if cfr_refs else ""

    return {
        "fr_document_number":   doc.get("document_number", ""),
        "fr_citation_official": doc.get("citation", ""),
        "fr_title":             doc.get("title", ""),
        "fr_publication_date":  doc.get("publication_date", ""),
        "fr_agency_name":       all_agencies_str or primary.get("name", ""),
        "fr_parent_department": primary.get("parent_id") or "",
        "fr_sub_agency":        "",
        "cfr_references":       cfr_str,
        "fr_html_url":          doc.get("html_url", ""),
        "fr_significant":       "1" if doc.get("significant") else "0",
        "fr_type":              doc.get("type", ""),
        "fr_action":            doc.get("action", ""),
        "fr_abstract":          doc.get("abstract", "") or "",
        "fr_explanation":       "",
        "fr_cfr_topics":        "",
        "fr_topics":            ", ".join(doc.get("topics", []) or []),
        "fr_page_length":       str(doc.get("page_length", "") or ""),
        "fr_president":         (doc.get("president") or {}).get("name", "")
                                if isinstance(doc.get("president"), dict)
                                else (doc.get("president") or ""),
        "fr_effective_on":      doc.get("effective_on", "") or "",
        "rin_numbers":          "",
    }
