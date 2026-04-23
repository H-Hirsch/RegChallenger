"""
prediction.py
Claude API call for outcome prediction based on similar historical cases.
"""

import os
from datetime import date
import requests
from typing import List
import pandas as pd


API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2000


def _today_block() -> str:
    """Anchor prompts to the real current date so doctrine-era reasoning is correct."""
    today = date.today().isoformat()
    return (
        f"Today's date: {today}. Loper Bright (June 2024-present) is the binding "
        "deference regime today. Pre-June-2024 cases applied Chevron deference."
    )


def build_prediction_prompt(
    query_row: pd.Series,
    similar_cases: List[dict],
    query_opinion_text: str = "",
) -> str:
    """
    Build the prediction prompt for Claude.
    """
    def _s(d, k, default='Unknown'):
        v = d.get(k) if hasattr(d, 'get') else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return str(v)

    abstract = _s(query_row, 'fr_abstract', '')
    abstract_line = (
        f"- Abstract: {abstract[:800]}" if abstract and abstract != 'Unknown'
        else "- Abstract: (not available)"
    )

    # Query rule context — this is a rule under prospective analysis, NOT a
    # rule that has already been challenged. Only include rule-side fields.
    query_context = f"""SELECTED RULE (under prospective vulnerability analysis):
- Title: {_s(query_row, 'fr_title')}
- Agency: {_s(query_row, 'fr_agency_name')}
- Parent Department: {_s(query_row, 'fr_parent_department')}
- Published: {_s(query_row, 'fr_publication_date')} ({_s(query_row, 'administration_rule')} administration)
- Doctrine Era at publication: {_s(query_row, 'doctrine_era_rule')}
- CFR References: {_s(query_row, 'cfr_references', 'None')}
- Significant: {_s(query_row, 'fr_significant', '0')}
- Type / Action: {_s(query_row, 'fr_type')} / {_s(query_row, 'fr_action')}
{abstract_line}"""

    # Format similar cases
    similar_context = "\n\nMOST SIMILAR HISTORICAL CASES:\n"
    for i, case in enumerate(similar_cases, 1):
        similar_context += f"""
Case {i} (Similarity: {case['similarity_score']:.2f}):
- Rule: {_s(case, 'fr_title')[:120]}
- Agency: {_s(case, 'fr_agency_name')}
- Case: {_s(case, 'cl_case_name')}
- Circuit: {_s(case, 'circuit_name')}
- Filed: {_s(case, 'date_filed')} ({_s(case, 'doctrine_era')})
- Rule Published: {_s(case, 'fr_publication_date')} ({_s(case, 'administration_rule')} administration)
- OUTCOME: {_s(case, 'outcome_label')}
- Legal Reasoning: {_s(case, 'claude_reasoning', 'Not available')[:300]}
"""

    prompt = f"""You are a federal administrative law expert assessing the legal vulnerability of federal rules and regulations. Render a calibrated, two-sided assessment — identify both the factors that would expose the rule to challenge AND the factors that would favor the rule in court. Do not default to high vulnerability; many federal rules survive challenge.

{_today_block()}

Based on the selected rule and the most similar historical cases provided, assess:
1. How vulnerable this rule is to being struck down or weakened if challenged in court
2. Which circuit(s) would be most likely to hear a challenge — accounting for both strategic forum-shopping by challengers and any statutorily-mandated venue (e.g., exclusive DC Circuit review under the Clean Air Act, Communications Act, or similar statutes)
3. The specific legal vulnerabilities that could be exploited by challengers
4. The specific factors that would favor the rule's survival (clear statutory authority, comprehensive administrative record, procedural regularity, recent affirmances of similar regulatory actions, agency expertise in the subject matter)

{query_context}
{similar_context}

IMPORTANT CONTEXT:
- Base rate in the 251-case reference set: 43.0% Upheld, 24.7% Struck Down, 19.9% Mixed, 12.4% Unknown/Dismissed. The plurality outcome is UPHELD — calibrate your rating to this historical record.
- Under Chevron (1984-2024): Courts deferred to agency statutory interpretations if reasonable
- Under Loper Bright (June 2024-present): Courts interpret statutes de novo — no deference to agency interpretations
- The doctrine era at time of challenge matters enormously for vulnerability
- Rules from a prior administration are often more vulnerable under a new administration that may not defend them vigorously
- Consider procedural vulnerabilities (APA notice-and-comment, cost-benefit analysis) AND procedural strengths (thorough record, reasoned explanation)
- Consider substantive vulnerabilities (statutory authority, major questions doctrine) AND substantive strengths (clear statutory grant, long-standing regulatory tradition in the area)

Calibration guidance (apply against the 5 historical analogs provided):
- Reserve HIGH for cases where at least 3 of the 5 analogs were struck down AND the current doctrinal regime (Loper Bright) disfavors the rule AND the rule's defenses are weak.
- Use MODERATE when the 5 analogs are split between upheld and struck down, or when 3-4 were upheld but the rule has distinctive vulnerabilities (e.g., cross-administration posture, major-questions footprint).
- Use LOW when at least 3 of the 5 analogs were upheld, the rule rests on clear statutory authority, and it lacks the features that drive reversals (major questions doctrine, weak procedural record, cross-administration defense posture).

Provide your analysis in the following format:

VULNERABILITY: [High / Moderate / Low]
CONFIDENCE: [High / Medium / Low]
MOST LIKELY CIRCUIT: [Circuit name(s)]
KEY VULNERABILITIES:
[2-4 bullet points identifying specific legal vulnerabilities grounded in similar cases]
FACTORS FAVORING THE RULE:
[2-4 bullet points identifying specific factors that would support the rule's survival, grounded in similar cases where rules with comparable features were upheld]
REASONING:
[3-5 sentences weighing the vulnerabilities against the factors favoring the rule to arrive at the rating. Explicitly engage with cases on BOTH sides of the outcome ledger where applicable (upheld analogs and struck-down analogs). Whenever you reference a historical case from the MOST SIMILAR HISTORICAL CASES list above, you MUST cite it with the italicized case name FIRST, immediately followed by its rank in parentheses using the exact form "(Case N)" — for example: "the 5th Circuit's reasoning in *Mexican Gulf Fishing Co. v. U.S. Dep't of Commerce* (Case 2) applies here". Never put the rank before the name. Every case cited as a predictive influence must carry its "(Case N)" parenthetical. Per Bluebook convention, wrap every court-case name in single-asterisk markdown italics — e.g., *Loper Bright Enterprises v. Raimondo*, *Mexican Gulf Fishing Co. v. U.S. Dep't of Commerce*, *California Sea Urchin Comm'n v. Combs*. Italicize both full and short-form case names (including "v." and any procedural phrase like "In re"). Do not use double asterisks.]"""

    return prompt


def call_claude(prompt: str) -> str:
    """Call Claude API and return response text."""
    if not API_KEY:
        return "Error: ANTHROPIC_API_KEY not set"

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception as e:
        return f"Error calling AI API: {e}"


def parse_prediction(response_text: str) -> dict:
    """Parse Claude's structured response into components."""
    result = {
        'vulnerability': None,
        'confidence': None,
        'most_likely_circuit': None,
        'key_vulnerabilities': [],
        'favoring_factors': [],
        'reasoning': None,
        'raw_response': response_text,
    }

    lines = response_text.strip().split('\n')
    current_section = None
    vulnerabilities = []
    favoring = []
    reasoning_lines = []

    def _norm(s):
        """Strip markdown bold/italic wrappers for header matching."""
        return s.replace('**', '').replace('__', '').strip()

    def _is_bullet(s):
        return s.lstrip('*').lstrip().startswith(('-', '•')) or s.startswith(('-', '•'))

    for line in lines:
        line = line.strip()
        norm = _norm(line)
        upper = norm.upper()
        if upper.startswith('VULNERABILITY:'):
            result['vulnerability'] = _norm(norm.split(':', 1)[1])
        elif upper.startswith('CONFIDENCE:'):
            result['confidence'] = _norm(norm.split(':', 1)[1])
        elif upper.startswith('MOST LIKELY CIRCUIT:'):
            result['most_likely_circuit'] = _norm(norm.split(':', 1)[1])
        elif upper.startswith('KEY VULNERABILITIES:'):
            current_section = 'vulnerabilities'
        elif upper.startswith('FACTORS FAVORING THE RULE:'):
            current_section = 'favoring'
        elif upper.startswith('REASONING:'):
            current_section = 'reasoning'
        elif current_section == 'vulnerabilities' and _is_bullet(line):
            clean = line.lstrip('-•* ').replace('**', '').strip()
            if clean:
                vulnerabilities.append(clean)
        elif current_section == 'favoring' and _is_bullet(line):
            clean = line.lstrip('-•* ').replace('**', '').strip()
            if clean:
                favoring.append(clean)
        elif current_section == 'reasoning' and line:
            reasoning_lines.append(_norm(line))

    result['key_vulnerabilities'] = vulnerabilities
    result['favoring_factors'] = favoring
    result['reasoning'] = ' '.join(reasoning_lines) if reasoning_lines else None

    return result


def build_historical_explanation_prompt(
    rule: dict,
    cases: List[dict],
    opinions: dict,
) -> str:
    """Prompt Claude to explain the actual historical outcome(s) for a rule."""
    def _s(d, k, default='Unknown'):
        v = d.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return str(v)

    rule_ctx = f"""RULE CHALLENGED:
- Title: {_s(rule, 'fr_title')}
- Agency: {_s(rule, 'fr_agency_name')}
- Published: {_s(rule, 'fr_publication_date')} ({_s(rule, 'administration_rule')} administration)
- Citation: {_s(rule, 'fr_citation_official')}
- CFR References: {_s(rule, 'cfr_references')}
- Type / Action: {_s(rule, 'fr_type')} / {_s(rule, 'fr_action')}
- Significant: {_s(rule, 'fr_significant')}
- Abstract: {_s(rule, 'fr_abstract')[:600]}"""

    cases_ctx = "\n\nCOURT CHALLENGE(S) IN OUR DATASET:\n"
    for i, c in enumerate(cases, 1):
        op = opinions.get(_s(c, 'cl_cluster_id', ''), '')
        op_excerpt = (op[:6000] + '... [truncated]') if len(op) > 6000 else op
        cases_ctx += f"""
Case {i}:
- Case Name: {_s(c, 'cl_case_name')}
- Circuit: {_s(c, 'circuit_name')}
- Filed: {_s(c, 'date_filed')} ({_s(c, 'administration_case')} administration)
- Doctrine Era at filing: {_s(c, 'doctrine_era')}
- Outcome: {_s(c, 'outcome_label')}
- Pipeline summary of challenge: {_s(c, 'claude_reasoning', 'Not available')[:500]}
- OPINION TEXT (excerpt):
{op_excerpt if op_excerpt else '[Opinion text not available]'}
"""

    return f"""You are a federal administrative law expert explaining the historical outcome of a court challenge to a federal rule. Synthesize the rule's regulatory context, the political/doctrinal moment, and the actual judicial reasoning into a clear, useful narrative for a regulatory analyst.

{_today_block()}

{rule_ctx}
{cases_ctx}

CONTEXT:
- Pre-Loper Bright (before June 2024): Chevron deference applied — courts deferred to reasonable agency interpretations
- Post-Loper Bright (June 2024+): No Chevron deference — courts interpret statutes de novo
- Cross-administration challenges (rule from one party, challenge under another) are often more vulnerable
- Major Questions Doctrine (West Virginia v. EPA, 2022) restricts agency action on issues of vast economic/political significance
- If the opinion text excerpt above reads "[Opinion text not available]", say so explicitly in LEGAL REASONING and base the analysis on metadata only — do not fabricate quotations or doctrinal citations you cannot verify from the provided material.

Produce your analysis in this exact format:

WHAT HAPPENED:
[2-3 sentences: who challenged, what the court did, the legal disposition]

LEGAL REASONING:
[3-5 sentences explaining the court's reasoning, drawing specifically from the opinion text when available. Cite the specific legal doctrines, statutes, or precedents the court relied on. Per Bluebook convention, wrap every court-case name in single-asterisk markdown italics — e.g., *Loper Bright Enterprises v. Raimondo*, *West Virginia v. EPA*. Italicize both full and short-form case names (including "v." and any procedural phrase like "In re"). Do not use double asterisks.]

KEY FACTORS:
[2-4 bullet points: the structural/categorical factors that drove this outcome — e.g., circuit ideology, doctrine era, cross-administration dynamics, agency's procedural posture]

IMPLICATIONS:
[2-3 sentences: what this case tells us about the vulnerability of similar rules going forward, particularly under the current Loper Bright doctrinal regime]"""


def parse_historical_explanation(text: str) -> dict:
    """Parse the structured historical explanation."""
    result = {
        'what_happened': None,
        'legal_reasoning': None,
        'key_factors': [],
        'implications': None,
        'raw_response': text,
    }
    sections = {'WHAT HAPPENED:': 'what_happened',
                'LEGAL REASONING:': 'legal_reasoning',
                'KEY FACTORS:': 'key_factors',
                'IMPLICATIONS:': 'implications'}
    current = None
    buf = []

    def _flush():
        if current is None:
            return
        joined = ' '.join(b.strip() for b in buf if b.strip())
        if current == 'key_factors':
            items = [b.lstrip('-•* ').replace('**', '').strip()
                     for b in buf if b.strip().lstrip('-•* ')]
            result[current] = [i for i in items if i]
        else:
            result[current] = joined or None

    for raw in text.strip().split('\n'):
        line = raw.strip()
        norm = line.replace('**', '').replace('__', '').strip()
        upper = norm.upper()
        matched = next((k for k in sections if upper.startswith(k)), None)
        if matched:
            _flush()
            current = sections[matched]
            buf = []
            tail = norm.split(':', 1)[1].strip()
            if tail:
                buf.append(tail)
        elif current:
            buf.append(line)
    _flush()
    return result


def explain_historical_outcome(
    rule: dict,
    cases: List[dict],
    opinions: dict,
) -> dict:
    """Generate Claude narrative explaining the actual outcome of a previously-challenged rule."""
    prompt = build_historical_explanation_prompt(rule, cases, opinions)
    response = call_claude(prompt)
    return parse_historical_explanation(response)


def rerank_candidates(
    query_row,
    candidates: List[dict],
    opinions: dict,
    top_k: int = 5,
) -> List[int]:
    """
    LLM rerank of embedding candidates using opinion-text reasoning.

    Args:
        query_row: dict-like with query rule metadata
        candidates: list of dicts, each with fr_title, fr_agency_name,
                    cl_case_name, outcome_label, cl_cluster_id, similarity_score
        opinions: {cl_cluster_id(str): opinion_text or None}
        top_k: number of reranked candidates to return

    Returns:
        List of original candidate indices in reranked order, length <= top_k.
        Falls back to embedding order on any parse/API failure.
    """
    if not candidates:
        return []
    if not API_KEY:
        return list(range(min(top_k, len(candidates))))

    import json as _json
    q = dict(query_row) if not isinstance(query_row, dict) else query_row

    def _g(d, k, default=''):
        v = d.get(k, default)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return str(v)

    query_block = (
        f"Title: {_g(q, 'fr_title')}\n"
        f"Agency: {_g(q, 'fr_agency_name')}\n"
        f"CFR: {_g(q, 'cfr_references')}\n"
        f"Administration: {_g(q, 'administration_rule')}\n"
        f"Doctrine Era: {_g(q, 'doctrine_era')}\n"
        f"Abstract: {_g(q, 'fr_abstract')[:600]}"
    )

    def _norm(s):
        return (s or '').strip().lower()

    def _agency_set(s):
        return {a.strip().lower() for a in str(s or '').split(';') if a.strip()}

    q_agencies = _agency_set(_g(q, 'fr_agency_name'))
    q_circuit = _norm(_g(q, 'circuit_name'))
    q_doctrine = _norm(_g(q, 'doctrine_era'))
    q_admin = _norm(_g(q, 'administration_rule'))
    q_cfr = _norm(_g(q, 'cfr_references'))

    def _cfr_overlap(a, b):
        if not a or not b:
            return False
        a_parts = {p.strip() for p in a.replace(';', ',').split(',') if p.strip()}
        b_parts = {p.strip() for p in b.replace(';', ',').split(',') if p.strip()}
        return bool(a_parts & b_parts)

    cand_blocks = []
    for i, c in enumerate(candidates):
        op = opinions.get(str(_g(c, 'cl_cluster_id')), '') or ''
        op_snippet = op[:2500] if op else '(no opinion text on disk)'
        sim = c.get('similarity_score')
        sim_str = f"{float(sim):.3f}" if isinstance(sim, (int, float)) else 'n/a'
        c_agencies = _agency_set(_g(c, 'fr_agency_name'))
        c_circuit = _norm(_g(c, 'circuit_name'))
        c_doctrine = _norm(_g(c, 'doctrine_era'))
        c_admin = _norm(_g(c, 'administration_rule'))
        c_cfr = _norm(_g(c, 'cfr_references'))
        matches = {
            'same_agency': bool(q_agencies) and bool(q_agencies & c_agencies),
            'same_circuit': bool(q_circuit) and q_circuit == c_circuit,
            'same_doctrine_era': bool(q_doctrine) and q_doctrine == c_doctrine,
            'same_administration': bool(q_admin) and q_admin == c_admin,
            'cfr_overlap': _cfr_overlap(q_cfr, c_cfr),
        }
        match_str = ', '.join(f"{k}={v}" for k, v in matches.items())
        cand_blocks.append(
            f"[{i}] Rule: {_g(c, 'fr_title')[:150]}\n"
            f"    Agency: {_g(c, 'fr_agency_name')} | Case: {_g(c, 'cl_case_name')} | "
            f"Circuit: {_g(c, 'circuit_name')} | Outcome: {_g(c, 'outcome_label')}\n"
            f"    Doctrine: {_g(c, 'doctrine_era')} | Admin: {_g(c, 'administration_rule')} | "
            f"CFR: {_g(c, 'cfr_references')}\n"
            f"    Embedding similarity (hybrid text score, 0-1): {sim_str}\n"
            f"    Categorical matches vs query: {match_str}\n"
            f"    Opinion excerpt: {op_snippet}"
        )

    prompt = f"""You are a federal administrative law expert. Rerank the following historical rule-challenge cases by how legally relevant each is to the SELECTED RULE, for the purpose of predicting how the selected rule would fare if challenged.

{_today_block()}

Judge relevance using overlap in legal doctrines the court applied (e.g., Chevron, Loper Bright, major questions, arbitrary-and-capricious, statutory authority, procedural/APA), similarity of regulatory action, substantive subject-matter overlap, and the legal framework in place when the case was decided. Raw topical overlap matters less than doctrinal fit.

Each candidate includes precomputed metadata flags and an embedding similarity score. Use them to verify your factual claims — if a flag is False, do not assert the match — but do NOT name the flags in your explanations. Write naturally, as if briefing a regulatory analyst.

SELECTED RULE (the rule under analysis):
{query_block}

HISTORICAL RULES / CASES (indexed candidates):
{chr(10).join(cand_blocks)}

Return ONLY a JSON object (no prose, no markdown) of the form:
{{"ranking": [{{"index": i1, "rule_analogy": "...", "case_relevance": "..."}}, ...]}}

- "rule_analogy": Given the selected rule and this historical rule side-by-side, what makes them analogous as regulatory instruments? Consider the issuing agency and statutory authority, the regulatory mechanism, CFR overlap, the administration that promulgated each, and substantive subject matter. Write as flowing prose, not a checklist. Do not discuss the court case, circuit, outcome, or doctrine era here.

- "case_relevance": Given that this historical rule is analogous to the selected rule, why does its corresponding court case inform the prediction — and why does the case rank here relative to the cases above and below it? Consider the reviewing circuit, whether the case was decided under a doctrine regime still in force today (post-Loper Bright is the binding standard now), the outcome and what it signals, and what the opinion excerpt reveals about the court's reasoning. Explain what makes it more or less predictive than its neighbors.

Keep each explanation to 2-3 sentences of natural prose. Return the top {top_k} candidates from most to least legally relevant. Only include indices that appear in the candidate list."""

    raw = call_claude(prompt)
    try:
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("no JSON object in response")
        obj = _json.loads(raw[start:end + 1])
        entries = obj.get('ranking', [])
        ranking = []
        reasons = {}
        for entry in entries:
            reason_obj = {}
            if isinstance(entry, dict):
                idx = int(entry.get('index', -1))
                for key in ('rule_analogy', 'case_relevance'):
                    val = entry.get(key)
                    if isinstance(val, str) and val.strip():
                        reason_obj[key] = val.strip()
                # back-compat: if Claude returned legacy "reason" field, keep it
                legacy = entry.get('reason')
                if isinstance(legacy, str) and legacy.strip() and not reason_obj:
                    reason_obj['rule_analogy'] = legacy.strip()
            elif isinstance(entry, (int, str)):
                idx = int(entry)
            else:
                continue
            if 0 <= idx < len(candidates):
                ranking.append(idx)
                if reason_obj:
                    reasons[idx] = reason_obj
        # dedupe preserving order
        seen = set()
        ranking = [i for i in ranking if not (i in seen or seen.add(i))]
        if not ranking:
            raise ValueError("empty ranking")
        # pad with any unranked candidates in original order
        for i in range(len(candidates)):
            if i not in seen:
                ranking.append(i)
        return ranking[:top_k], reasons
    except Exception:
        return list(range(min(top_k, len(candidates)))), {}


def predict_outcome(
    query_row: pd.Series,
    similar_cases: List[dict],
    query_opinion_text: str = "",
) -> dict:
    """
    Main prediction function.

    Args:
        query_row: pandas Series with rule metadata
        similar_cases: list of dicts with similar case data + similarity scores
        query_opinion_text: full opinion text for the query case (optional)

    Returns:
        dict with prediction results
    """
    prompt = build_prediction_prompt(query_row, similar_cases, query_opinion_text)
    response = call_claude(prompt)
    result = parse_prediction(response)
    # Retry once if the response failed to parse key fields (transient API issue
    # or malformed output). This catches cases like empty vulnerability/reasoning
    # before surfacing a blank assessment to the user.
    if not result.get('vulnerability') or not result.get('reasoning'):
        response = call_claude(prompt)
        retry = parse_prediction(response)
        if retry.get('vulnerability') and retry.get('reasoning'):
            result = retry
    return result
