# RegChallenger — Claude Code Briefing

## Project Overview

A Streamlit dashboard ("RegChallenger — Federal Rule Vulnerability Analysis") that assesses the legal vulnerability of federal rules using RAG-based similarity search over historical cases and Claude AI.

---

## Current Status (as of 2026-04-15)

**Pipeline: COMPLETE.** 251 confirmed rulemaking cases (briefing previously said 244; the enriched step3 added 7 more `FOUND` rows) with full FR metadata in `dashboard/data/`.

**Dashboard: FR SEARCH FLOW WORKING END-TO-END.** Live FR API search → results table → row selection → either "previously challenged" historical analysis OR on-the-fly embedding + AI vulnerability prediction. Both AI paths return structured output rendered in styled cards.

**Retrieval: UPGRADED 2026-04-15.** mpnet embeddings + LLM-rerank pipeline now in place. 4,264 opinion texts available locally (synced from EC2). Prediction is stable across reruns (`temperature=0`) and independent of the display slider (`PREDICTION_N=5` fixed).

---

## What Was Built Today (2026-04-15, post opinion-sync)

### Retrieval quality overhaul
- **Embedder upgrade**: `all-MiniLM-L6-v2` → `all-mpnet-base-v2` (384 → 768 dim). Cache key now includes `MODEL_NAME` so embeddings auto-invalidate on future model swaps (`utils/embeddings.py`).
- **Boost reduction**: `AGENCY_MATCH_BOOST` 0.15 → 0.08 (was dominating and compressing scores for single-agency queries).
- **LLM reranker** added (`rerank_candidates` in `utils/prediction.py`): embeddings pull a top-15 candidate pool; Claude then reranks by *legal* relevance using ~2500-char opinion excerpts for each candidate. Candidates pass with doctrine era, administration, outcome, and case metadata. Falls back to embedding order on API failure.
- **Prediction decoupled from display slider**: `PREDICTION_N = 5` fixed pool for the vulnerability assessment; the sidebar "similar cases to retrieve" slider only controls how many of those 5 are rendered. Previously, changing the slider mid-session would re-run Claude and shift the vulnerability rating — no longer.
- **Session caching** of analysis results keyed by `(chosen_doc,)` — toggling "show full opinion text" or any other display-only widget no longer triggers re-embedding / rerank / prediction calls.
- **Determinism**: `temperature=0` set on all Claude calls (rerank, prediction, historical explanation) for stable run-to-run output.

### UX/display refinements
- **Similarity display**: replaced raw `%` with qualitative buckets — **Strong match** (≥0.75), **Moderate match** (0.60–0.75), **Weak match** (<0.60). Expander label includes the bucket; a small note reads "Order reflects AI reranking by legal relevance" to explain why bucket can disagree with display order.
- **Historical outcome language**: "Highly Vulnerable / Moderately Vulnerable / Not Vulnerable" replaced with **Struck Down / Mixed Outcome / Upheld** for decided historical cases (court-case card, similar-cases expander, Historical Outcome right-column box). Vulnerability language is retained only in the **AI Prediction** box for uncharted query rules, where it's a prediction rather than a fact.
- **Circuit Distribution card**: "3 High / 0 Low" → "3 Struck Down / 0 Upheld".
- **Layout**: "Most Relevant Historical Cases" (renamed from "Most Similar") is now full-width below the AI Prediction / Historical Outcome row, giving opinion text room to breathe (prior layout placed it in a narrow left column that broke legal citations awkwardly).
- **Removed all user-visible "Claude" references** → "AI" (error captions, notes, labels). Internal docstrings/comments untouched.
- **Checkbox label**: "Economically significant only" → "Significant only" (the FR API param `conditions[significant]=1` covers all significance categories, not just Section 3(f)(1) economically significant).
- **Microcopy**: "Click a row to analyze that rule." → "Select a row…"; main rule card link "→ View on FederalRegister.gov" → "→ FederalRegister.gov" (matching the compact similar-case card style); CourtListener links now read "→ CourtListener.com".
- **Scales glyph**: swapped `⚖️` (emoji) → `⚖︎` (text variant, VS15 selector) everywhere *except* `page_icon` — less sticker-like in the dashboard's typographic context, but browser tab favicons still need emoji presentation.
- **Agency dropdown**: filtered out the defunct **ACTION** agency (merged into AmeriCorps in 1993, hadn't issued rules in decades).
- **CSV download removed**: raw-FR-API export added no dashboard-specific value — will reintroduce later if we export enrichment (predictions, similar-case attribution).

### CourtListener 404 fix
- `cl_slugged_url(url, case_name)` helper (`app.py`): when a stored `cl_opinion_url` is slugless (`/opinion/187165/`), appends a hyphenated slug derived from `cl_case_name` (`/opinion/187165/state-of-new-jersey-v-epa/`). CL no longer reliably redirects slugless URLs — slugged URLs resolve. Applied in both the main court-case card and similar-cases expander.

### Document-type filter — confirmed not removable
- Dataset is ~76% `Rule`, but genuinely contains 19 `Notice` and 8 `Proposed Rule` entries (`fr_type` distribution in `step3_output.csv`). Kept the document-type multiselect with `RULE` default so Notices and Proposed Rules remain reachable.

### Key file touchpoints today
- `utils/embeddings.py` — model + boost + cache key
- `utils/prediction.py` — new `rerank_candidates`, `temperature=0` in `call_claude`
- `app.py` — layout, bucket labels, historical-outcome helper, slug helper, cache, PREDICTION_N, ACTION filter, many microcopy edits

---

## What Was Built Last Session (2026-04-14)

### 1. `.env` config (python-dotenv)
- `requirements.txt` — added `python-dotenv>=1.0.0`
- `app.py` loads `.env` from project root before importing `prediction.py` (so `ANTHROPIC_API_KEY` is in `os.environ` when `prediction.py` reads it at import time)
- `.gitignore` — excludes `.env`, `embeddings_cache.json`, `secrets.toml`
- `.env.example` — template

### 2. `utils/fr_api.py` — Live FR API client
- `get_agencies()` — fetches and caches the full FR agency list (~470 agencies) for the sidebar multiselect
- `search_documents(...)` — full search wrapper (term, agency_slugs, date range, CFR title/part, significant_only, doc_types, per_page, page)
- `normalize_fr_result(doc)` — converts an FR API document into the same field schema as `step3_output.csv` (so the rest of the dashboard reuses the same display/embedding code)

### 3. `utils/embeddings.py` — On-the-fly query embedding
- `get_embedder()` — module-level cached SentenceTransformer (avoids reloading)
- `embed_query_rule(rule_dict)` — embeds an arbitrary FR rule dict using `build_rule_text`
- `find_similar_to_query(q_emb, q_row, df, embeddings, top_k)` — hybrid scoring against pre-computed dataset embeddings

### 4. `utils/prediction.py` — Historical-outcome explainer
- `explain_historical_outcome(rule, cases, opinions)` — when a selected FR rule was previously challenged in our dataset, Claude generates a structured 4-section narrative (`WHAT HAPPENED` / `LEGAL REASONING` / `KEY FACTORS` / `IMPLICATIONS`) drawing on the actual opinion text + claude_reasoning + categorical context
- `parse_historical_explanation(text)` — companion parser
- Defensive NaN handling added to `build_prediction_prompt` (case dict values can be NaN floats)

### 5. `app.py` — Major rewrite around FR search
- **New entry flow**: sidebar search form (with cycling typed-effect placeholder) → results table with row selection → analysis
- **Two paths after selection**:
  - Previously challenged (⚖️ flag) → render court case card(s) + Claude historical narrative
  - Not challenged → render rule card + on-the-fly embed + top-k similar cases + Claude vulnerability prediction
- **Layout**: 2-column (left: rule card + similar cases; right: AI prediction or historical outcome). Eliminated whitespace gap by computing similar cases before column rendering and placing them in col1.
- Helper functions: `render_rule_card`, `render_court_case_card`, `render_prediction_box`, `render_similar_cases`, `render_circuit_distribution`, `render_search_sidebar`
- **HTML safety**: `_h(value)` helper escapes user-controlled fields; switched problematic cards from `st.markdown` to `st.html` (no markdown interpretation, fixes broken rendering on titles/abstracts containing `<` `>`)
- **Pagination**: `_run_search(criteria, page)` stores criteria in session_state; ◀ Prev / Next ▶ buttons + jump-to-page input
- **Download all results as CSV**: button below results table iterates every page from FR API and offers full CSV
- **Sidebar**: 460px wide (CSS override), search form with form_submit_button (clear_on_submit=False, key="fr_keyword_input" so typed term persists)
- **Cycling typed-effect placeholder** for keyword input via `streamlit.components.v1.html` injecting JS that targets `section[data-testid="stSidebar"] input[type="text"]` in `window.parent.document`. Pauses when input has any value. Examples in `EXAMPLE_KEYWORDS` (20 non-partisan topics across env/health/labor/finance/immigration/defense)
- **Branding**: "RegChallenger" + "— Federal Rule Vulnerability Analysis" suffix in lighter gold; subtitle "AI-powered assessment of legal challenge risk for federal rules"
- **Top padding** of main `.block-container` reduced to 1.5rem so title aligns with sidebar header

---

## File Structure

```
dashboard/
├── app.py                          # Main Streamlit app
├── utils/
│   ├── data_loader.py              # CSV merge + opinion loader + lookup_previously_challenged
│   ├── embeddings.py               # MiniLM embeddings + hybrid scoring
│   ├── fr_api.py                   # Live Federal Register API client (NEW)
│   └── prediction.py               # Claude prediction + historical explanation
├── data/
│   ├── step1_output.csv            # 10,680 rows
│   ├── step2_output.csv            # 4,403 rows
│   ├── step3_output.csv            # 251 rulemaking rows (enriched)
│   ├── opinions/                   # full pipeline opinions (NEEDS DOWNLOAD from EC2)
│   └── validation_opinions/        # 77 validation opinion files (current fallback)
├── requirements.txt
├── .env                            # NOT committed
├── .env.example
├── .gitignore
└── CLAUDE_CODE_BRIEFING.md
```

---

## Multimodal Model Architecture (unchanged)

### Text Embeddings (Semantic Similarity)
**Model:** `all-MiniLM-L6-v2` (sentence-transformers)

**What gets embedded per rule:**
```
fr_title | fr_abstract (500 chars) | fr_explanation (300 chars) |
Agency: {fr_agency_name} | CFR: {cfr_references} |
CFR Topics: {fr_cfr_topics} | Topics: {fr_topics} |
Case: {cl_case_name} | {claude_reasoning (400 chars)}
```

### Categorical Variables (Structural Similarity)
Applied as boosts on top of cosine similarity:

| Variable | Boost | Purpose |
|----------|-------|---------|
| `fr_agency_name` | +0.15 | Same regulatory domain |
| `doctrine_era` | +0.05 | Same legal framework |
| `administration_rule` | +0.05 | Same political context for promulgation |

For an on-the-fly query rule (FR search hit not in our dataset), `enrich_query_rule()` derives `administration_rule` and `doctrine_era_rule` from `fr_publication_date`, and sets `doctrine_era` / `administration_case` to TODAY's values (since a hypothetical challenge would be filed now).

Score capped at 1.0. Final = cosine_similarity + categorical_boost.

### Categorical Variables in Prediction Prompt
- `fr_significant`, `fr_page_length`, `fr_action`, `administration_rule` vs `administration_case`, `doctrine_era`
- `fr_president` is display-only (doesn't distinguish Trump45/Trump47)

### Vulnerability Framing (challenger's perspective)
- OUTCOME=1 (Affirmed) → Rule Upheld → Low Vulnerability
- OUTCOME=2 (Reversed/Vacated) → Rule Struck Down → High Vulnerability
- OUTCOME=3 (Mixed) → Moderate Vulnerability
- OUTCOME=6 (Remanded) → Moderate Vulnerability

---

## Pipeline Outputs (in dashboard/data/)

**step1_output.csv** (10,680 rows): `cl_cluster_id`, `idb_docket`, `circuit`, `appellant`, `appellee`, `cl_case_name`, `cl_case_name_full`, `cl_opinion_url`, `agency`, `agency_name`, `outcome`, `apptype`, `date_filed`, `confidence`, `administration_case`, `doctrine_era`, `outcome_label`, `judgment_date`, `doctrine_era_judgment`, `administration_judgment`

**step2_output.csv** (4,403 rows): `cl_cluster_id`, `circuit`, `agency`, `agency_name`, `cl_case_name`, `date_filed`, `outcome`, `claude_case_type`, `claude_challenged_fr`, `claude_reasoning`, `claude_confidence`, `has_fr_citations`, `opinion_text_len`

**step3_output.csv** (251 rulemaking rows, enriched): `cl_cluster_id`, `fr_document_number`, `fr_citation_official`, `fr_title`, `fr_publication_date`, `fr_agency_name`, `fr_parent_department`, `fr_sub_agency`, `rin_numbers`, `cfr_references`, `fr_html_url`, `fr_significant`, `fr_lookup_status`, `fr_type`, `fr_action`, `fr_abstract`, `fr_explanation`, `fr_cfr_topics`, `fr_topics`, `fr_page_length`, `fr_president`, `fr_effective_on`

**CRITICAL: Join key is `cl_cluster_id`** (NOT `validation_id`).

---

## Data Loader Key Functions

`load_dataset(step1, step2, step3)` — merges all three on `cl_cluster_id`, returns `(all_cases_df, rulemakings_df)`. Filters to `claude_case_type == 'RULEMAKING'` and `fr_lookup_status startswith 'FOUND'`. Deduplicates by `cl_cluster_id`. Adds derived `administration_rule` / `doctrine_era_rule` from publication date.

`load_opinion_text(opinions_dir, cluster_id)` — looks for `cluster_{id}.txt`, falls back to `cluster_{id}_*.txt` (validation format)

`lookup_previously_challenged(rulemakings_df, fr_document_number)` — returns matching rows if rule was previously challenged

`_clean_cluster_id(val)` — strips `.0` float suffix

---

## App Architecture (current — implemented)

**Imports:** `dotenv.load_dotenv` runs first so `ANTHROPIC_API_KEY` is available before `prediction.py` reads it.

**OPINIONS_DIR:** Auto-detects `data/opinions/` (full pipeline) or falls back to `data/validation_opinions/` (current state, only 77 files).

**Flow:**
1. Load 251 rulemaking cases + compute embeddings (cached)
2. Sidebar: search form (keyword w/ typed-effect placeholder, agency multiselect, date range, CFR title/part, doc types, significance, results-per-page) + analysis settings (top_k, show_opinion) + reference-dataset stats
3. Main:
   - On submit → live FR API call (cached 10 min) → results stored in session_state with pagination
   - Results table (`st.dataframe` with `selection_mode="single-row"`, `on_select="rerun"`) — ⚖️ icon flags previously-challenged rules
   - Pagination controls (◀ Prev / Next ▶ / jump-to-page) + Download all CSV button
   - On row selection:
     - **Previously challenged**: rule card → court case card(s) → "Historical Outcome" badge + Claude narrative explanation (4 sections)
     - **Not challenged**: rule card → on-the-fly embed → top-k similar cases → Claude vulnerability prediction
   - Similar cases panels (rendered in left column to avoid whitespace) + circuit distribution

---

## Styling (GW Colors)

- Primary: `#033C5A` (GW navy)
- Accent: `#D6BF91` (GW gold)
- Fonts: Playfair Display (headers), Source Sans 3 (body)
- Similarity bar: `#CCE3F4` → `#0075C8`
- Vulnerability colors: red (#dc2626) = high, orange (#d97706) = moderate, green (#16a34a) = low
- Sidebar: 460px fixed width, `#faf8f3` background

---

## Pending Issues (priority order)

1. 🔲 **Move dashboard out of iCloud Drive** before demo presentation
2. 🔲 **Git repo** — README, organize into `src/` folder
3. 🔲 **Presentation slides**
4. 🔲 **[Future / post-capstone] Structured doctrinal tagging of opinions** — one-time Claude pass over the 4,264 opinions to extract a per-case JSON profile (challenge type, doctrines cited, holding). Then match query against candidates via embedding + overlap scoring on these tags. Discussed today as the next major retrieval-quality improvement after reranking. Rough cost: ~$5–10, runs once, caches to JSON on disk.
5. ✅ ~~FR Search interface~~ — DONE
6. ✅ ~~Opinion-text sync from EC2~~ — DONE (4,264 files in `data/opinions/` as of 2026-04-15)

### Smaller polish items observed during build
- The `st.dataframe` toolbar (eye/download/search/fullscreen icons) cannot be hidden via Streamlit API
- File-watching is flaky on macOS without `watchdog` installed (`pip install watchdog` would fix the "changes don't appear" issue without manual restart)
- The `top_k` slider should probably be capped at `PREDICTION_N = 5` with a label like "Similar cases to show" since it's purely a display control now — not yet done

---

## Doctrine Era Boundaries
- Pre-Chevron (pre-1984): before 1984-06-25
- Chevron Era (1984–2024): 1984-06-25 to 2019-06-26
- Post-Kisor (2019–2022): 2019-06-26 to 2022-06-30
- Post-WV v. EPA (2022–2024): 2022-06-30 to 2024-06-28
- Post-Loper Bright (2024–): 2024-06-28 onward

## Administration Boundaries
- Bush 43: 2001-01-20 to 2009-01-20
- Obama: 2009-01-20 to 2017-01-20
- Trump (1st / Trump45): 2017-01-20 to 2021-01-20
- Biden: 2021-01-20 to 2025-01-20
- Trump (2nd / Trump47): 2025-01-20 onward

**NOTE:** `fr_president` from FR API does NOT distinguish Trump45 vs Trump47 — both return "Donald Trump". Always use derived `administration_rule`/`administration_case` for scoring and display.

## Outcome Codes (IDB)
- 1: Affirmed → Rule Upheld → Low Vulnerability
- 2: Reversed/Vacated → Rule Struck Down → High Vulnerability
- 3: Mixed → Moderate Vulnerability
- 5: Dismissed
- 6: Remanded → Moderate Vulnerability
- 7: Other

---

## How to Run

```bash
cp .env.example .env
# edit .env to add ANTHROPIC_API_KEY
pip install -r requirements.txt
streamlit run app.py
```
