# RegChallenger

**AI-powered assessment of legal vulnerability for federal rules.**

Live dashboard: [regchallenger.streamlit.app](https://regchallenger.streamlit.app/)

## Problem

Federal rules routinely face court challenges, but agencies, affected industries, and public-interest groups have no fast way to estimate how a specific rule would fare if litigated. Expert legal analysis is slow and expensive, and a rule's vulnerability depends on fact-specific details (the reviewing circuit, the doctrinal era, the agency's statutory authority, and the pattern of outcomes in analogous historical challenges) that are hard to weigh consistently by hand.

## Proposed Solution
 
RegChallenger is a dashboard that, given any final rule published in the Federal Register, produces a structured vulnerability assessment grounded in historical analogs. The pipeline:

1. **Retrieves** the 15 most similar historical rulemaking challenges from a curated reference set of 395 challenged rulemakings (FY 2008–2026), using hybrid dense-embedding + categorical similarity.
2. **Reranks** those candidates with Claude by *legal* relevance — doctrinal fit, reviewing circuit, current doctrine regime, outcome informativeness.
3. **Predicts** vulnerability (`Highly` / `Moderately` / `Minimally Vulnerable`) with a two-sided assessment: factors exposing the rule and factors favoring its survival, each cited to specific analog cases.
4. **Explains historical outcomes** when the selected rule has already been challenged — synthesizing the opinion text, doctrinal era, and political moment.

Leave-one-out validation on 20 stratified cases (10 upheld / 10 struck down) shows 75% directional accuracy when the tool commits to a High or Low rating (100% precision on High calls). See `DEVELOPMENT_LOG.md` for details.

## Tech Stack

- **Streamlit** — web UI
- **Anthropic Claude (Sonnet)** — LLM reranker and vulnerability assessor
- **sentence-transformers (`all-mpnet-base-v2`)** — 768-dim dense embeddings
- **Federal Register API** — live rule search
- **CourtListener** — source of historical court opinions (upstream)
- **pandas / numpy** — data handling
- **plotly** — visualizations

## Steps to Launch the Demo

### Option A — use the hosted dashboard

Open [regchallenger.streamlit.app](https://regchallenger.streamlit.app/). No setup required.

### Option B — run locally

```bash
git clone https://github.com/H-Hirsch/RegChallenger.git
cd RegChallenger
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env    # paste your Anthropic API key
streamlit run src/app.py
```

Opens at `http://localhost:8501`. First run generates the embeddings cache (~20 seconds); subsequent runs are instant.

Requires an Anthropic API key (Claude). The dashboard gracefully degrades without one — similarity search and historical lookups still work; AI predictions and narrative explanations are disabled.

## Repository Layout

```
RegChallenger/
├── README.md
├── requirements.txt              # Dashboard dependencies
├── src/                          # Source code
│   ├── app.py                    # Streamlit entry point
│   └── utils/                    # Data loading, embeddings, prediction, FR API
├── data/                         # Dataset (CSVs + opinion texts)
└── DEVELOPMENT_LOG.md
```

## Dataset

- **395 rulemaking challenges** — federal rules that went to a published circuit-court opinion, enriched with Federal Register metadata (agency, CFR references, publication date, abstract, topics, significance flag).
- **7,773 matched appellate cases** in the broader corpus (the rulemaking challenges are the subset Claude classified as RULEMAKING and that successfully matched a Federal Register rule).
- **Outcome distribution**: 46.6% Upheld, 24.1% Struck Down, 18.5% Mixed, 10.9% Dismissed/Other.
- **Coverage**: FY 2008–2026, all circuits.
- Opinion text for each case is in `data/opinions/`.

The upstream data pipeline used to build this dataset (FJC IDB matching → CourtListener opinion fetch → Federal Register enrichment) is maintained privately. The processed outputs used by the dashboard are included in `data/`.

## Known Limitations

- Leave-one-out validation is optimistic (the tool was designed around this dataset).
- Reference set only contains rules that were actually challenged — unchallenged rules aren't represented, which biases retrieval toward controversial regulatory domains.
- Multi-agency joint rulemakings in the historical corpus carry only the primary agency; query-side handling of joint agencies was added but the corpus side is unchanged.
- CourtListener coverage gaps result in a ~42% match rate for pre-2015 cases; more recent cases match at 50–60%.
- FERC and FCC cases citing internal order numbers rather than FR page citations are underrepresented.
- Tool output is an analytical aid, not a legal opinion.
