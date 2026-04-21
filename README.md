# RegChallenger

**AI-powered assessment of legal vulnerability for federal rules.**

RegChallenger is a Streamlit dashboard that estimates how a federal rule would fare if challenged in court. A user searches the Federal Register, selects a rule, and the tool retrieves the most legally analogous historical challenges, reranks them by doctrinal relevance, and produces a structured vulnerability assessment grounded in those analogs.

Developed as a Master's capstone project at the George Washington University Regulatory Studies Center.

## What it does

Given any final rule published in the Federal Register:

1. **Retrieves** the 15 most similar historical rulemaking challenges from a curated reference set of 251 appellate cases (FY 2008–2026) using hybrid embedding + categorical similarity.
2. **Reranks** those candidates with Claude by *legal* relevance — doctrinal fit, reviewing circuit, current doctrine regime, outcome informativeness.
3. **Predicts** vulnerability (`Highly` / `Moderately` / `Minimally Vulnerable`) with a two-sided assessment: factors exposing the rule and factors favoring its survival, grounded in specific analog cases.
4. **Explains historical outcomes** when the selected rule has already been challenged — synthesizing the opinion text, doctrinal era, and political moment.

## Dataset

- **251 rulemaking challenges** — federal rules that went to a published circuit-court opinion, enriched with Federal Register metadata (agency, CFR references, publication date, abstract, significance).
- **Outcome distribution**: 43% Upheld, 24.7% Struck Down, 19.9% Mixed, 12.4% Dismissed/Unknown.
- **Coverage**: FY 2008–2026, all circuits.
- Opinion text for each case is included in `data/opinions/`.

The upstream pipeline (CourtListener scraping, Federal Register matching, data enrichment) will be released separately.

## Running locally

```bash
git clone https://github.com/H-Hirsch/RegChallenger.git
cd RegChallenger
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your Anthropic API key
streamlit run app.py
```

Opens at `http://localhost:8501`. First run will generate the embeddings cache (~20 seconds); subsequent runs are instant.

Requires an Anthropic API key (Claude). The dashboard gracefully degrades without one — similarity search and historical lookups still work; AI predictions and narrative explanations are disabled.

## Validation

Leave-one-out validation on 20 stratified cases (10 upheld / 10 struck down) shows 75% directional accuracy when the tool commits to a High or Low rating (100% precision on High calls). See `DEVELOPMENT_LOG.md` for details.

## Stack

- **Streamlit** — UI
- **Anthropic Claude (Sonnet)** — LLM reranker and assessor
- **sentence-transformers (`all-mpnet-base-v2`)** — 768-dim dense embeddings
- **Federal Register API** — live rule search
- **CourtListener** — court opinion source (upstream)

## Known limitations

- Leave-one-out validation is optimistic (the tool was designed around this dataset).
- Reference set only contains rules that were actually challenged — unchallenged rules aren't represented, which biases retrieval toward controversial regulatory domains.
- Multi-agency joint rulemakings in the historical corpus carry only the primary agency; query-side handling of joint agencies was added but the corpus side is unchanged.
- Tool output is an analytical aid, not a legal opinion.

## License

See [LICENSE](LICENSE).
