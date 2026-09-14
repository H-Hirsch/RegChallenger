#!/usr/bin/env python3
"""
build_embeddings_cache.py — regenerate data/embeddings_cache.json.

WHEN TO RUN
    After ANY change to the rulemaking corpus that alters what gets embedded:
      * a new pipeline run / added or removed rulemaking cases
      * re-classification that changes which cases are RULEMAKING
      * FR-enrichment edits that rewrite embedded text (title/abstract/
        explanation/agency/CFR refs/topics) or changed claude_reasoning
    Then COMMIT the regenerated data/embeddings_cache.json so the deployed
    Streamlit Cloud app loads embeddings from disk instead of downloading the
    420MB model and re-encoding every rule on every cold start.

WHAT DOES *NOT* REQUIRE A REGENERATE
    Opinion .txt files. Embeddings are built ONLY from FR rule metadata +
    claude_reasoning (see embeddings.build_rule_text). Opinions are loaded
    separately at display/prediction time and never enter the embedding.

SAFETY
    The cache is a pure speed optimization. If it is stale or missing, the app
    silently recomputes at runtime (just slower) — it never serves wrong data.
    NOTE: the cache key is based on the rulemaking row count/index (+ whether
    fr_abstract exists + model name), NOT a content hash. So same-size text
    edits will NOT auto-invalidate — you must rerun this script for those.

USAGE
    python scripts/build_embeddings_cache.py
    python scripts/build_embeddings_cache.py --data /path/to/data --batch-size 16
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils.data_loader import load_dataset          # noqa: E402
from utils import embeddings as E                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Regenerate the embeddings cache.")
    ap.add_argument("--data", default=str(ROOT / "data"),
                    help="folder holding step1/2/3_output.csv (default: repo data/)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="encode batch size; lower it if you hit memory limits")
    a = ap.parse_args()
    d = Path(a.data)

    merged, rm = load_dataset(
        str(d / "step1_output.csv"),
        str(d / "step2_output.csv"),
        str(d / "step3_output.csv"),
    )
    print(f"rulemakings loaded: {len(rm)} rows")

    # Build the EXACT cache key that embeddings.compute_embeddings expects,
    # so the app cache-HITS on this file (no runtime recompute).
    ids = sorted(rm.index.astype(str).tolist())
    has_abstract = "fr_abstract" in rm.columns
    key = "_".join(ids) + f"_abstract={has_abstract}_model={E.MODEL_NAME}"

    texts = [E.build_rule_text(row) for _, row in rm.iterrows()]
    model = E.get_embedder()   # downloads the model on first use; that's fine here

    vecs = []
    for i in range(0, len(texts), a.batch_size):
        chunk = model.encode(
            texts[i:i + a.batch_size],
            show_progress_bar=False,
            batch_size=a.batch_size,
            convert_to_numpy=True,
        )
        vecs.append(np.asarray(chunk, dtype=np.float32))
    emb = np.vstack(vecs)

    out = d / "embeddings_cache.json"
    with open(out, "w") as f:
        json.dump({"key": key, "embeddings": emb.tolist()}, f)

    # sanity: confirm the app would cache-hit on what we just wrote
    hit = (json.load(open(out)).get("key") == key)
    print(f"wrote {out}")
    print(f"  shape={emb.shape}  has_abstract={has_abstract}  cache_hit_key_match={hit}")
    print("Next: commit data/embeddings_cache.json (and the refreshed CSVs) and push.")


if __name__ == "__main__":
    main()
