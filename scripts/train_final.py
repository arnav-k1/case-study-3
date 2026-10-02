"""Train the app's final model on the full data and save lightweight artifacts.

Usage: python scripts/train_final.py [--family implicitmf] [--params '{"embedding_size":64}']
Defaults to the best test-NDCG model in reports/tables/model_comparison.csv.

Training data: ALL users, movies with >= 50 ratings (same candidate universe as
the offline comparison), i.e. almost all 32M ratings.

Artifacts (models/):
- final_scorer.pkl      scorer component only (no training dataset inside)
- app_meta.parquet      movie metadata + popularity for the app
- when_to_watch.pkl     (built by validate_when_to_watch.py / here if missing)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from recsys import data  # noqa: E402
from recsys import when_to_watch as w  # noqa: E402
from recsys.models import SEED, ModelSpec, scorer_of, to_dataset, train  # noqa: E402

MIN_ITEM = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--params")
    args = ap.parse_args()
    if args.family:
        family, params = args.family, json.loads(args.params or "{}")
    else:
        comp = pd.read_csv(data.ROOT / "reports" / "tables" / "model_comparison.csv")
        best = comp.sort_values("ndcg", ascending=False).iloc[0]
        family, params = best["family"], json.loads(best["params"])
    fam = "iknn" if family == "iknn_imp" else family
    print("final model:", family, params, flush=True)

    r = data.load_ratings()
    counts = r["movieId"].value_counts()
    keep = counts.index[counts >= MIN_ITEM]
    r = r[r["movieId"].isin(keep)]
    print(f"training on {len(r):,} ratings, {r.userId.nunique():,} users, {r.movieId.nunique():,} movies", flush=True)
    spec = ModelSpec("final", fam, params)
    pipe, secs = train(spec, to_dataset(r), seed=SEED)
    print(f"trained in {secs:.0f}s", flush=True)
    scorer = scorer_of(pipe)
    if hasattr(scorer, "user_embeddings"):
        scorer.user_embeddings = None  # app always folds in from the query history
        scorer.users = None
    data.MODEL_DIR.mkdir(exist_ok=True)
    with open(data.MODEL_DIR / "final_scorer.pkl", "wb") as f:
        pickle.dump({"scorer": scorer, "family": family, "params": params, "train_ratings": len(r), "train_seconds": secs}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # movie metadata for the app
    movies = data.load_movies()
    summ = pd.read_parquet(data.PROC_DIR / "item_summary.parquet")
    meta = movies.merge(summ, on="movieId", how="left")
    meta["in_model"] = meta["movieId"].isin(keep)
    meta["pop_pct"] = meta["n_ratings"].rank(pct=True)
    meta = meta[["movieId", "title", "clean_title", "year", "genres", "genre_list", "imdbId", "tmdbId", "n_ratings", "mean_rating", "n_users", "pop_pct", "in_model"]]
    meta.to_parquet(data.MODEL_DIR / "app_meta.parquet", index=False)

    if not (data.MODEL_DIR / "when_to_watch.pkl").exists():
        w.save(w.build())
    pd.Series({"family": family, "params": json.dumps(params), "train_ratings": len(r), "train_users": int(r.userId.nunique()),
               "train_movies": int(r.movieId.nunique()), "train_seconds": round(secs, 1)}).to_csv(
        data.ROOT / "reports" / "tables" / "final_model.csv", header=["value"])
    print("saved artifacts", flush=True)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"total {time.time() - t:.0f}s")
