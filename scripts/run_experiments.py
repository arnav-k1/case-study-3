"""Model comparison: tune on validation users, evaluate best-per-family on test users.

Sample: 25,000 seeded users; movies with >= 50 ratings in the full data.
Validation: 3,000 users (last-5 holdout) drawn from non-test users; models trained
on everything else. Test: 5,000 different users (last-5 holdout).

Outputs: reports/tables/experiments.csv, model_comparison.csv, per_user_test.parquet,
         reports/figures/model_*.png
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from recsys import data  # noqa: E402
from recsys import evaluate as ev  # noqa: E402
from recsys.models import SEED, ModelSpec, to_dataset, train  # noqa: E402

TABLES = data.ROOT / "reports" / "tables"
N_USERS, MIN_ITEM, N_TEST, N_VAL = 25_000, 50, 5_000, 3_000

FAMILY_LABEL = {
    "pop": "Most popular",
    "bias": "Bias baseline",
    "iknn": "Item k-NN (explicit)",
    "iknn_imp": "Item k-NN (implicit)",
    "uknn": "User k-NN (implicit)",
    "biasedmf": "Biased MF (ALS, explicit)",
    "implicitmf": "Implicit MF (ALS)",
}


def grid() -> list[tuple[str, ModelSpec]]:
    g: list[tuple[str, ModelSpec]] = [("pop", ModelSpec("pop", "pop"))]
    for d in [0.0, 5.0, 25.0]:
        g.append(("bias", ModelSpec("bias", "bias", {"damping": d}, predicts_ratings=True)))
    for k in [20, 50]:
        for mn in [1, 5]:
            g.append(("iknn", ModelSpec("iknn", "iknn", {"max_nbrs": k, "min_nbrs": mn, "feedback": "explicit"}, predicts_ratings=True)))
    for k in [20, 50]:
        g.append(("iknn_imp", ModelSpec("iknn_imp", "iknn", {"max_nbrs": k, "feedback": "implicit"})))
    for k in [30, 60]:
        g.append(("uknn", ModelSpec("uknn", "uknn", {"max_nbrs": k, "feedback": "implicit"})))
    for f in [32, 64, 128]:
        for reg in [0.1, 1.0]:
            g.append(("biasedmf", ModelSpec("biasedmf", "biasedmf", {"embedding_size": f, "regularization": reg, "epochs": 10}, predicts_ratings=True)))
    for f in [32, 64, 128]:
        for wgt in [10.0, 40.0]:
            g.append(("implicitmf", ModelSpec("implicitmf", "implicitmf", {"embedding_size": f, "weight": wgt, "regularization": 0.1, "epochs": 10})))
    return g


def run_one(phase: str, family: str, spec: ModelSpec, split: ev.Split, ctx: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ds = to_dataset(split.train)
    pipe, t_train = train(spec, ds, seed=SEED)
    t0 = time.perf_counter()
    item_pop = split.train.groupby("movieId")["userId"].nunique() / split.train["userId"].nunique()
    summ, per_user, recs = ev.evaluate_pipeline(pipe, split.test, item_pop, ctx["genre_sets"], ctx["n_catalog"], predicts_ratings=spec.predicts_ratings)
    row = {"phase": phase, "family": family, "model": FAMILY_LABEL[family], "params": json.dumps(spec.params), "train_s": round(t_train, 2), "eval_s": round(time.perf_counter() - t0, 2), **summ}
    print(f"[{phase}] {family:10s} {spec.params} ndcg={summ['ndcg']:.4f} recall={summ['recall']:.4f} cov={summ['coverage']:.3f} rmse={summ.get('rmse', float('nan')):.4f} ({t_train:.0f}s+{row['eval_s']:.0f}s)", flush=True)
    return row, per_user, recs


def main() -> None:
    t_start = time.perf_counter()
    ratings = data.load_ratings()
    movies = data.load_movies()
    sample = data.build_sample(ratings, n_users=N_USERS, min_item_ratings=MIN_ITEM, seed=SEED)
    del ratings
    ctx = {
        "genre_sets": {int(m): frozenset(g) for m, g in zip(movies.movieId, movies.genre_list)},
        "n_catalog": int(sample["movieId"].nunique()),
    }
    info = {"sample_ratings": len(sample), "sample_users": int(sample.userId.nunique()), "sample_movies": ctx["n_catalog"]}
    print(info, flush=True)

    test_split = ev.temporal_user_holdout(sample, N_TEST, n_holdout=5, seed=SEED)
    test_users = test_split.test["userId"].unique()
    # validation: different users, holdout taken from the TEST split's training data
    val_split = ev.temporal_user_holdout(test_split.train, N_VAL, n_holdout=5, seed=SEED + 1, exclude_users=test_users)
    info.update({"test_users": len(test_users), "val_users": int(val_split.test.userId.nunique()),
                 "test_ratings_heldout": len(test_split.test), "test_relevant_share": float((test_split.test.rating >= ev.REL_THRESHOLD).mean())})
    pd.Series(info).to_csv(TABLES / "experiment_setup.csv", header=["value"])

    # ---------------------------------------------------------------- tuning
    rows = []
    for family, spec in grid():
        row, _, _ = run_one("validation", family, spec, val_split, ctx)
        rows.append(row)
        pd.DataFrame(rows).to_csv(TABLES / "experiments.csv", index=False)
    val = pd.DataFrame(rows)
    best = val.loc[val.groupby("family")["ndcg"].idxmax()]
    print("best per family:\n", best[["family", "params", "ndcg"]], flush=True)

    # ---------------------------------------------------------------- test
    per_user_all, recs_all = [], []
    for _, b in best.iterrows():
        spec = next(s for f, s in grid() if f == b.family and json.dumps(s.params) == b.params)
        row, per_user, recs = run_one("test", b.family, spec, test_split, ctx)
        rows.append(row)
        per_user_all.append(per_user.assign(family=b.family))
        recs_all.append(recs.assign(family=b.family))
        pd.DataFrame(rows).to_csv(TABLES / "experiments.csv", index=False)
    exp = pd.DataFrame(rows)
    test = exp[exp.phase == "test"].copy()
    per_user = pd.concat(per_user_all, ignore_index=True)
    per_user.to_parquet(TABLES / "per_user_test.parquet", index=False)
    pd.concat(recs_all, ignore_index=True).to_parquet(data.PROC_DIR / "test_recs.parquet", index=False)

    # paired bootstrap: each model minus most-popular, NDCG@10, over common users
    pv = per_user.pivot_table(index="userId", columns="family", values="NDCG@10")
    rng = np.random.default_rng(SEED)
    diffs = {}
    for fam in pv.columns:
        d = (pv[fam] - pv["pop"]).dropna().to_numpy()
        idx = rng.integers(0, len(d), size=(1000, len(d)))
        bm = d[idx].mean(axis=1)
        diffs[fam] = (float(d.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)))
    test["ndcg_minus_pop"] = test.family.map(lambda f: diffs[f][0])
    test["ndcg_minus_pop_lo"] = test.family.map(lambda f: diffs[f][1])
    test["ndcg_minus_pop_hi"] = test.family.map(lambda f: diffs[f][2])
    test = test.sort_values("ndcg", ascending=False)
    test.to_csv(TABLES / "model_comparison.csv", index=False)
    print(test[["model", "params", "ndcg", "ndcg_lo", "ndcg_hi", "recall", "precision", "mrr", "coverage", "avg_pop", "ild", "rmse"]].to_string(), flush=True)
    print(f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
