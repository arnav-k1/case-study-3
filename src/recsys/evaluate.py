"""Temporal holdout splitting and top-N / rating-prediction evaluation.

Ranking metrics (NDCG, Recall, Precision, RecipRank) come from LensKit's
`lenskit.metrics`. Beyond-accuracy metrics (coverage, novelty/popularity,
intra-list genre diversity) and the bootstrap CI are implemented here because
LensKit has no genre-aware diversity metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lenskit import predict as lk_predict
from lenskit.batch import recommend as batch_recommend
from lenskit.data import ItemList, ItemListCollection
from lenskit.metrics import NDCG, Precision, Recall, RecipRank, RunAnalysis

REL_THRESHOLD = 4.0  # a held-out rating >= 4 stars counts as a relevant item
K = 10


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame  # all held-out ratings of test users


def temporal_user_holdout(
    ratings: pd.DataFrame,
    n_test_users: int,
    n_holdout: int = 5,
    seed: int = 42,
    exclude_users: np.ndarray | None = None,
) -> Split:
    """Sample `n_test_users`; hold out each one's `n_holdout` most recent ratings.

    Ties on timestamp are broken by movieId so the split is deterministic.
    All other ratings (other users, and test users' earlier ratings) train.
    """
    users = np.sort(ratings["userId"].unique())
    if exclude_users is not None:
        users = np.setdiff1d(users, exclude_users)
    rng = np.random.default_rng(seed)
    test_users = rng.choice(users, size=min(n_test_users, len(users)), replace=False)
    is_test_user = ratings["userId"].isin(test_users)
    tu = ratings[is_test_user].sort_values(["userId", "timestamp", "movieId"])
    rank_from_end = tu.groupby("userId").cumcount(ascending=False)
    test_idx = tu.index[rank_from_end < n_holdout]
    test = ratings.loc[test_idx].reset_index(drop=True)
    train = ratings.drop(index=test_idx).reset_index(drop=True)
    return Split(train=train, test=test)


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics
# --------------------------------------------------------------------------- #
def intra_list_diversity(item_ids: list[int], genre_sets: dict[int, frozenset]) -> float:
    """Mean pairwise Jaccard distance between genre sets of the listed items."""
    sets = [genre_sets.get(int(i), frozenset()) for i in item_ids]
    n = len(sets)
    if n < 2:
        return np.nan
    tot, cnt = 0.0, 0
    for a in range(n):
        for b in range(a + 1, n):
            u = sets[a] | sets[b]
            j = len(sets[a] & sets[b]) / len(u) if u else 1.0
            tot += 1.0 - j
            cnt += 1
    return tot / cnt


def ndcg_at_k(rec_ids: list[int], relevant: set[int], k: int = K, clip: bool = True) -> float:
    """Binary-relevance NDCG@k (reference implementation used in tests).

    clip=True matches LensKit and Jarvelin & Kekalainen (2002): the discount is
    1 / log2(max(rank, 2)), so ranks 1 and 2 are undiscounted. clip=False gives the
    common 1 / log2(rank + 1) variant.
    """
    def disc(rank: int) -> float:
        return 1.0 / np.log2(max(rank, 2)) if clip else 1.0 / np.log2(rank + 1)

    gains = [1.0 if i in relevant else 0.0 for i in rec_ids[:k]]
    dcg = sum(g * disc(r + 1) for r, g in enumerate(gains))
    ideal = sum(disc(r + 1) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else np.nan


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over users (rows)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# --------------------------------------------------------------------------- #
# Evaluation of a trained pipeline
# --------------------------------------------------------------------------- #
def _truth_collection(test: pd.DataFrame) -> ItemListCollection:
    rel = test[test["rating"] >= REL_THRESHOLD]
    df = rel.rename(columns={"userId": "user_id", "movieId": "item_id"})[["user_id", "item_id", "rating"]]
    return ItemListCollection.from_df(df, "user_id")


def evaluate_pipeline(
    pipe,
    test: pd.DataFrame,
    item_pop: pd.Series,
    genre_sets: dict[int, frozenset],
    n_catalog: int,
    predicts_ratings: bool = False,
    n_jobs: int | None = 1,  # Windows spawn pool is far slower than sequential here (D10)
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (summary dict, per-user metric frame, recommendations frame).

    item_pop: fraction of training users who rated each movie (indexed by movieId).
    """
    test_users = np.sort(test["userId"].unique())
    recs = batch_recommend(pipe, test_users, n=K, n_jobs=n_jobs)
    rec_df = recs.to_df().rename(columns={"user_id": "userId", "item_id": "movieId"})

    truth = _truth_collection(test)
    ra = RunAnalysis()
    ra.add_metric(NDCG(K))
    ra.add_metric(Recall(K))
    ra.add_metric(Precision(K))
    ra.add_metric(RecipRank(K))
    res = ra.measure(recs, truth)
    per_user = res.list_metrics().reset_index()
    per_user = per_user.rename(columns={"user_id": "userId"})
    # LensKit scores lists for users that have no relevant items as NaN/0;
    # restrict ranking metrics to users with >= 1 relevant held-out item.
    has_rel = set(test.loc[test["rating"] >= REL_THRESHOLD, "userId"].unique())
    per_user = per_user[per_user["userId"].isin(has_rel)]

    # beyond-accuracy
    grouped = rec_df.sort_values(["userId", "rank"]).groupby("userId")["movieId"].apply(list)
    ild = grouped.map(lambda ids: intra_list_diversity(ids, genre_sets))
    pop = grouped.map(lambda ids: float(np.mean([item_pop.get(i, 0.0) for i in ids])))
    beyond = pd.DataFrame({"ild": ild, "avg_pop": pop}).reset_index()
    per_user = per_user.merge(beyond, on="userId", how="outer")

    summary: dict = {}
    for col, name in [("NDCG", "ndcg"), ("Recall", "recall"), ("Precision", "precision"), ("RecipRank", "mrr"), ("ild", "ild"), ("avg_pop", "avg_pop")]:
        colname = next((c for c in per_user.columns if c.lower().startswith(col.lower())), col)
        m, lo, hi = bootstrap_ci(per_user[colname].to_numpy())
        summary[name] = m
        summary[f"{name}_lo"] = lo
        summary[f"{name}_hi"] = hi
    summary["coverage"] = rec_df["movieId"].nunique() / n_catalog
    summary["n_users_ranked"] = int(per_user["NDCG" if "NDCG" in per_user else per_user.columns[1]].notna().sum())
    summary["n_test_users"] = int(len(test_users))

    if predicts_ratings:
        # Per-user loop instead of lenskit.batch.predict: in lenskit 2025.8.1 the
        # batch collector rejects item-kNN outputs whose `nbr_counts` field dtype
        # differs between users (double vs int32).
        tdf = test.rename(columns={"userId": "user_id", "movieId": "item_id"})[["user_id", "item_id", "rating"]]
        rows = []
        for uid, grp in tdf.groupby("user_id"):
            out = lk_predict(pipe, int(uid), ItemList(item_ids=grp["item_id"].to_numpy()))
            rows.append(pd.DataFrame({"user_id": uid, "item_id": out.ids(), "score": out.scores()}))
        preds = pd.concat(rows, ignore_index=True).merge(tdf, on=["user_id", "item_id"])
        err = preds["score"] - preds["rating"]
        ok = err.notna()
        summary["rmse"] = float(np.sqrt((err[ok] ** 2).mean()))
        summary["mae"] = float(err[ok].abs().mean())
        summary["pred_coverage"] = float(ok.mean())
        # CI over users for RMSE: bootstrap per-user squared error means
        se_user = (err[ok] ** 2).groupby(preds.loc[ok, "user_id"]).mean().to_numpy()
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(se_user), size=(1000, len(se_user)))
        rm = np.sqrt(se_user[idx].mean(axis=1))
        summary["rmse_lo"], summary["rmse_hi"] = float(np.quantile(rm, 0.025)), float(np.quantile(rm, 0.975))
    return summary, per_user, rec_df
