"""Verification tests ("did we build it right?").

Fast tests use toy data. Tests marked `data` need data/processed (run the
pipeline first); tests marked `artifacts` need models/ from train_final.py.
Missing inputs skip rather than fail.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from recsys import data
from recsys import evaluate as ev
from recsys.models import ModelSpec, recommend_for_history, to_dataset, train

HAS_DATA = (data.PROC_DIR / "ratings.parquet").exists()
HAS_ART = (data.MODEL_DIR / "final_scorer.pkl").exists() and (data.MODEL_DIR / "when_to_watch.pkl").exists()


# --------------------------------------------------------------------------- toy fixtures
@pytest.fixture(scope="module")
def toy() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 81):
        items = rng.choice(np.arange(1, 61), size=20, replace=False)
        for k, i in enumerate(items):
            rows.append((u, int(i), float(rng.integers(1, 11)) / 2, 1_000_000_000 + u * 1000 + k))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


# --------------------------------------------------------------------------- metrics (hand-computed)
def test_ndcg_hand_computed():
    # relevant = {2, 4}; list = 1 2 3 4. LensKit / Jarvelin-Kekalainen discount 1/log2(max(rank,2)):
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1 = 2 -> 0.75
    assert ev.ndcg_at_k([1, 2, 3, 4], {2, 4}, k=10) == pytest.approx(0.75)
    # textbook 1/log2(rank+1) variant: (1/log2 3 + 1/log2 5) / (1 + 1/log2 3)
    want = (1 / np.log2(3) + 1 / np.log2(5)) / (1 + 1 / np.log2(3))
    assert ev.ndcg_at_k([1, 2, 3, 4], {2, 4}, clip=False) == pytest.approx(want)
    assert ev.ndcg_at_k([2, 4, 1], {2, 4}) == pytest.approx(1.0)
    assert ev.ndcg_at_k([1, 3], {2}) == 0.0


def test_lenskit_ndcg_matches_reference():
    from lenskit.data import ItemList
    from lenskit.metrics import NDCG, call_metric

    recs = ItemList(item_ids=[1, 2, 3, 4], ordered=True)
    truth = ItemList(item_ids=[2, 4])
    assert call_metric(NDCG(10), recs, truth) == pytest.approx(ev.ndcg_at_k([1, 2, 3, 4], {2, 4}))


def test_intra_list_diversity_hand_computed():
    g = {1: frozenset({"A", "B"}), 2: frozenset({"B"}), 3: frozenset({"C"})}
    # pairs: (1,2) J=1/2 -> d=.5 ; (1,3) d=1 ; (2,3) d=1  -> mean 2.5/3
    assert ev.intra_list_diversity([1, 2, 3], g) == pytest.approx(2.5 / 3)


def test_bootstrap_ci_contains_mean():
    v = np.arange(100, dtype=float)
    m, lo, hi = ev.bootstrap_ci(v)
    assert lo < m < hi and m == pytest.approx(49.5)


# --------------------------------------------------------------------------- split
def test_temporal_holdout_takes_latest(toy):
    sp = ev.temporal_user_holdout(toy, n_test_users=10, n_holdout=5, seed=1)
    assert len(sp.test) == 50
    for u, g in sp.test.groupby("userId"):
        train_u = sp.train[sp.train.userId == u]
        assert g.timestamp.min() > train_u.timestamp.max()
    assert len(sp.train) + len(sp.test) == len(toy)


# --------------------------------------------------------------------------- models
@pytest.mark.parametrize("family,params", [("iknn", {"feedback": "implicit"}), ("implicitmf", {"embedding_size": 8, "epochs": 3}), ("biasedmf", {"embedding_size": 8, "epochs": 3})])
def test_recs_exclude_rated_and_cold_start(toy, family, params):
    pipe, _ = train(ModelSpec("m", family, params), to_dataset(toy))
    from lenskit import recommend

    for u in [1, 2, 3]:
        recs = recommend(pipe, u, n=10)
        seen = set(toy.loc[toy.userId == u, "movieId"])
        assert not (set(recs.ids()) & seen)
    hist = {1: 5.0, 2: 4.5, 3: 1.0, 4: 4.0, 5: 3.0}
    cold = recommend_for_history(pipe, hist, n=10)
    assert len(cold) == 10 and not (set(cold.movieId) & set(hist))


def test_reproducible_with_seed(toy):
    spec = ModelSpec("m", "implicitmf", {"embedding_size": 8, "epochs": 3})
    p1, _ = train(spec, to_dataset(toy), seed=7)
    p2, _ = train(spec, to_dataset(toy), seed=7)
    from lenskit import recommend

    r1, r2 = recommend(p1, 5, n=10), recommend(p2, 5, n=10)
    assert list(r1.ids()) == list(r2.ids())
    np.testing.assert_allclose(r1.scores(), r2.scores(), rtol=1e-5)


def test_build_sample_seeded(toy):
    a = data.build_sample(toy, n_users=20, min_item_ratings=1, min_user_ratings=5, seed=3)
    b = data.build_sample(toy, n_users=20, min_item_ratings=1, min_user_ratings=5, seed=3)
    assert a.equals(b) and a.userId.nunique() == 20


def test_parse_title():
    assert data.parse_title("Matrix, The (1999)") == ("The Matrix", 1999.0)
    name, yr = data.parse_title("Untitled")
    assert name == "Untitled" and np.isnan(yr)


# --------------------------------------------------------------------------- full data
@pytest.mark.skipif(not HAS_DATA, reason="processed data not built")
def test_loader_row_counts_and_no_duplicates():
    r = data.load_ratings(columns=["userId", "movieId"])
    assert len(r) == data.EXPECTED_ROWS["ratings"]
    assert not r.duplicated().any()
    assert len(data.load_movies()) == data.EXPECTED_ROWS["movies"]
    assert len(data.load_tags()) == data.EXPECTED_ROWS["tags"]


# --------------------------------------------------------------------------- when-to-watch + app
@pytest.mark.skipif(not HAS_ART, reason="model artifacts not built")
def test_when_to_watch_valid_for_every_recommendation():
    from recsys import app_core as core
    from recsys import when_to_watch as w

    art = core.load_artifacts()
    hist = {int(m): 5.0 for m in art.meta.sort_values("n_users", ascending=False).index[:5]}
    recs = core.recommend(art, hist, n=24, today=date(2026, 9, 26))
    assert len(recs) == 24
    for _, r in recs.iterrows():
        assert r["action"] in {"now", "save"}
        assert isinstance(r["headline"], str) and r["headline"]
        assert isinstance(r["reason"], str) and r["reason"]
        assert r["when"] == "Now" or r["when"] in w.MONTH_NAMES
        assert np.isfinite(r["fit_now"]) and r["fit_now"] > 0
    # also for arbitrary catalog items, including one with no ratings history
    for mid in list(art.meta.index[:50]) + [999_999_999]:
        s = w.suggest(art.wtw, int(mid), date(2026, 1, 15))
        assert s.action in {"now", "save"} and s.reason


@pytest.mark.skipif(not HAS_ART, reason="model artifacts not built")
def test_when_to_watch_intuition():
    from recsys import when_to_watch as w

    m = w.load()
    hol = m["holiday"]
    xmas = hol.index[hol.holiday == "christmas"][:20]
    for mid in xmas:
        s = w.suggest(m, int(mid), date(2026, 6, 1))
        assert s.action == "save" and s.when == "December"
        s = w.suggest(m, int(mid), date(2026, 12, 10))
        assert s.action == "now"


@pytest.mark.skipif(not HAS_ART, reason="model artifacts not built")
def test_app_cold_start_flow():
    from recsys import app_core as core

    art = core.load_artifacts()
    grid = core.onboarding_titles(art)
    assert len(grid) >= 20
    hist = {int(m): 5.0 for m in grid.index[:core.MIN_RATINGS_FOR_RECS]}
    recs = core.recommend(art, hist, n=12, filters=core.Filters(genres=["Comedy"], decade_range=(1990, 2010), hidden_gems=0.5))
    assert len(recs) == 12
    assert not (set(recs.movieId) & set(hist))
    assert recs.genre_list.map(lambda g: "Comedy" in g).all()
    assert recs.year.between(1990, 2019).all()
    assert len(core.search_titles(art, "toy story")) >= 1
