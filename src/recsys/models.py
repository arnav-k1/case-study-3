"""Model definitions (LensKit 2025.x pipelines), training, persistence, and scoring."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from lenskit import Pipeline, recommend, topn_pipeline
from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.basic import BiasScorer, PopScorer
from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df
from lenskit.knn import ItemKNNScorer, UserKNNScorer
from lenskit.training import TrainingOptions

from recsys.data import MODEL_DIR

SEED = 42


@dataclass
class ModelSpec:
    """A named model family plus hyperparameters."""

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    predicts_ratings: bool = False

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return self.name + "(" + ", ".join(f"{k}={v}" for k, v in self.params.items()) + ")"


def make_scorer(spec: ModelSpec):
    p = spec.params
    match spec.family:
        case "pop":
            return PopScorer()
        case "bias":
            return BiasScorer(damping=p.get("damping", 5.0))
        case "iknn":
            return ItemKNNScorer(
                max_nbrs=p.get("max_nbrs", 20),
                min_nbrs=p.get("min_nbrs", 1),
                min_sim=p.get("min_sim", 1e-6),
                save_nbrs=p.get("save_nbrs"),
                feedback=p.get("feedback", "explicit"),
            )
        case "uknn":
            return UserKNNScorer(
                max_nbrs=p.get("max_nbrs", 30),
                min_nbrs=p.get("min_nbrs", 1),
                min_sim=p.get("min_sim", 1e-6),
                feedback=p.get("feedback", "explicit"),
            )
        case "biasedmf":
            return BiasedMFScorer(
                embedding_size=p.get("embedding_size", 64),
                epochs=p.get("epochs", 10),
                regularization=p.get("regularization", 0.1),
                damping=p.get("damping", 5.0),
            )
        case "implicitmf":
            return ImplicitMFScorer(
                embedding_size=p.get("embedding_size", 64),
                epochs=p.get("epochs", 10),
                regularization=p.get("regularization", 0.1),
                weight=p.get("weight", 40.0),
                use_ratings=p.get("use_ratings", False),
            )
    raise ValueError(f"unknown family {spec.family}")


def to_dataset(ratings: pd.DataFrame) -> Dataset:
    """Build a LensKit Dataset from a MovieLens-style frame."""
    df = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    cols = [c for c in ["user_id", "item_id", "rating", "timestamp"] if c in df.columns]
    return from_interactions_df(df[cols])


def build_pipeline(spec: ModelSpec, n: int = 10) -> Pipeline:
    return topn_pipeline(make_scorer(spec), predicts_ratings=spec.predicts_ratings, n=n, name=spec.name)


def train(spec: ModelSpec, data: Dataset, n: int = 10, seed: int = SEED) -> tuple[Pipeline, float]:
    pipe = build_pipeline(spec, n=n)
    t0 = time.perf_counter()
    pipe.train(data, TrainingOptions(rng=seed))
    return pipe, time.perf_counter() - t0


def save_pipeline(pipe: Pipeline, name: str, meta: dict | None = None) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"pipeline": pipe, "meta": meta or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pipeline(name: str) -> tuple[Pipeline, dict]:
    with open(MODEL_DIR / f"{name}.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["meta"]


def recommend_for_history(
    pipe: Pipeline,
    history: dict[int, float],
    n: int = 10,
    candidates: list[int] | None = None,
) -> pd.DataFrame:
    """Top-n recommendations for a (possibly new) user given {movieId: rating}.

    Works without retraining: the query carries the user's ratings and the
    scorer folds them in (item k-NN neighbours / ALS user-embedding solve).
    Rated movies are excluded by the pipeline's candidate selector; `candidates`
    is further filtered here as a safeguard.
    """
    hist = ItemList(item_ids=np.array(list(history.keys()), dtype=np.int64), rating=np.array(list(history.values()), dtype=np.float32))
    query = RecQuery(user_id=None, user_items=hist)
    if candidates is not None:
        cand = [c for c in candidates if c not in history]
        recs = recommend(pipe, query, n=n, items=np.array(cand, dtype=np.int64))
    else:
        recs = recommend(pipe, query, n=n)
    df = recs.to_df()
    df = df.rename(columns={"item_id": "movieId"})
    df = df[~df["movieId"].isin(history)]  # safeguard
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["movieId", "score", "rank"]].reset_index(drop=True)


def recommend_for_user(pipe: Pipeline, user_id: int, n: int = 10) -> pd.DataFrame:
    recs = recommend(pipe, int(user_id), n=n)
    df = recs.to_df().rename(columns={"item_id": "movieId"})
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["movieId", "score", "rank"]].reset_index(drop=True)


def scorer_of(pipe: Pipeline):
    return pipe.component("scorer")
