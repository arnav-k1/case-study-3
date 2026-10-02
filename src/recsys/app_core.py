"""UI-independent core of the recommender app (imported by app/app.py and tests).

Scoring calls the trained LensKit scorer component directly with the user's
ratings in the query (fold-in), so new users need no retraining and existing
MovieLens users are handled the same way from their stored history.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lenskit.data import ItemList, RecQuery

from recsys import data
from recsys import when_to_watch as w
from recsys.explain import Explainer

MIN_RATINGS_FOR_RECS = 5


@dataclass
class Filters:
    genres: list[str] = field(default_factory=list)  # any-of
    decade_range: tuple[int, int] = (1900, 2030)
    hidden_gems: float = 0.0  # 0 = off .. 1 = strong popularity down-weighting
    season_boost: bool = True  # re-rank by timing fit for today/context
    min_ratings: int = 50  # candidate floor on number of ratings


@dataclass
class Artifacts:
    scorer: object
    family: str
    meta: pd.DataFrame  # indexed by movieId
    wtw: dict
    explainer: Explainer | None
    candidates: np.ndarray


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    with open(data.MODEL_DIR / "final_scorer.pkl", "rb") as f:
        obj = pickle.load(f)
    meta = pd.read_parquet(data.MODEL_DIR / "app_meta.parquet").set_index("movieId")
    wtw = w.load()
    try:
        explainer = Explainer(obj["scorer"])
    except TypeError:
        explainer = None
    cand = meta.index[meta["in_model"]].to_numpy()
    return Artifacts(obj["scorer"], obj["family"], meta, wtw, explainer, cand)


def search_titles(art: Artifacts, text: str, limit: int = 15) -> pd.DataFrame:
    if not text or len(text.strip()) < 2:
        return art.meta.iloc[0:0]
    t = text.strip().lower()
    m = art.meta
    hit = m[m["title"].str.lower().str.contains(t, regex=False) | m["clean_title"].str.lower().str.contains(t, regex=False)]
    return hit.sort_values("n_ratings", ascending=False).head(limit)


def onboarding_titles(art: Artifacts, per_genre: int = 2, since: int = 1975) -> pd.DataFrame:
    """Popular, diverse titles for the pick-your-favorites grid."""
    m = art.meta[(art.meta["in_model"]) & (art.meta["year"].fillna(0) >= since)]
    genres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    picks: list[int] = []
    for g in genres:
        sub = m[m["genre_list"].map(lambda gl: g in gl)].sort_values("n_users", ascending=False)
        for mid in sub.index:
            if mid not in picks:
                picks.append(int(mid))
                if sum(1 for p in picks if g in m.at[p, "genre_list"]) >= per_genre:
                    break
    top = m.sort_values("n_users", ascending=False).index[:40]
    for mid in top:
        if len(picks) >= 36:
            break
        if int(mid) not in picks:
            picks.append(int(mid))
    return m.loc[picks]


def load_user_history(user_id: int) -> pd.DataFrame:
    """All ratings of one MovieLens user (fast: parquet row-group pruning)."""
    tbl = pq.read_table(data.PROC_DIR / "ratings.parquet", filters=[("userId", "==", int(user_id))])
    return tbl.to_pandas()


def _score(art: Artifacts, history: dict[int, float], cand: np.ndarray) -> np.ndarray:
    hist = ItemList(item_ids=np.fromiter(history.keys(), dtype=np.int64), rating=np.fromiter(history.values(), dtype=np.float32))
    out = art.scorer(query=RecQuery(user_id=None, user_items=hist), items=ItemList(item_ids=cand))
    return np.asarray(out.scores(), dtype=np.float64)


def recommend(
    art: Artifacts,
    history: dict[int, float],
    n: int = 12,
    filters: Filters | None = None,
    today: date | None = None,
    context: str = "any",
    profile: dict | None = None,
) -> pd.DataFrame:
    """Top-n recommendations with explanations and when-to-watch suggestions."""
    filters = filters or Filters()
    today = today or date.today()
    if len(history) == 0:
        raise ValueError("need at least one rating")
    m = art.meta
    cand = art.candidates
    cm = m.loc[cand]
    keep = ~np.isin(cand, np.fromiter(history.keys(), dtype=np.int64))
    keep &= (cm["n_ratings"].fillna(0).to_numpy() >= filters.min_ratings)
    yr = cm["year"].astype("float").fillna(0).to_numpy()
    keep &= (yr >= filters.decade_range[0]) & (yr <= filters.decade_range[1] + 9)
    if filters.genres:
        gs = set(filters.genres)
        keep &= cm["genre_list"].map(lambda gl: bool(gs & set(gl))).to_numpy()
    cand = cand[keep]
    if len(cand) == 0:
        return pd.DataFrame()
    scores = _score(art, history, cand)
    ok = np.isfinite(scores)
    cand, scores = cand[ok], scores[ok]
    if len(cand) == 0:
        return pd.DataFrame()

    # match % = percentile of the raw model score among eligible candidates
    match = pd.Series(scores).rank(pct=True).to_numpy()
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    logpop = np.log1p(m.loc[cand, "n_ratings"].fillna(0).to_numpy())
    zpop = (logpop - logpop.mean()) / (logpop.std() + 1e-9)
    final = z - filters.hidden_gems * zpop

    # shortlist, then attach timing (cheap per-item logic)
    short_n = min(len(cand), max(n * 8, 100))
    idx = np.argsort(-final)[:short_n]
    sug = w.suggest_many(art.wtw, cand[idx], today=today, context=context, profile=profile)
    fit = sug["fit_now"].clip(0.5, 2.0).to_numpy()
    final_short = final[idx] + (np.log(fit) * 2.0 if filters.season_boost else 0.0)
    order = np.argsort(-final_short)[:n]
    sel = idx[order]
    out = sug.iloc[order].reset_index(drop=True)
    out.insert(0, "match", match[sel])
    # fit relative to the model's own score for films this user loved (>= 4 stars)
    liked = np.array([m for m, r in history.items() if r >= 4.0 and m in art.meta.index and art.meta.at[m, "in_model"]], dtype=np.int64)
    if len(liked) == 0:
        liked = np.array([m for m in history if m in art.meta.index and art.meta.at[m, "in_model"]], dtype=np.int64)
    ref = float(np.median(_score(art, history, liked))) if len(liked) else float(np.max(scores))
    rel = scores[sel] / max(ref, 1e-6)
    out.insert(0, "fit_label", [fit_label(x) for x in rel])
    out.insert(0, "fit", np.clip(rel, 0.0, 1.0))
    out.insert(0, "score", scores[sel])
    meta_cols = m.loc[cand[sel], ["title", "clean_title", "year", "genre_list", "n_ratings", "mean_rating", "pop_pct"]].reset_index(drop=True)
    out = pd.concat([meta_cols, out], axis=1)
    out["because"] = [explain_one(art, int(mid), history) for mid in out["movieId"]]
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def fit_label(rel: float) -> str:
    """Relative fit: model score divided by its median score for films the user loved."""
    if rel >= 0.9:
        return "Excellent"
    if rel >= 0.7:
        return "Strong"
    if rel >= 0.5:
        return "Good"
    return "Worth a look"


def explain_one(art: Artifacts, movie_id: int, history: dict[int, float]) -> str | None:
    if art.explainer is None:
        return None
    b = art.explainer.because(movie_id, history, top=1)
    if not b:
        return None
    anchor, sim = b[0]
    t = art.meta.at[anchor, "clean_title"] if anchor in art.meta.index else str(anchor)
    y = art.meta.at[anchor, "year"] if anchor in art.meta.index else None
    return f"{t} ({int(y)})" if y is not None and not pd.isna(y) else t
