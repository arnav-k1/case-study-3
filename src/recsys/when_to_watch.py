"""When should you watch it? Context-aware viewing suggestions.

A contextual *post-filter* (Adomavicius & Tuzhilin, 2011): the collaborative
filtering model decides WHAT to recommend; this module attaches WHEN, with a
one-line reason and the evidence behind it.

Signals (all derived from MovieLens 32M; no external data):
1. Monthly seasonality per movie: lift = share of the movie's ratings that fall
   in month m ÷ share of ALL ratings that fall in month m. Movie lifts are
   shrunk toward the average lift of the movie's genres (empirical-Bayes style)
   so thinly rated movies do not get extreme values.
2. Holiday tags ("christmas", "halloween", ...) applied by >= 2 distinct users
   map a movie to a calendar window.
3. Weekly rhythm: weekend (Sat-Sun) share of the movie's/genre's ratings vs.
   the overall weekend share.
4. Personal rhythm: when a known user has enough time-stamped history, the
   weekend share of their own ratings in the movie's genres.
5. Length proxy: movies that >= 2 users tagged "long", "too long", ... are
   flagged for the "short on time" context (runtime is not in MovieLens).

Only "clean" ratings feed signals 1 and 3: ratings entered > 24 h after the
user's first rating (not onboarding backfill) and >= 2 calendar years after the
film's release (no release-date spike).

KEY LIMITATION: a rating timestamp is when someone RATED a movie, not
necessarily when they WATCHED it (Harper & Konstan, 2015). Returning-session
ratings are the closest available proxy for recent viewing.
"""
from __future__ import annotations

import calendar
import pickle
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from recsys import data

MONTH_NAMES = list(calendar.month_name)  # index 1..12
WEEKEND = (5, 6)  # Sat, Sun (Monday = 0), US Central

# Holiday tags -> calendar window. `anchor` is the key date used for "in N days".
HOLIDAYS: dict[str, dict] = {
    "christmas": {"tags": ["christmas", "christmas movie", "christmas eve", "xmas", "xmas theme", "santa claus"], "months": [12], "anchor": (12, 25), "label": "Christmas season"},
    "halloween": {"tags": ["halloween", "scary movies to see on halloween"], "months": [10], "anchor": (10, 31), "label": "Halloween"},
    "thanksgiving": {"tags": ["thanksgiving"], "months": [11], "anchor": (11, 26), "label": "Thanksgiving"},
    "new_year": {"tags": ["new year's eve", "new year's", "new years eve"], "months": [12, 1], "anchor": (12, 31), "label": "New Year's Eve"},
    "valentine": {"tags": ["valentine's day", "valentines day", "valentine"], "months": [2], "anchor": (2, 14), "label": "Valentine's Day"},
    "summer": {"tags": ["summer", "summer vacation", "summer camp", "summer romance"], "months": [6, 7, 8], "anchor": (6, 21), "label": "summer"},
    "winter": {"tags": ["winter", "snow"], "months": [12, 1, 2], "anchor": (12, 21), "label": "winter"},
}
LONG_TAGS = ["too long", "long", "overlong", "long movie", "way too long", "far too long", "too long!"]

MIN_TAG_USERS = 2  # a tag must be applied by >= 2 distinct users to count
PRIOR_STRENGTH = 100.0  # pseudo-ratings pulling a movie's lift toward its genre prior
SEASONAL_MIN_LIFT = 1.15
SEASONAL_MIN_Z = 3.0
SEASONAL_MIN_PEAK_N = 20
RECUR_MIN_YEAR_N = 24  # a year counts toward recurrence if it has >= 24 clean ratings (2/month)
RECUR_MIN_YEARS = 5
RECUR_MIN_SHARE = 0.70
HOLIDAY_MIN_WINDOW_LIFT = 1.10  # holiday tag groups must show this lift in their window to be used
GENRE_HINT_LIFT = 1.08  # genre-level month lean worth mentioning
# Out-of-time validation (scripts/validate_when_to_watch.py) showed that
# movie-level peaks only replicate when they fall in a validated holiday month
# (Oct, Dec) or recur in >= 90% of years; other peaks regress to ~1.0.
RECUR_STRONG = 0.90
WEEKEND_BEST = 1.08  # weekend lift above this => "best on a weekend"
WEEKNIGHT_OK = 0.97  # weekend lift below this => "fine for a weeknight"
LEAD_DAYS = 21  # within this many days of a holiday anchor, "watch now"


@dataclass
class Suggestion:
    movieId: int
    action: str  # "now" | "save"
    when: str  # short label, e.g. "Now", "December", "Weekend"
    headline: str  # e.g. "Save for December"
    reason: str  # one-line human-readable evidence
    weekly: str  # weekend/weeknight hint
    fit_now: float  # multiplicative timing fit for re-ranking (1 = neutral)
    peak_month: int | None
    peak_lift: float | None
    evidence_n: int  # clean ratings behind the seasonal estimate
    holiday: str | None
    long_flag: bool
    personal: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Model building
# --------------------------------------------------------------------------- #
def _genre_table(movies: pd.DataFrame) -> pd.DataFrame:
    mg = movies[["movieId", "genre_list"]].explode("genre_list").dropna()
    return mg.rename(columns={"genre_list": "genre"})


def build(max_year: int | None = None, min_year: int | None = None) -> dict:
    """Build the when-to-watch model from the precomputed clean aggregates.

    `max_year`/`min_year` restrict the rating years used (for out-of-time
    validation: fit on < 2019, test on >= 2019).
    """
    movies = data.load_movies()
    mm = pd.read_parquet(data.PROC_DIR / "agg_movie_month_clean.parquet")
    if max_year is not None:
        mm = mm[mm["year_rated"] <= max_year]
    if min_year is not None:
        mm = mm[mm["year_rated"] >= min_year]
    obs = mm.groupby(["movieId", "month"])["n"].sum().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    p_month = obs.sum(axis=0) / obs.values.sum()  # P(m) over clean ratings

    # genre-level lifts
    mg = _genre_table(movies)
    g_obs = obs.reset_index().merge(mg, on="movieId").drop(columns="movieId").groupby("genre").sum()
    g_lift = (g_obs.div(g_obs.sum(axis=1), axis=0)) / p_month
    prior = mg[mg["movieId"].isin(obs.index)].merge(g_lift, left_on="genre", right_index=True).groupby("movieId")[list(range(1, 13))].mean()
    prior = prior.reindex(obs.index).fillna(1.0)

    n_tot = obs.sum(axis=1)
    expected = np.outer(n_tot, p_month)
    pseudo = PRIOR_STRENGTH * p_month.to_numpy()[None, :]
    lift = (obs.to_numpy() + pseudo * prior.to_numpy()) / (expected + pseudo)
    lift = pd.DataFrame(lift, index=obs.index, columns=range(1, 13))
    raw_lift = pd.DataFrame(np.where(expected > 0, obs.to_numpy() / np.maximum(expected, 1e-9), np.nan), index=obs.index, columns=range(1, 13))

    peak = lift.idxmax(axis=1)
    pk = peak.to_numpy() - 1
    rows = np.arange(len(peak))
    o_pk = obs.to_numpy()[rows, pk]
    e_pk = expected[rows, pk]
    z = (o_pk - e_pk) / np.sqrt(np.maximum(e_pk, 1e-9))
    meta = pd.DataFrame({
        "n_clean": n_tot.astype(int),
        "peak_month": peak.astype(int),
        "peak_lift": lift.to_numpy()[rows, pk],
        "peak_obs": o_pk.astype(int),
        "peak_z": z,
    }, index=obs.index)
    # Recurrence: a real season repeats. For each movie, look at every year with
    # >= RECUR_MIN_YEAR_N clean ratings and check whether the peak month's share
    # exceeded that year's overall share for the month. One-off spikes (a site
    # feature, a TV airing) fail this test even with a huge z-score, because
    # ratings are clustered in time and the Poisson z is overconfident.
    ym = mm.groupby(["movieId", "year_rated", "month"])["n"].sum().reset_index()
    p_ym = ym.groupby(["year_rated", "month"])["n"].sum()
    p_ym = (p_ym / p_ym.groupby(level=0).transform("sum")).rename("p_ym")
    n_y = ym.groupby(["movieId", "year_rated"])["n"].sum().rename("n_y").reset_index()
    n_y = n_y[n_y["n_y"] >= RECUR_MIN_YEAR_N]
    n_y["month"] = n_y["movieId"].map(meta["peak_month"])
    n_y = n_y.merge(ym, on=["movieId", "year_rated", "month"], how="left").fillna({"n": 0})
    n_y = n_y.merge(p_ym.reset_index(), on=["year_rated", "month"], how="left")
    n_y["above"] = (n_y["n"] / n_y["n_y"]) > n_y["p_ym"]
    rec = n_y.groupby("movieId").agg(recur_years=("above", "size"), recurrence=("above", "mean"))
    meta = meta.join(rec)
    meta["recur_years"] = meta["recur_years"].fillna(0).astype(int)
    meta["recurrence"] = meta["recurrence"].fillna(0.0)
    meta["seasonal"] = (
        (meta.peak_lift >= SEASONAL_MIN_LIFT)
        & (meta.peak_z >= SEASONAL_MIN_Z)
        & (meta.peak_obs >= SEASONAL_MIN_PEAK_N)
        & (meta.recur_years >= RECUR_MIN_YEARS)
        & (meta.recurrence >= RECUR_MIN_SHARE)
    )

    # validated holiday months (from tag groups that pass HOLIDAY_MIN_WINDOW_LIFT) filled in below
    meta["seasonal_used"] = False

    # weekly rhythm (clean ratings; all years — used for suggestions only)
    dw = pd.read_parquet(data.PROC_DIR / "agg_movie_dow_clean.parquet")
    dw["weekend"] = dw["dow"].isin(WEEKEND)
    wk = dw.groupby(["movieId", "weekend"])["n"].sum().unstack(fill_value=0)
    wk.columns = ["weekday", "weekend"] if list(wk.columns) == [False, True] else [str(c) for c in wk.columns]
    base_we = wk["weekend"].sum() / wk.values.sum()
    gwk = wk.reset_index().merge(mg, on="movieId").groupby("genre")[["weekday", "weekend"]].sum()
    g_we_lift = (gwk["weekend"] / gwk.sum(axis=1)) / base_we
    m_prior = mg.merge(g_we_lift.rename("gl"), left_on="genre", right_index=True).groupby("movieId")["gl"].mean()
    m_prior = m_prior.reindex(wk.index).fillna(1.0)
    n_wk = wk.sum(axis=1)
    we_lift = (wk["weekend"] + PRIOR_STRENGTH * base_we * m_prior) / ((n_wk + PRIOR_STRENGTH) * base_we)

    # holiday + length tags
    tags = data.load_tags()
    tag_users = tags.groupby(["movieId", "tag_norm"])["userId"].nunique().rename("users").reset_index()
    holiday_rows = []
    for key, h in HOLIDAYS.items():
        sub = tag_users[tag_users["tag_norm"].isin(h["tags"])].groupby("movieId")["users"].sum()
        for mid, u in sub[sub >= MIN_TAG_USERS].items():
            holiday_rows.append((int(mid), key, int(u)))
    hol = pd.DataFrame(holiday_rows, columns=["movieId", "holiday", "tag_users"])
    # a movie with several holiday tags keeps the most-tagged one
    hol = hol.sort_values("tag_users", ascending=False).drop_duplicates("movieId").set_index("movieId")
    long_u = tag_users[tag_users["tag_norm"].isin(LONG_TAGS)].groupby("movieId")["users"].sum()
    long_ids = set(long_u[long_u >= MIN_TAG_USERS].index.astype(int))

    # empirical lift of each holiday tag group in its window (evidence for reasons)
    hol_lift = {}
    for key, h in HOLIDAYS.items():
        mids = hol.index[hol["holiday"] == key]
        o = obs.reindex(mids).dropna()
        if len(o) == 0:
            continue
        share = o.sum(axis=0) / o.values.sum()
        lf = share / p_month
        hol_lift[key] = {
            "window_lift": float(o[h["months"]].values.sum() / o.values.sum() / p_month[h["months"]].sum()),
            "month_lift": {int(m): float(lf[m]) for m in range(1, 13)},
            "n_movies": int(len(o)),
            "n_ratings": int(o.values.sum()),
        }

    hol_months = sorted({m for k, v in hol_lift.items() if v["window_lift"] >= HOLIDAY_MIN_WINDOW_LIFT for m in HOLIDAYS[k]["months"]})
    meta["seasonal_used"] = meta["seasonal"] & (meta["peak_month"].isin(hol_months) | (meta["recurrence"] >= RECUR_STRONG))

    return {
        "holiday_months": hol_months,
        "p_month": p_month,
        "lift": lift.astype("float32"),
        "raw_lift": raw_lift.astype("float32"),
        "meta": meta,
        "genre_lift": g_lift,
        "weekend_lift": we_lift.astype("float32"),
        "genre_weekend_lift": g_we_lift,
        "base_weekend_share": float(base_we),
        "holiday": hol,
        "holiday_lift": hol_lift,
        "long_ids": long_ids,
        "genres": movies.set_index("movieId")["genre_list"].to_dict(),
        "years": (min_year, max_year),
    }


def save(model: dict, name: str = "when_to_watch") -> None:
    data.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(data.MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(name: str = "when_to_watch") -> dict:
    with open(data.MODEL_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Personal rhythm
# --------------------------------------------------------------------------- #
def personal_profile(user_ratings: pd.DataFrame, genres: dict[int, list[str]], min_n: int = 20) -> dict[str, tuple[float, int]]:
    """Weekend share of a user's returning-session ratings per genre.

    user_ratings: columns movieId, timestamp (Unix s). Returns {genre: (weekend_share, n)}
    for genres with >= min_n such ratings, plus "_all".
    """
    if user_ratings is None or len(user_ratings) == 0 or "timestamp" not in user_ratings:
        return {}
    ur = user_ratings.copy()
    ur = ur[ur["timestamp"] - ur["timestamp"].min() >= 86_400]  # drop onboarding day
    if len(ur) < min_n:
        return {}
    dow = pd.to_datetime(ur["timestamp"], unit="s", utc=True).dt.tz_convert(data.LOCAL_TZ).dt.dayofweek
    ur["we"] = dow.isin(WEEKEND).to_numpy()
    prof = {"_all": (float(ur["we"].mean()), int(len(ur)))}
    ur["genre"] = ur["movieId"].map(lambda m: genres.get(int(m), []))
    ex = ur.explode("genre").dropna(subset=["genre"])
    for g, grp in ex.groupby("genre"):
        if len(grp) >= min_n:
            prof[g] = (float(grp["we"].mean()), int(len(grp)))
    return prof


# --------------------------------------------------------------------------- #
# Suggestion rule
# --------------------------------------------------------------------------- #
def _next_anchor(today: date, month: int, day: int) -> date:
    d = date(today.year, month, day)
    return d if d >= today else date(today.year + 1, month, day)


def suggest(
    model: dict,
    movie_id: int,
    today: date | None = None,
    context: str = "any",
    profile: dict | None = None,
) -> Suggestion:
    """Viewing suggestion for one movie.

    context: "any" | "weeknight" | "weekend" | "short_on_time" — adjusts fit_now.
    """
    today = today or date.today()
    mid = int(movie_id)
    m0 = today.month
    meta = model["meta"]
    lift = model["lift"]
    has_season = mid in meta.index
    lift_row = lift.loc[mid] if has_season else None
    lift_now = float(lift_row[m0]) if has_season else 1.0
    n_clean = int(meta.at[mid, "n_clean"]) if has_season else 0
    peak_month = int(meta.at[mid, "peak_month"]) if has_season else None
    peak_lift = float(meta.at[mid, "peak_lift"]) if has_season else None
    seasonal = bool(meta.at[mid, "seasonal_used"]) if has_season else False
    long_flag = mid in model["long_ids"]

    holiday = model["holiday"]["holiday"].get(mid) if mid in model["holiday"].index else None
    if holiday is not None and model["holiday_lift"].get(holiday, {}).get("window_lift", 0) < HOLIDAY_MIN_WINDOW_LIFT:
        holiday = None  # tag group showed no real seasonal lift in the data -> not used
    action, when, headline, reason = "now", "Now", "Good to watch now", ""

    if holiday is not None:
        h = HOLIDAYS[holiday]
        tag_users = int(model["holiday"].at[mid, "tag_users"])
        hl = model["holiday_lift"].get(holiday, {})
        anchor = _next_anchor(today, *h["anchor"])
        days = (anchor - today).days
        ev = f"{hl['window_lift']:.1f}×" if hl else "higher"
        win = "–".join(MONTH_NAMES[m][:3] for m in (h["months"][0], h["months"][-1])) if len(h["months"]) > 1 else MONTH_NAMES[h["months"][0]]
        if m0 in h["months"] or 0 <= days <= LEAD_DAYS:
            headline = f"In season now: {h['label']}"
            when = "Now"
            reason = f"Tagged '{holiday.replace('_', ' ')}' by {tag_users} users; such films get {ev} their usual share of ratings in {win}."
        else:
            action, when = "save", MONTH_NAMES[h["months"][0]]
            headline = f"Save for {h['label']} ({win})"
            reason = f"Tagged '{holiday.replace('_', ' ')}' by {tag_users} users; such films get {ev} their usual share of ratings in {win}. Next {h['label']}: in {days} days."
        fit_now = max(lift_now, 1.0) if action == "now" else min(lift_now, 0.8)
    elif seasonal:
        pm = MONTH_NAMES[peak_month]
        if peak_month == m0 or lift_now >= 1.10:
            headline = f"Peak season now ({pm})" if peak_month == m0 else f"In season now (peak: {pm})"
            reason = f"Rated {lift_now:.1f}× more than usual in {MONTH_NAMES[m0]} ({n_clean:,} returning-session ratings)."
        else:
            action, when = "save", pm
            days = (_next_anchor(today, peak_month, 1) - today).days
            headline = f"Save for {pm}" + (f" (starts in {days} days)" if days <= LEAD_DAYS else "")
            reason = f"Rated {peak_lift:.1f}× more than usual in {pm}; {lift_now:.2f}× in {MONTH_NAMES[m0]} ({n_clean:,} returning-session ratings)."
        fit_now = lift_now
    else:
        headline = "Good any time of year"
        if has_season and n_clean >= 200:
            reason = f"No reliable seasonal pattern in {n_clean:,} returning-session ratings (highest month: {MONTH_NAMES[peak_month]}, {peak_lift:.2f}×)."
        elif has_season:
            reason = f"Too few time-stamped ratings ({n_clean}) to detect a season for this film."
        else:
            reason = "No time-stamped history for this film; no seasonal signal."
        # gentle genre-level lean (e.g., horror in October), mentioned but not acted on
        gl = model["genre_lift"]
        leans = [(g, m, float(gl.at[g, m])) for g in model["genres"].get(mid, []) if g in gl.index and g != "IMAX" for m in range(1, 13) if gl.at[g, m] >= GENRE_HINT_LIFT]
        if leans:
            g, m, v = max(leans, key=lambda x: x[2])
            reason += f" Genre lean: {g} ratings run {v:.2f}× in {MONTH_NAMES[m]}."
        fit_now = 1.0

    # weekly rhythm
    we = float(model["weekend_lift"].get(mid, np.nan)) if mid in model["weekend_lift"].index else np.nan
    if np.isnan(we):
        gl = [model["genre_weekend_lift"].get(g, 1.0) for g in model["genres"].get(mid, [])]
        we = float(np.mean(gl)) if gl else 1.0
    if we >= WEEKEND_BEST:
        weekly = f"Best on a weekend: {we:.2f}× the usual Sat–Sun share of ratings."
    elif we <= WEEKNIGHT_OK:
        weekly = f"Fine for a weeknight: rated more on weekdays ({we:.2f}× weekend share)."
    else:
        weekly = "Any night works: no weekday/weekend skew."
    if long_flag:
        weekly += " Users tag it as long — save it for a free evening."

    personal = None
    if profile:
        gs = [(g, profile[g]) for g in model["genres"].get(mid, []) if g in profile]
        base = profile.get("_all", (None, 0))[0]
        if gs and base is not None:
            g, (share, n) = max(gs, key=lambda x: x[1][1])
            if share - base >= 0.10:
                personal = f"You tend to rate {g} on weekends ({share:.0%} vs {base:.0%} of all your ratings)."
            elif base - share >= 0.10:
                personal = f"You tend to rate {g} on weekdays ({1 - share:.0%} vs {1 - base:.0%} overall)."

    # context adjustment
    if context == "weekend":
        fit_now *= we
    elif context == "weeknight":
        fit_now *= 1.0 / we
    elif context == "short_on_time" and long_flag:
        fit_now *= 0.5
    return Suggestion(
        movieId=mid, action=action, when=when, headline=headline, reason=reason, weekly=weekly,
        fit_now=float(fit_now), peak_month=peak_month, peak_lift=peak_lift, evidence_n=n_clean,
        holiday=holiday, long_flag=long_flag, personal=personal,
    )


def suggest_many(model: dict, movie_ids, today: date | None = None, context: str = "any", profile: dict | None = None) -> pd.DataFrame:
    return pd.DataFrame([suggest(model, m, today, context, profile).to_dict() for m in movie_ids])
