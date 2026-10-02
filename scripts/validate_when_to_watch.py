"""Validate the when-to-watch module.

1. Out-of-time test: fit seasonality on clean ratings from <= 2018, then check
   whether flagged movies' 2019-2023 ratings really concentrate in the
   predicted peak month (lift > 1), vs. two controls.
2. Holiday tag groups: observed lift in their calendar window.
3. Intuition check: suggestions for well-known holiday / non-holiday films.
4. Figure: monthly lift for tag groups and genres.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402

from recsys import data  # noqa: E402
from recsys import when_to_watch as w  # noqa: E402
from recsys.plotting import INK2, MUTED, SERIES, apply_style, save  # noqa: E402

TABLES = data.ROOT / "reports" / "tables"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SPLIT_YEAR = 2018
TODAY = date(2026, 9, 26)


def test_lift(obs_test: pd.DataFrame, p_test: pd.Series, movie_month: pd.Series) -> pd.Series:
    """Lift in the test period at a given month per movie."""
    o = obs_test.reindex(movie_month.index)
    n = o.sum(axis=1)
    got = np.array([o.at[m, mo] if m in o.index else np.nan for m, mo in movie_month.items()])
    return pd.Series(got / n.to_numpy() / p_test[movie_month.to_numpy()].to_numpy(), index=movie_month.index)


def main() -> None:
    apply_style()
    rng = np.random.default_rng(42)
    movies = data.load_movies().set_index("movieId")

    # ------------------------------------------------------------ 1. out-of-time test
    train_model = w.build(max_year=SPLIT_YEAR)
    mm = pd.read_parquet(data.PROC_DIR / "agg_movie_month_clean.parquet")
    te = mm[mm["year_rated"] > SPLIT_YEAR]
    obs_test = te.groupby(["movieId", "month"])["n"].sum().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    p_test = obs_test.sum(axis=0) / obs_test.values.sum()
    enough = obs_test.index[obs_test.sum(axis=1) >= 24]
    meta = train_model["meta"]
    meta = meta[meta.index.isin(enough)]

    groups = {
        "Final rule (used in app)": meta[meta.seasonal_used],
        "Flagged seasonal, other months": meta[meta.seasonal & ~meta.seasonal_used],
        "Peak only (lift ≥ 1.15, z ≥ 3; no recurrence)": meta[(meta.peak_lift >= w.SEASONAL_MIN_LIFT) & (meta.peak_z >= w.SEASONAL_MIN_Z) & ~meta.seasonal],
        "All other films (their highest month)": meta[~((meta.peak_lift >= w.SEASONAL_MIN_LIFT) & (meta.peak_z >= w.SEASONAL_MIN_Z))],
    }
    rows = []
    for name, g in groups.items():
        tl = test_lift(obs_test, p_test, g["peak_month"])
        rand_month = pd.Series(rng.integers(1, 13, size=len(g)), index=g.index)
        rl = test_lift(obs_test, p_test, rand_month)
        boots = [np.median(rng.choice(tl.dropna().to_numpy(), size=tl.notna().sum())) for _ in range(1000)]
        rows.append({
            "group": name,
            "movies": int(tl.notna().sum()),
            "train_median_peak_lift": float(g["peak_lift"].median()),
            "test_median_lift_at_peak": float(tl.median()),
            "test_median_lift_ci_lo": float(np.quantile(boots, 0.025)),
            "test_median_lift_ci_hi": float(np.quantile(boots, 0.975)),
            "test_share_lift_gt_1": float((tl > 1).mean()),
            "test_share_lift_ge_1_15": float((tl >= 1.15).mean()),
            "control_random_month_median_lift": float(rl.median()),
        })
    hol_tr = train_model["holiday"]
    for key, mo in [("christmas", 12), ("halloween", 10)]:
        ids = [i for i in hol_tr.index[hol_tr.holiday == key] if i in enough]
        tl = test_lift(obs_test, p_test, pd.Series(mo, index=ids))
        rl = test_lift(obs_test, p_test, pd.Series(rng.integers(1, 13, size=len(ids)), index=ids))
        boots = [np.median(rng.choice(tl.dropna().to_numpy(), size=tl.notna().sum())) for _ in range(1000)]
        rows.insert(0, {
            "group": f"'{key}'-tagged films ({MONTHS[mo - 1]})", "movies": int(tl.notna().sum()),
            "train_median_peak_lift": np.nan, "test_median_lift_at_peak": float(tl.median()),
            "test_median_lift_ci_lo": float(np.quantile(boots, 0.025)), "test_median_lift_ci_hi": float(np.quantile(boots, 0.975)),
            "test_share_lift_gt_1": float((tl > 1).mean()), "test_share_lift_ge_1_15": float((tl >= 1.15).mean()),
            "control_random_month_median_lift": float(rl.median()),
        })
    oot = pd.DataFrame(rows)
    oot.to_csv(TABLES / "wtw_out_of_time_validation.csv", index=False)
    print(oot.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(oot))[::-1]
    ax.barh(y, oot["test_median_lift_at_peak"], color=SERIES[0], height=0.5, label="Predicted peak month")
    err = np.vstack([oot["test_median_lift_at_peak"] - oot["test_median_lift_ci_lo"], oot["test_median_lift_ci_hi"] - oot["test_median_lift_at_peak"]])
    ax.errorbar(oot["test_median_lift_at_peak"], y, xerr=err, fmt="none", ecolor=INK2, capsize=3, lw=1)
    ax.scatter(oot["control_random_month_median_lift"], y, color=SERIES[1], zorder=3, s=40, label="Control: random month")
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y, [f"{g}\n(n = {n:,})" for g, n in zip(oot.group, oot.movies)], fontsize=9)
    ax.set_xlabel("Median lift in 2019–2023 (1.0 = no seasonal effect)")
    ax.set_title("Out-of-time check: seasons learned on ≤2018 ratings, tested on 2019–2023")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "wtw_out_of_time.png")

    # ------------------------------------------------------------ full model
    model = w.build()
    w.save(model)
    hl = pd.DataFrame([
        {"holiday": k, "window": "–".join(MONTHS[m - 1] for m in (w.HOLIDAYS[k]["months"][0], w.HOLIDAYS[k]["months"][-1])),
         "tagged_movies": v["n_movies"], "clean_ratings": v["n_ratings"], "window_lift": v["window_lift"],
         "used": v["window_lift"] >= w.HOLIDAY_MIN_WINDOW_LIFT}
        for k, v in model["holiday_lift"].items()
    ]).sort_values("window_lift", ascending=False)
    hl.to_csv(TABLES / "wtw_holiday_tag_lift.csv", index=False)
    print(hl.to_string())
    me = model["meta"]
    pd.Series({
        "movies_with_clean_ratings": int(len(me)),
        "movies_flagged_seasonal": int(me.seasonal.sum()),
        "movies_seasonal_used": int(me.seasonal_used.sum()),
        "movies_with_used_holiday_tag": int(model["holiday"]["holiday"].isin(hl.loc[hl.used, "holiday"]).sum()),
        "movies_flagged_long": int(len(model["long_ids"])),
        "base_weekend_share": model["base_weekend_share"],
    }).to_csv(TABLES / "wtw_summary.csv", header=["value"])
    top = me[me.seasonal_used].sort_values("peak_lift", ascending=False).head(20).join(movies[["title"]])
    top["peak_month"] = top["peak_month"].map(lambda m: MONTHS[m - 1])
    top[["title", "n_clean", "peak_month", "peak_lift", "recur_years", "recurrence"]].to_csv(TABLES / "wtw_top_seasonal_movies.csv")

    # ------------------------------------------------------------ 3. intuition check
    probe = {
        "Home Alone (1990)": 586, "Elf (2003)": 5452 if 5452 in movies.index else None,
        "It's a Wonderful Life (1946)": 953, "A Christmas Story (1983)": 2804, "The Nightmare Before Christmas (1993)": 551,
        "Halloween (1978)": 1982, "Hocus Pocus (1993)": 2413 if 2413 in movies.index else None,
        "Trick 'r Treat (2007)": 71500, "The Ten Commandments (1956)": 7386,
        "Toy Story (1995)": 1, "The Shawshank Redemption (1994)": 318, "The Dark Knight (2008)": 58559,
    }
    # resolve by title where the id guess is wrong
    def find(title_part: str, year: int) -> int | None:
        c = movies[(movies.clean_title.str.lower() == title_part.lower()) & (movies.year == year)]
        return int(c.index[0]) if len(c) else None
    resolved = {}
    for label, mid in probe.items():
        name, yr = label.rsplit(" (", 1)
        found = find(name, int(yr.rstrip(")")))
        resolved[label] = found if found is not None else mid
    rows = []
    for label, mid in resolved.items():
        if mid is None:
            continue
        s = w.suggest(model, mid, TODAY)
        rows.append({"movie": label, "movieId": mid, "headline": s.headline, "reason": s.reason, "weekly": s.weekly})
    intu = pd.DataFrame(rows)
    intu.to_csv(TABLES / "wtw_intuition_check.csv", index=False)
    print(intu[["movie", "headline"]].to_string())

    # ------------------------------------------------------------ 4. seasonal lift figure
    obs = mm.groupby(["movieId", "month"])["n"].sum().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    p = obs.sum(axis=0) / obs.values.sum()

    def group_lift(mids) -> pd.Series:
        o = obs.reindex(list(mids)).dropna()
        return (o.sum(axis=0) / o.values.sum()) / p, int(o.values.sum())

    hol = model["holiday"]
    gl = model["genre_lift"]
    series = [
        ("'christmas'-tagged films", *group_lift(hol.index[hol.holiday == "christmas"])),
        ("'halloween'-tagged films", *group_lift(hol.index[hol.holiday == "halloween"])),
        ("Horror (genre)", gl.loc["Horror"], None),
        ("Musical (genre)", gl.loc["Musical"], None),
        ("Documentary (genre)", gl.loc["Documentary"], None),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(1, 13)
    for i, (lab, s, n) in enumerate(series):
        ax.plot(x, s.to_numpy(), color=SERIES[i], marker="o", ms=5, lw=2, label=lab)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1)
    ax.set_xticks(x, MONTHS)
    ax.set_ylabel("Lift vs. month's overall share")
    ax.set_title("Seasonal lift of rating activity (returning-session ratings, films ≥2 yrs old)")
    ax.legend(ncol=2, loc="upper left")
    for i, (lab, s, n) in enumerate(series[:2]):
        m = int(s.idxmax())
        ax.annotate(f"{s.max():.2f}×", (m, s.max()), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=INK2)
    save(fig, "wtw_seasonal_lift.png")
    print("validation done")


if __name__ == "__main__":
    main()
