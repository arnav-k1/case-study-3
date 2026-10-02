"""Exploratory data analysis on the FULL MovieLens 32M dataset.

Outputs
- reports/figures/eda_*.png (200 dpi)
- reports/tables/dataset_summary.csv, eda_*.csv
- data/processed/agg_*.parquet  (time aggregates reused by when_to_watch.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from recsys import data  # noqa: E402
from recsys.plotting import (  # noqa: E402
    BLUE, DIVERGING_MID, DIVERGING_RED, INK2, MUTED, ORANGE, SEQ_BLUES, apply_style, save,
)

TABLES = data.ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOWS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIVERGING = LinearSegmentedColormap.from_list("div", [SEQ_BLUES[5], SEQ_BLUES[2], DIVERGING_MID, "#f0a3a2", DIVERGING_RED])


def main() -> None:
    apply_style()
    print("loading ratings ...")
    r = data.load_ratings()
    movies = data.load_movies()
    tags = data.load_tags()
    r = data.add_local_time(r)
    n_users, n_items, n_r = r.userId.nunique(), r.movieId.nunique(), len(r)
    assert n_r == data.EXPECTED_ROWS["ratings"]

    # ------------------------------------------------------------------ summary
    user_n = r.groupby("userId").size()
    item_n = r.groupby("movieId").size()
    first_ts = r.groupby("userId")["timestamp"].transform("min")
    same_day = (r["timestamp"] - first_ts) < 86_400
    burst_share = float(same_day.mean())
    per_user_burst = same_day.groupby(r["userId"]).mean()
    top1pct_items = item_n.sort_values(ascending=False).head(max(1, len(item_n) // 100)).sum() / n_r
    summary = {
        "ratings": n_r,
        "users": n_users,
        "movies_rated": n_items,
        "movies_in_catalog": len(movies),
        "tag_applications": len(tags),
        "density_pct": 100 * n_r / (n_users * n_items),
        "sparsity_pct": 100 * (1 - n_r / (n_users * n_items)),
        "mean_rating": float(r.rating.mean()),
        "median_ratings_per_user": float(user_n.median()),
        "mean_ratings_per_user": float(user_n.mean()),
        "max_ratings_per_user": int(user_n.max()),
        "median_ratings_per_movie": float(item_n.median()),
        "movies_with_lt5_ratings_pct": float(100 * (item_n < 5).mean()),
        "movies_with_ge50_ratings": int((item_n >= 50).sum()),
        "ratings_share_top1pct_movies": float(100 * top1pct_items),
        "share_ratings_within_24h_of_users_first_rating_pct": 100 * burst_share,
        "median_user_share_first_day_pct": float(100 * per_user_burst.median()),
        "users_with_all_ratings_on_first_day_pct": float(100 * (per_user_burst == 1).mean()),
        "first_rating_utc": str(pd.to_datetime(r.timestamp.min(), unit="s")),
        "last_rating_utc": str(pd.to_datetime(r.timestamp.max(), unit="s")),
        "half_star_share_pct": float(100 * (r.rating % 1 != 0).mean()),
    }
    pd.Series(summary).to_csv(TABLES / "dataset_summary.csv", header=["value"])
    print(pd.Series(summary))

    # ------------------------------------------------------------------ fig: rating distribution
    dist = r.rating.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(dist.index, dist.values / 1e6, width=0.38, color=BLUE)
    ax.set_xticks(dist.index)
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Ratings (millions)")
    ax.set_title("Rating distribution (32.0M ratings)")
    ax.grid(axis="x", visible=False)
    save(fig, "eda_rating_distribution.png")

    # ------------------------------------------------------------------ fig: long tail
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, counts, what, color in [(axes[0], user_n, "user", BLUE), (axes[1], item_n, "movie", ORANGE)]:
        s = np.sort(counts.to_numpy())[::-1]
        ax.loglog(np.arange(1, len(s) + 1), s, color=color)
        ax.set_xlabel(f"{what.capitalize()} rank (most active first)")
        ax.set_ylabel(f"Ratings per {what}")
        ax.set_title(f"Ratings per {what} (log-log)")
        ax.axhline(np.median(s), color=MUTED, lw=1, ls="--")
        ax.text(1.5, np.median(s) * 1.25, f"median = {np.median(s):,.0f}", color=INK2, fontsize=9)
    fig.suptitle(f"Long tail: matrix is {summary['sparsity_pct']:.2f}% empty", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    save(fig, "eda_long_tail.png")

    # ------------------------------------------------------------------ fig: ratings per year
    per_year = r.groupby("year_rated").size()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(per_year.index, per_year.values / 1e6, color=BLUE, width=0.8)
    ax.set_xlabel("Year rated (US Central)")
    ax.set_ylabel("Ratings (millions)")
    ax.set_title("Ratings per year, 1995–2023")
    ax.grid(axis="x", visible=False)
    save(fig, "eda_ratings_per_year.png")
    per_year.rename("ratings").to_csv(TABLES / "eda_ratings_per_year.csv")

    # ------------------------------------------------------------------ genre stats
    mg = movies[["movieId", "genre_list"]].explode("genre_list").dropna()
    mg = mg.rename(columns={"genre_list": "genre"})
    r["r2"] = r["rating"].astype("float64") ** 2
    item_stats = r.groupby("movieId").agg(n=("rating", "size"), s=("rating", "sum"), s2=("r2", "sum"))
    r.drop(columns="r2", inplace=True)
    g = mg.merge(item_stats, left_on="movieId", right_index=True)
    genre = g.groupby("genre").agg(n=("n", "sum"), s=("s", "sum"), s2=("s2", "sum"), movies=("movieId", "nunique"))
    genre["mean"] = genre.s / genre.n
    genre["sd"] = np.sqrt(genre.s2 / genre.n - genre["mean"] ** 2)
    genre = genre.drop(index=[x for x in ["IMAX"] if x in genre.index]).sort_values("n", ascending=True)
    genre[["n", "movies", "mean", "sd"]].to_csv(TABLES / "eda_genre_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    axes[0].barh(genre.index, genre.n / 1e6, color=BLUE, height=0.7)
    axes[0].set_xlabel("Ratings (millions; a movie counts in each of its genres)")
    axes[0].set_title("Genre popularity")
    axes[0].grid(axis="y", visible=False)
    order = genre.index
    axes[1].scatter(genre["mean"], order, color=ORANGE, s=40, zorder=3)
    overall = summary["mean_rating"]
    axes[1].axvline(overall, color=MUTED, ls="--", lw=1)
    axes[1].text(overall + 0.01, -0.9, f"all ratings {overall:.2f}", color=INK2, fontsize=9)
    axes[1].set_xlabel("Mean rating (stars)")
    axes[1].set_title("Average rating by genre")
    axes[1].grid(axis="y", visible=False)
    save(fig, "eda_genres.png")

    # ------------------------------------------------------------------ fig: hour x day-of-week
    hd = r.groupby(["dow", "hour"]).size().unstack(fill_value=0)
    hd_share = hd / hd.values.sum() * 100
    fig, ax = plt.subplots(figsize=(11, 3.6))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUES)
    im = ax.imshow(hd_share.values, aspect="auto", cmap=cmap)
    ax.set_yticks(range(7), DOWS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (US Central; converted from UTC)")
    ax.set_title("When ratings are entered: day of week × hour")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% of all ratings")
    save(fig, "eda_hour_dow.png")
    hd.to_csv(TABLES / "eda_dow_hour_counts.csv")

    # ------------------------------------------------------------------ time aggregates for when-to-watch
    # Two noise sources are removed for the "clean" aggregates:
    #  (1) onboarding bursts: ratings within 24 h of a user's first rating are
    #      mostly backfilled history (Harper & Konstan, 2015), not recent viewing;
    #  (2) release spikes: ratings in the release year or the next one follow the
    #      release calendar, so a December release would look like a "December movie".
    yr = movies.set_index("movieId")["year"].astype("float32")
    age = r["year_rated"].astype("float32") - r["movieId"].map(yr).to_numpy()
    mature = (age >= 2).to_numpy()
    returning = (~same_day).to_numpy()
    clean = mature & returning
    for period in ["month", "dow", "hour"]:
        for suffix, mask in [("all", None), ("clean", clean)]:
            sub = r if mask is None else r[mask]
            # month aggregates keep the rating year so seasonality can be validated out of time
            keys = ["movieId", "year_rated", "month"] if period == "month" else ["movieId", period]
            agg = sub.groupby(keys).size().rename("n").reset_index()
            agg["n"] = agg["n"].astype(np.int32)
            agg.to_parquet(data.PROC_DIR / f"agg_movie_{period}_{suffix}.parquet", index=False)
    item_summary = r.groupby("movieId").agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"), n_users=("userId", "nunique")).reset_index()
    item_summary.to_parquet(data.PROC_DIR / "item_summary.parquet", index=False)
    filt = pd.Series({
        "mature_share_pct": 100 * float(mature.mean()),
        "returning_share_pct": 100 * float(returning.mean()),
        "clean_share_pct": 100 * float(clean.mean()),
        "clean_ratings": int(clean.sum()),
    })
    filt.to_csv(TABLES / "eda_time_filter_summary.csv", header=["value"])
    print(filt)

    # ------------------------------------------------------------------ genre x month lift (clean ratings)
    mm = pd.read_parquet(data.PROC_DIR / "agg_movie_month_clean.parquet").groupby(["movieId", "month"]).n.sum().reset_index()
    gm = mm.merge(mg, on="movieId").groupby(["genre", "month"]).n.sum().unstack(fill_value=0)
    gm = gm.drop(index=[x for x in ["IMAX"] if x in gm.index])
    month_tot = mm.groupby("month").n.sum()
    # lift = P(genre | month) / P(genre)  (controls for overall monthly activity)
    lift = (gm / month_tot) / (gm.sum(axis=1) / month_tot.sum()).to_numpy()[:, None]
    lift.to_csv(TABLES / "eda_genre_month_lift.csv")
    order = lift.max(axis=1).sort_values(ascending=False).index
    lift = lift.loc[order]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    span = max(0.15, float(np.abs(lift.values - 1).max()))
    norm = TwoSlopeNorm(vmin=1 - span, vcenter=1.0, vmax=1 + span)
    im = ax.imshow(lift.values, aspect="auto", cmap=DIVERGING, norm=norm)
    ax.set_yticks(range(len(lift)), lift.index)
    ax.set_xticks(range(12), MONTHS)
    ax.grid(False)
    for i in range(lift.shape[0]):
        for j in range(12):
            v = lift.values[i, j]
            if abs(v - 1) >= 0.06:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v - 1) >= 0.6 * span else "#0b0b0b")
    ax.set_title("Genre seasonality (returning-session ratings, films ≥2 yrs old)\n"
                 "lift = genre's share of the month's ratings ÷ its average share")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Lift (1.0 = no seasonal effect)")
    save(fig, "eda_genre_month_lift.png")

    # ------------------------------------------------------------------ top tags
    top_tags = tags.groupby("tag_norm").agg(applications=("movieId", "size"), movies=("movieId", "nunique"), users=("userId", "nunique"))
    top_tags = top_tags.sort_values("users", ascending=False)
    top_tags.head(50).to_csv(TABLES / "eda_top_tags.csv")
    t20 = top_tags.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(t20.index, t20.users, color=BLUE, height=0.7)
    ax.set_xlabel("Distinct users who applied the tag")
    ax.set_title("Top 20 tags (by number of distinct users)")
    ax.grid(axis="y", visible=False)
    save(fig, "eda_top_tags.png")
    tag_users = tags.groupby("userId").size()
    print("tagging users:", tag_users.size, "top-1% users share of tags:",
          tag_users.sort_values(ascending=False).head(max(1, tag_users.size // 100)).sum() / len(tags))
    pd.Series({
        "tagging_users": int(tag_users.size),
        "tag_share_from_top1pct_taggers_pct": float(100 * tag_users.sort_values(ascending=False).head(max(1, tag_users.size // 100)).sum() / len(tags)),
        "distinct_tags": int(tags.tag_norm.nunique()),
    }).to_csv(TABLES / "eda_tag_summary.csv", header=["value"])

    # ------------------------------------------------------------------ onboarding burst
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(per_user_burst * 100, bins=20, color=BLUE, rwidth=0.9)
    ax.set_xlabel("% of a user's ratings entered within 24 h of their first rating")
    ax.set_ylabel("Users")
    ax.set_title("Most ratings are entered in an onboarding burst")
    ax.grid(axis="x", visible=False)
    save(fig, "eda_onboarding_burst.png")
    print("EDA done")


if __name__ == "__main__":
    main()
