"""CineCompass — MovieLens 32M recommender with "when to watch" suggestions.

Run:  streamlit run app/app.py
Requires artifacts from:  python scripts/train_final.py
"""
from __future__ import annotations

import html
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys import app_core as core  # noqa: E402
from recsys import personas  # noqa: E402
from recsys import when_to_watch as w  # noqa: E402

st.set_page_config(page_title="CineCompass", page_icon="🎬", layout="wide")

GENRE_COLORS = {
    "Action": "#e34948", "Adventure": "#eb6834", "Animation": "#1baf7a", "Children": "#2a9d8f", "Comedy": "#eda100",
    "Crime": "#52514e", "Documentary": "#6d6a64", "Drama": "#2a78d6", "Fantasy": "#4a3aa7", "Film-Noir": "#0b0b0b",
    "Horror": "#8b1e1e", "IMAX": "#898781", "Musical": "#e87ba4", "Mystery": "#5b4a8a", "Romance": "#d55181",
    "Sci-Fi": "#184f95", "Thriller": "#7a3b12", "War": "#556b2f", "Western": "#a0642c",
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;
       background: rgba(128,128,128,.04); min-height: 250px;}
.card h4 {margin: 0 0 2px 0; font-size: 1.05rem; line-height: 1.3;}
.muted {opacity: .72; font-size: .85rem;}
.chip {display:inline-block; padding: 1px 8px; margin: 3px 4px 0 0; border-radius: 999px; font-size: .72rem; color: #fff;}
.bar {height: 6px; border-radius: 3px; background: rgba(128,128,128,.2); margin: 8px 0 2px 0;}
.bar > div {height: 6px; border-radius: 3px; background: #2a78d6;}
.when {margin-top: 10px; padding: 8px 10px; border-radius: 8px; font-size: .86rem;}
.when.now {background: rgba(27,175,122,.12); border-left: 3px solid #1baf7a;}
.when.save {background: rgba(237,161,0,.14); border-left: 3px solid #eda100;}
.why {font-size: .84rem; margin-top: 8px;}
.small {font-size: .78rem; opacity: .8; margin-top: 4px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model …")
def artifacts() -> core.Artifacts:
    return core.load_artifacts()


@st.cache_data(show_spinner=False)
def onboarding(_art_id: int) -> pd.DataFrame:
    return core.onboarding_titles(artifacts())


def chips(genres) -> str:
    genres = [g for g in genres if g != "IMAX"]  # a format, not a genre
    return "".join(f'<span class="chip" style="background:{GENRE_COLORS.get(g, "#898781")}">{html.escape(g)}</span>' for g in genres)


def card(row: pd.Series) -> str:
    year = "" if pd.isna(row["year"]) else f" ({int(row['year'])})"
    because = f'<div class="why">👍 Because you liked <b>{html.escape(row["because"])}</b></div>' if row.get("because") else ""
    personal = f'<div class="small">🙋 {html.escape(row["personal"])}</div>' if row.get("personal") else ""
    icon = "▶️" if row["action"] == "now" else "📅"
    return f"""
<div class="card">
  <h4>{html.escape(str(row['clean_title']))}{year}</h4>
  <div>{chips(row['genre_list'])}</div>
  <div class="bar"><div style="width:{row['fit'] * 100:.0f}%"></div></div>
  <div class="muted">Fit: <b>{row['fit_label']}</b> · avg ★ {row['mean_rating']:.1f} from {int(row['n_ratings']):,} ratings</div>
  {because}
  <div class="when {row['action']}"><b>{icon} {html.escape(row['headline'])}</b><br>
    <span class="small">{html.escape(row['reason'])}</span><br>
    <span class="small">🗓️ {html.escape(row['weekly'])}</span>{personal}
  </div>
</div>"""


def render_grid(recs: pd.DataFrame, cols: int = 3) -> None:
    for start in range(0, len(recs), cols):
        cs = st.columns(cols)
        for c, (_, row) in zip(cs, recs.iloc[start:start + cols].iterrows()):
            c.markdown(card(row), unsafe_allow_html=True)


def calendar_view(recs: pd.DataFrame, today: date) -> None:
    order = ["Now"] + [w.MONTH_NAMES[(today.month - 1 + k) % 12 + 1] for k in range(1, 12)]
    groups = recs.groupby("when")
    for label in order:
        if label not in groups.groups:
            continue
        g = groups.get_group(label)
        st.markdown(f"**{'Watch now' if label == 'Now' else 'Save for ' + label}** · {len(g)} film(s)")
        st.markdown(" · ".join(f"{t} ({int(y)})" if not pd.isna(y) else t for t, y in zip(g.clean_title, g.year)))


def star_input(mid: int, key: str) -> None:
    """1-5 star widget bound to st.session_state.ratings (clicking the same star again clears it)."""
    cur = st.session_state.ratings.get(mid)
    val = st.feedback("stars", key=key, default=None if cur is None else int(cur) - 1)
    if val is None:
        st.session_state.ratings.pop(mid, None)
    else:
        st.session_state.ratings[mid] = float(val + 1)


# --------------------------------------------------------------------------- state
art = artifacts()
if "ratings" not in st.session_state:
    st.session_state.ratings = {}  # movieId -> stars

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎬 CineCompass")
    st.caption("What to watch — and when.")
    mode = st.radio("Who's watching?", ["I'm new: rate a few movies", "Demo: an existing MovieLens user"], index=0)
    st.divider()
    st.subheader("Viewing context")
    today = st.date_input("Date", value=date.today(), help="Seasonal suggestions are relative to this date.")
    ctx_label = st.radio("When are you watching?", ["Just browsing", "Tonight (weeknight)", "This weekend"], index=0)
    short = st.checkbox("Short on time", help="MovieLens has no runtimes; films that ≥2 users tagged 'long'/'too long' are pushed down.")
    context = "short_on_time" if short else {"Just browsing": "any", "Tonight (weeknight)": "weeknight", "This weekend": "weekend"}[ctx_label]
    season = st.toggle("Favor what's in season now", value=True)
    st.divider()
    st.subheader("Filters")
    all_genres = sorted({g for gl in art.meta["genre_list"] for g in gl} - {"IMAX"})
    genres = st.multiselect("Genres", all_genres)
    decades = st.slider("Decade", 1920, 2020, (1920, 2020), step=10)
    gems = st.slider("Hidden gems", 0.0, 1.0, 0.0, 0.25, help="0 = no adjustment · 1 = strongly favor less-rated titles")
    n_show = st.select_slider("How many", [6, 9, 12, 18, 24], value=12)

filters = core.Filters(genres=genres, decade_range=decades, hidden_gems=gems, season_boost=season)

# --------------------------------------------------------------------------- main
profile = None
if mode.startswith("Demo"):
    st.header("Recommendations for a MovieLens user")
    c1, c2 = st.columns([1, 3])
    uid = c1.number_input("User ID (1–200,948)", min_value=1, max_value=200_948, value=st.session_state.get("uid", 3), step=1)
    hist_df = core.load_user_history(int(uid))
    hist_df = hist_df[hist_df.movieId.isin(art.meta.index)]
    history = dict(zip(hist_df.movieId.astype(int), hist_df.rating.astype(float)))
    profile = w.personal_profile(hist_df, art.wtw["genres"])
    liked = hist_df.sort_values(["rating", "timestamp"], ascending=[False, False]).head(8)
    c2.markdown(f"**{len(hist_df):,} ratings.** Recent favorites: " + ", ".join(art.meta.loc[liked.movieId, "clean_title"].tolist()))
else:
    st.header("Tell us a few movies you love")
    st.caption(f"Rate at least {core.MIN_RATINGS_FOR_RECS} movies — recommendations appear instantly (no account, nothing stored).")
    # quick start: example taste profiles (also reachable as ?persona=pixar|horror|scifi|romcom)
    qp = st.query_params.get("persona")
    if qp in personas.KEYS and not st.session_state.get("_persona_loaded"):
        st.session_state.ratings = personas.history_for(art.meta, personas.KEYS[qp])
        st.session_state._persona_loaded = True
    st.write("Or try an example profile:")
    pcols = st.columns(len(personas.PERSONAS))
    for c, (key, name) in zip(pcols, personas.KEYS.items()):
        if c.button(name, key=f"p_{key}", use_container_width=True):
            st.session_state.ratings = personas.history_for(art.meta, name)
            for k in [k for k in st.session_state if str(k).startswith(("g", "s")) and str(k)[1:].isdigit()]:
                del st.session_state[k]
            st.rerun()
    q = st.text_input("🔍 Search any movie", placeholder="e.g. Inception, Toy Story, Amélie …")
    if q:
        hits = core.search_titles(art, q)
        for mid, r in hits.iterrows():
            a, b = st.columns([4, 2])
            a.write(f"**{r['clean_title']}** ({'' if pd.isna(r['year']) else int(r['year'])}) · {r['genres'].replace('|', ', ')}")
            with b:
                star_input(int(mid), f"s{mid}")
    with st.expander("…or pick from popular titles", expanded=len(st.session_state.ratings) < core.MIN_RATINGS_FOR_RECS):
        grid = onboarding(0)
        st.caption("Click the stars for movies you have seen; skip the rest.")
        cols = st.columns(4)
        for i, (mid, r) in enumerate(grid.iterrows()):
            with cols[i % 4]:
                st.markdown(f"<div style='min-height:2.6em;font-size:.92rem'><b>{html.escape(r['clean_title'])}</b> ({int(r['year'])})</div>", unsafe_allow_html=True)
                star_input(int(mid), f"g{mid}")
    history = dict(st.session_state.ratings)
    n_r = len(history)
    st.progress(min(1.0, n_r / core.MIN_RATINGS_FOR_RECS), text=f"{n_r} rated" + ("" if n_r >= core.MIN_RATINGS_FOR_RECS else f" — {core.MIN_RATINGS_FOR_RECS - n_r} more to go"))
    if n_r and st.button("Clear my ratings"):
        st.session_state.ratings = {}
        st.rerun()

if len(history) >= (1 if mode.startswith("Demo") else core.MIN_RATINGS_FOR_RECS):
    recs = core.recommend(art, history, n=n_show, filters=filters, today=today, context=context, profile=profile)
    if recs.empty:
        st.warning("No movies match these filters — try widening the genre or decade.")
    else:
        tab1, tab2, tab3 = st.tabs(["🎯 For you", "🗓️ Your movie calendar", "ℹ️ How it works"])
        with tab1:
            render_grid(recs)
        with tab2:
            st.caption("Recommendations grouped by the suggested time to watch.")
            calendar_view(recs, today)
        with tab3:
            st.markdown(
                f"""
**What:** a collaborative-filtering model (LensKit, `{art.family}`) trained on the MovieLens 32M ratings scores every film
from *your* ratings — no retraining needed. **Fit** compares the model's score for a film with its score for the films you rated 4★ or higher
(Excellent ≥ 90% of that level, Strong ≥ 70%, Good ≥ 50%).
**Because you liked** names the film you rated highly that is most similar in the model's item space.

**When:** timing suggestions come only from signals that held up in an out-of-time test (2019–2023):
films tagged *christmas* or *halloween* by users, and films whose ratings rise in the same month year after year.
Weekend/weeknight hints compare a film's weekend share of ratings with the overall share.

**Limitation:** a rating timestamp is when someone *rated* a movie, not necessarily when they *watched* it.
Only ratings from returning sessions (not the sign-up burst) of films at least two years old are used as evidence.
"""
            )
else:
    st.info("Rate a few movies to unlock recommendations.")
