"""Data acquisition, preprocessing, and loading for MovieLens 32M.

Raw CSVs live in data/raw/ml-32m/ and are converted once to compact Parquet
files in data/processed/. All downstream code reads the Parquet files.
"""
from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ML_DIR = RAW_DIR / "ml-32m"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

ML32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
EXPECTED_MD5 = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
    "tags.csv": "963bf4fa4de6b8901868fddd3eb54567",
}
EXPECTED_ROWS = {"ratings": 32_000_204, "tags": 2_000_072, "movies": 87_585}

# Time zone used for all hour-of-day / day-of-week analysis. The course is at
# Texas A&M, so US Central is the natural "local" reference; the raw data are UTC
# and MovieLens users are worldwide, so this is a convention, not a user's clock.
LOCAL_TZ = "America/Chicago"

_YEAR_RE = re.compile(r"\((\d{4})(?:[-–]\d{0,4})?\)\s*$")


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(force: bool = False, verify: bool = True) -> Path:
    """Download and extract ml-32m.zip into data/raw/. Returns the extracted dir."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / "ml-32m.zip"
    if force or not zpath.exists():
        print(f"downloading {ML32M_URL} ...")
        urllib.request.urlretrieve(ML32M_URL, zpath)
    if force or not (ML_DIR / "ratings.csv").exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(RAW_DIR)
    if verify:
        for name, md5 in EXPECTED_MD5.items():
            got = _md5(ML_DIR / name)
            if got != md5:
                raise RuntimeError(f"MD5 mismatch for {name}: {got} != {md5}")
        print("MD5 checksums verified")
    return ML_DIR


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def parse_title(title: str) -> tuple[str, float]:
    """Split 'Toy Story (1995)' -> ('Toy Story', 1995). Year is NaN if absent.

    Also moves trailing articles back: 'Matrix, The' -> 'The Matrix'.
    """
    t = title.strip()
    m = _YEAR_RE.search(t)
    year = float(m.group(1)) if m else np.nan
    name = t[: m.start()].strip() if m else t
    # "Shawshank Redemption, The" -> "The Shawshank Redemption"
    am = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Das|Die|Der)$", name)
    if am:
        name = f"{am.group(2)} {am.group(1)}"
    # Also handle "Name, The (Alt Name)" patterns
    am = re.match(r"^(.*), (The|A|An) (\(.*\))$", name)
    if am:
        name = f"{am.group(2)} {am.group(1)} {am.group(3)}"
    return name, year


def preprocess(force: bool = False) -> None:
    """Convert raw CSVs to Parquet with compact dtypes."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    out = PROC_DIR / "ratings.parquet"
    if force or not out.exists():
        print("ratings.csv -> parquet")
        r = pd.read_csv(
            ML_DIR / "ratings.csv",
            dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
            engine="pyarrow",
        )
        r.to_parquet(out, index=False)
        del r

    out = PROC_DIR / "movies.parquet"
    if force or not out.exists():
        print("movies.csv -> parquet")
        m = pd.read_csv(ML_DIR / "movies.csv", dtype={"movieId": np.int32})
        parsed = m["title"].map(parse_title)
        m["clean_title"] = [p[0] for p in parsed]
        m["year"] = pd.array([p[1] for p in parsed], dtype="Float32").astype("Int16")
        m["genres"] = m["genres"].replace("(no genres listed)", "")
        m["genre_list"] = m["genres"].map(lambda g: [x for x in g.split("|") if x])
        links = pd.read_csv(ML_DIR / "links.csv", dtype={"movieId": np.int32, "imdbId": str, "tmdbId": "Int64"})
        m = m.merge(links, on="movieId", how="left")
        m.to_parquet(out, index=False)

    out = PROC_DIR / "tags.parquet"
    if force or not out.exists():
        print("tags.csv -> parquet")
        t = pd.read_csv(
            ML_DIR / "tags.csv",
            dtype={"userId": np.int32, "movieId": np.int32, "tag": str, "timestamp": np.int64},
            keep_default_na=False,
        )
        t["tag_norm"] = t["tag"].str.strip().str.lower()
        t.to_parquet(out, index=False)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ratings(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / "ratings.parquet", columns=columns)


def load_movies() -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / "movies.parquet")


def load_tags() -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / "tags.parquet")


def add_local_time(df: pd.DataFrame, ts_col: str = "timestamp", tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Add month / dow / hour columns (in `tz`) derived from Unix seconds."""
    dt = pd.to_datetime(df[ts_col], unit="s", utc=True).dt.tz_convert(tz)
    df = df.copy()
    df["year_rated"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["dow"] = dt.dt.dayofweek.astype(np.int8)  # Monday=0
    df["hour"] = dt.dt.hour.astype(np.int8)
    return df


# --------------------------------------------------------------------------- #
# Sampling for model comparison
# --------------------------------------------------------------------------- #
def build_sample(
    ratings: pd.DataFrame,
    n_users: int = 25_000,
    min_item_ratings: int = 50,
    min_user_ratings: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Seeded sample for model comparison.

    1. Keep items with >= `min_item_ratings` ratings in the *full* data.
    2. Among users with >= `min_user_ratings` remaining ratings, sample `n_users`
       uniformly at random with a fixed seed.
    """
    item_counts = ratings["movieId"].value_counts()
    keep_items = item_counts.index[item_counts >= min_item_ratings]
    r = ratings[ratings["movieId"].isin(keep_items)]
    user_counts = r["userId"].value_counts()
    eligible = np.sort(user_counts.index[user_counts >= min_user_ratings].to_numpy())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    return r[r["userId"].isin(chosen)].reset_index(drop=True)


if __name__ == "__main__":
    download()
    preprocess()
