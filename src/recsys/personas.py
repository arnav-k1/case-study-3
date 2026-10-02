"""Example taste profiles: used for persona validation and the app's quick-start buttons."""
from __future__ import annotations

import pandas as pd

PERSONAS: dict[str, list[tuple[str, int]]] = {
    "Pixar & animation fan": [("Toy Story", 1995), ("Finding Nemo", 2003), ("Monsters, Inc.", 2001), ("The Incredibles", 2004), ("Up", 2009),
                              ("WALL·E", 2008), ("Ratatouille", 2007), ("Inside Out", 2015), ("Coco", 2017), ("Toy Story 3", 2010)],
    "1970s–80s horror fan": [("Halloween", 1978), ("The Shining", 1980), ("The Texas Chainsaw Massacre", 1974), ("A Nightmare on Elm Street", 1984),
                             ("The Thing", 1982), ("Alien", 1979), ("The Exorcist", 1973), ("Poltergeist", 1982), ("The Evil Dead", 1981), ("Friday the 13th", 1980)],
    "Nolan-style sci-fi fan": [("Inception", 2010), ("Interstellar", 2014), ("The Prestige", 2006), ("Memento", 2000), ("The Matrix", 1999),
                               ("Arrival", 2016), ("Blade Runner 2049", 2017), ("Tenet", 2020), ("The Dark Knight", 2008), ("Minority Report", 2002)],
    "Romantic-comedy fan": [("When Harry Met Sally...", 1989), ("Notting Hill", 1999), ("Pretty Woman", 1990), ("Sleepless in Seattle", 1993),
                            ("You've Got Mail", 1998), ("Love Actually", 2003), ("10 Things I Hate About You", 1999), ("Crazy, Stupid, Love.", 2011),
                            ("The Proposal", 2009), ("Four Weddings and a Funeral", 1994)],
}

KEYS = {"pixar": "Pixar & animation fan", "horror": "1970s–80s horror fan", "scifi": "Nolan-style sci-fi fan", "romcom": "Romantic-comedy fan"}


def resolve(meta: pd.DataFrame, title: str, year: int) -> int:
    m = meta[(meta.year == year) & (meta.clean_title.str.lower() == title.lower())]
    if len(m) == 0:
        m = meta[(meta.year == year) & meta.clean_title.str.lower().str.startswith(title.lower()[:12])]
    if len(m) == 0:
        raise KeyError(f"{title} ({year}) not found")
    return int(m.sort_values("n_ratings", ascending=False).index[0])


def history_for(meta: pd.DataFrame, persona: str, stars: float = 5.0) -> dict[int, float]:
    return {resolve(meta, t, y): stars for t, y in PERSONAS[persona]}
