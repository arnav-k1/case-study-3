"""Persona sanity checks (validation): synthetic cold-start users -> top-10 lists.

Each persona rates 10 real movies 5 stars. Recommendations come from the final
app model via the same code path as the app (app_core.recommend), with season
boost OFF so the lists reflect pure taste; the when-to-watch suggestion is shown
for the date 2026-09-26.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import app_core as core  # noqa: E402
from recsys import data  # noqa: E402
from recsys.personas import PERSONAS, history_for  # noqa: E402

TODAY = date(2026, 9, 26)


def main() -> None:
    art = core.load_artifacts()
    rows = []
    for persona, films in PERSONAS.items():
        hist = history_for(art.meta, persona)
        recs = core.recommend(art, hist, n=10, filters=core.Filters(season_boost=False), today=TODAY)
        for _, r in recs.iterrows():
            rows.append({
                "persona": persona, "rank": int(r["rank"]), "movie": f"{r.clean_title} ({int(r.year)})",
                "genres": ", ".join(r.genre_list), "because_you_liked": r.because, "when": r.headline,
            })
        print(f"\n== {persona}")
        print(recs[["rank", "clean_title", "year", "because", "headline"]].to_string(index=False))
    out = pd.DataFrame(rows)
    out.to_csv(data.ROOT / "reports" / "tables" / "personas.csv", index=False)


if __name__ == "__main__":
    main()
