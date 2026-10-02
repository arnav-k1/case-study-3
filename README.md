# CineCompass — MovieLens 32M recommender with "when to watch" suggestions (READ ME GENERATED WITH AI)

A movie recommender built with **LensKit for Python** on **MovieLens 32M**. It compares seven recommendation models, suggests when to watch each film, and serves recommendations through a Streamlit app.

## Project layout

```
app/app.py          Streamlit app
src/recsys/         data loading, models, evaluation, when-to-watch, explanations, app logic
scripts/            eda · validate_when_to_watch · run_experiments · make_figures · train_final · personas
tests/              pytest checks
reports/            figures/, tables/, literature_review.md, eda_notes.md
assignment/         original assignment files and templates
DECISIONS.md        design decisions and their reasons
```

## How to run (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

.venv\Scripts\python -m recsys.data                    # download MovieLens 32M, verify, convert to Parquet
.venv\Scripts\python scripts\eda.py                     # exploratory analysis
.venv\Scripts\python scripts\validate_when_to_watch.py  # when-to-watch model + out-of-time check
.venv\Scripts\python scripts\run_experiments.py         # model tuning and comparison (~45 min)
.venv\Scripts\python scripts\make_figures.py            # comparison figures
.venv\Scripts\python scripts\train_final.py --family implicitmf --params "{\"embedding_size\": 128, \"weight\": 10.0, \"regularization\": 0.1, \"epochs\": 10}"
.venv\Scripts\python scripts\personas.py                # persona sanity check
.venv\Scripts\python -m pytest                          # tests
.venv\Scripts\streamlit run app\app.py                  # app at http://localhost:8501
```

The dataset is not included in this project; the first command downloads it from GroupLens.

## Key results (5,000 test users, last 5 ratings held out)

| Model                         | NDCG@10 [95% CI]     | Recall@10 | Coverage |
| ----------------------------- | -------------------- | --------- | -------- |
| User k-NN (implicit)          | 0.060 [0.056, 0.065] | 0.089     | 6.5%     |
| **Implicit MF (used in app)** | 0.058 [0.055, 0.063] | 0.087     | 15.3%    |
| Item k-NN (implicit)          | 0.053                | 0.077     | 3.9%     |
| Most popular                  | 0.037                | 0.051     | 1.2%     |

When-to-watch was tested out of time (fit ≤ 2018, tested on 2019–2023). Christmas-tagged films received 1.74× their usual December share of ratings, and Halloween-tagged films 1.76× their usual October share. One-off seasonal peaks did not repeat (0.98×), so they are not used.
