# Chase Tracker + Cricket Arcade

An IPL data project in two parts, both live:

- **Chase tracker** ([chase.f4w4z.dev](https://chase.f4w4z.dev)): an XGBoost model that gives the chasing team's win probability after every ball of an IPL second innings, with a replay of any match.
- **Cricket Arcade** ([chase.f4w4z.dev/arcade/](https://chase.f4w4z.dev/arcade/)): four trivia mini-games built from the same data, with an LLM pipeline where every fact is verified by SQL before a player sees it.

Both run as static sites. There is no server to keep awake and no API key in production.

## Data

- Source: [Cricsheet](https://cricsheet.org) IPL ball-by-ball JSON
- 295,732 deliveries across 1,243 matches, validated with run-total consistency checks
- Cleaned into Parquet (`data/processed/deliveries.parquet`), with wicket kind and dismissed player included

## Part 1: Chase tracker

**Pipeline:** Cricsheet JSON -> parse and validate (Pandas) -> Parquet -> feature engineering -> XGBoost -> trees exported to JSON -> evaluated in the browser.

**Features:** runs left, balls left, wickets left, current run rate, required run rate, target, venue.

**Results** (test set: 16,166 deliveries from the held-out 2025 and 2026 seasons; train: 122,857 chase deliveries)

| Metric | Model | Constant-prior baseline |
|---|---|---|
| Log-loss | **0.576** | 0.689 |
| Brier score | **0.174** | 0.248 |

Calibration plot: `reports/calibration.png`.

**Design decisions**
- **Time-based split.** The last two seasons are held out so the model never sees the future.
- **No server.** The tree ensemble is exported to JSON and evaluated in the browser. Inputs and thresholds are rounded to float32 to match XGBoost, and predictions agree with Python to within 3e-7.
- **One feature module.** `src/features.py` is used by training, serving and export, so they cannot drift apart.
- **Saved venue encoding** (`models/venues.json`); unseen venues fall back to an "unknown" code.
- **Terminal states are exact.** A chase with no runs left is 100%, and one out of wickets or balls is 0%.

## Part 2: Cricket Arcade

Four games: **Quick quiz**, **Higher or lower**, **Guess the player** (progressive clues) and **Chase predictor** (real chases frozen mid-innings; beat the model above, scored with a Brier-style rule).

**LLM proposes, SQL verifies, validators gate.** The LLM never states a fact that reaches a player.

| Stage | What the LLM does | What checks it |
|---|---|---|
| Rephrase | Rewrites a template question and adds a fun fact | Every number must appear in the verified facts; the answer must not leak; the question must still ask the same thing |
| Clues | Turns 5 verified facts into clues; never sees the player's name | Numbers grounded in the facts; the name must not appear in any clue |
| Mine | Writes new SQL for a trivia question | sqlglot AST guard (one SELECT, one table, no file or system functions), locked-down DuckDB, 10 s timeout, strict ranking, 4 distinct answers, question grounded in the result |

Anything rejected is discarded and the template version stays. A weak model can reduce variety but not correctness. Mined questions keep their SQL, shown in the page under "How we know".

The bank is built offline into `static/arcade/bank.json`, so the live site has no runtime LLM cost. Two modes:
- **Template mode** needs no key.
- **LLM mode** works with Groq, Gemini, OpenAI or a local Ollama. Replies are cached on disk, so re-runs are free.

Validation pass rates are printed after every LLM build. Latest run: Counts: {'quiz': 62, 'higher_lower': 102, 'players': 120, 'chase': 240}.

More detail: `arcade/README.md`.

## Run it locally

```powershell
pip install -r requirements.txt
pip install -r arcade/requirements.txt

python src/ingest.py            # download + clean -> data/processed/deliveries.parquet
python src/train.py             # train, print metrics, save model + calibration plot
python src/export_static.py     # export model + every match curve to static/data
python arcade/build_bank.py     # build the arcade question bank (template mode)

python -m http.server -d static # open http://localhost:8000 and http://localhost:8000/arcade/
```

LLM mode (free key from Groq or Google AI Studio; never commit it):

```powershell
$env:GROQ_API_KEY = "your-key"
python arcade/build_bank.py --mode llm --provider groq
```

Tests:

```powershell
pytest tests/test_arcade.py     # 24 tests: SQL guard, grounding, answer correctness vs pandas, LLM stages with a mock model
```

A FastAPI service (`src/api.py`) is kept for local use: `POST /predict`, `GET /matches`, `GET /match/{id}/curve`, `GET /seasons`, `GET /health`.

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"target":180,"runs_scored":90,"balls_bowled":60,"wickets_fallen":3}'
```

## Deploy

Hosted on Vercel from this repo: Root Directory `static`, Framework Preset "Other", no build command. Every push to `main` redeploys. The custom domain is a CNAME to Vercel. Commit `static/data/` and `static/arcade/bank.json`, because the sites read them as plain files.

## Repo layout

```
src/            ingest, features, train, export_static, api
static/         tracker site (index.html, data/) and arcade/ (index.html, bank.json)
arcade/         fact queries, question builders, SQL guard, LLM client, build_bank.py
models/         trained model + venue map + metrics
reports/        calibration plot
tests/          arcade test suite
```

## Known limitations and next steps

- No batter or bowler strength features yet; player-level ratings are the likely next lift.
- Interrupted (DLS) chases use the adjusted target but not the reduced overs.
- Arcade questions cover IPL only, and the bank is rebuilt by hand when new matches are added.
- To do: load the data into PostgreSQL (`python src/ingest.py --db-url ...`) and add a demo GIF at the top of this README.
