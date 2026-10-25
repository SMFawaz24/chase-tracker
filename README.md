# Cricket Win-Probability Engine

Ball-by-ball data pipeline, XGBoost model, FastAPI service and web dashboard that predict the chasing team's win probability after every delivery of an IPL second innings.

## Pipeline
Cricsheet JSON -> parse/validate (Pandas) -> Parquet / PostgreSQL (AWS RDS) -> feature engineering -> XGBoost -> FastAPI -> dashboard

## Run
```bash
pip install -r requirements.txt
python src/ingest.py            # downloads + cleans data -> data/processed/deliveries.parquet
python src/train.py             # trains, prints metrics, saves model + calibration plot
uvicorn src.api:app --reload    # open http://127.0.0.1:8000
```

API endpoints: `POST /predict`, `GET /matches`, `GET /match/{id}/curve`, `GET /seasons`, `GET /health`.

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"target":180,"runs_scored":90,"balls_bowled":60,"wickets_fallen":3}'
```

## Results
- Data: **295,732 deliveries** across **1,243 IPL matches** (Cricsheet), validated with run-total consistency checks
- Training set: 122,857 chase deliveries | Test set: 16,166 deliveries from the held-out **2025 and 2026 seasons**
- Log-loss: **0.576** (constant-prior baseline 0.689)
- Brier score: **0.174** (baseline 0.248)
- Calibration plot: `reports/calibration.png`

## Design decisions
- **Time-based split** (last 2 seasons held out) so the model never sees the future.
- Features: runs left, balls left, wickets left, current run rate, required run rate, target, venue.
- Shared `features.py` is used by both training and serving, so the two cannot drift apart.
- Venue encoding is saved with the model (`models/venues.json`); unseen venues fall back to an "unknown" code.

## Known limitations / next steps
- No batter or bowler strength features yet (the next lift is likely player-level ratings).
- Interrupted (DLS) chases use the adjusted target but not the reduced overs.
- [ ] Load to AWS RDS PostgreSQL (`python src/ingest.py --db-url ...`)
- [ ] Deploy (Render / Railway) + demo GIF
