"""FastAPI service: win-probability predictions + per-match curves + dashboard.

Run from the project root:
    uvicorn src.api:app --reload      (or: cd src && uvicorn api:app --reload)
Then open http://127.0.0.1:8000
"""
import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, add_state_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "winprob.json"
VENUES_PATH = ROOT / "models" / "venues.json"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "processed" / "deliveries.parquet"
STATIC = ROOT / "static"

app = FastAPI(title="Cricket Win-Probability Engine", version="1.0")

model = XGBClassifier()
model.load_model(str(MODEL_PATH))
venue_map: dict = json.loads(VENUES_PATH.read_text())
metrics: dict = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

# Chase data only (innings 2) for the match explorer
_df = pd.read_parquet(DATA_PATH)
_chases = _df[(_df["innings_no"] == 2) & _df["target"].notna()].sort_values(["match_id", "over", "ball_idx"]).copy()
_chases["match_id"] = _chases["match_id"].astype(str)
_match_index = (
    _chases.groupby("match_id")
    .agg(
        date=("date", "first"),
        season=("season", "first"),
        venue=("venue", "first"),
        chasing=("batting_team", "first"),
        defending=("bowling_team", "first"),
        target=("target", "first"),
        winner=("winner", "first"),
    )
    .reset_index()
    .sort_values("date", ascending=False)
)


class PredictIn(BaseModel):
    target: int = Field(..., ge=1, description="Runs the chasing team must reach")
    runs_scored: int = Field(..., ge=0)
    balls_bowled: int = Field(..., ge=0, le=119, description="Legal balls bowled so far")
    wickets_fallen: int = Field(..., ge=0, le=9)
    venue: str | None = None


def _predict(rows: pd.DataFrame) -> list[float]:
    return model.predict_proba(rows[FEATURES])[:, 1].tolist()


@app.get("/health")
def health():
    return {"status": "ok", "matches": int(len(_match_index)), "metrics": metrics}


@app.post("/predict")
def predict(body: PredictIn):
    runs_left = body.target - body.runs_scored
    balls_left = 120 - body.balls_bowled
    if runs_left <= 0:
        return {"win_probability": 1.0, "note": "target already reached"}
    row = pd.DataFrame(
        [
            {
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_left": 10 - body.wickets_fallen,
                "crr": body.runs_scored * 6 / max(body.balls_bowled, 1),
                "rrr": runs_left * 6 / max(balls_left, 1),
                "target": body.target,
                "venue_code": venue_map.get(body.venue, -1) if body.venue else -1,
            }
        ]
    )
    p = _predict(row)[0]
    return {"win_probability": round(p, 4), "chasing_team_win_pct": round(p * 100, 1)}


@app.get("/venues")
def venues():
    return sorted(venue_map.keys())


@app.get("/matches")
def matches(season: str | None = None, limit: int = Query(200, le=2000)):
    m = _match_index
    if season:
        m = m[m["season"] == season]
    out = m.head(limit).copy()
    out["date"] = out["date"].astype(str).str[:10]
    return out.to_dict(orient="records")


@app.get("/seasons")
def seasons():
    return sorted(_match_index["season"].unique().tolist(), reverse=True)


@app.get("/match/{match_id}/curve")
def curve(match_id: str):
    d = _chases[_chases["match_id"] == match_id].copy()
    if d.empty:
        raise HTTPException(404, "match not found (or no chase)")
    d = add_state_features(d, venue_map)
    d["win_prob"] = _predict(d)
    # Terminal states are certain: target reached -> 1; out of wickets/balls short of target -> 0.
    # (The model is only trained on live states, so don't let it extrapolate here.)
    d.loc[d["runs_left"] <= 0, "win_prob"] = 1.0
    d.loc[(d["runs_left"] > 0) & ((d["wickets_left"] <= 0) | (d["balls_left"] <= 0)), "win_prob"] = 0.0
    chasing = d["batting_team"].iloc[0]
    won = d["winner"].iloc[0] == chasing
    points = [
        {
            "ball": int(i),
            "over": f"{int(r.over)}.{int(r.ball_idx) + 1}",
            "win_prob": round(float(r.win_prob), 4),
            "runs": int(r.runs_scored),
            "wickets": int(r.wickets_fallen),
            "balls": int(r.balls_bowled),
            "wicket_fell": bool(r.is_wicket),
        }
        for i, r in enumerate(d.itertuples(index=False), start=1)
    ]
    return {
        "match_id": match_id,
        "chasing": chasing,
        "defending": d["bowling_team"].iloc[0],
        "target": int(d["target"].iloc[0]),
        "winner": d["winner"].iloc[0],
        "chasing_won": bool(won),
        "venue": d["venue"].iloc[0],
        "date": str(d["date"].iloc[0])[:10],
        "points": points,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
