"""Ingest Cricsheet ball-by-ball JSON into a clean deliveries table.

Usage:
    python src/ingest.py --out data/processed/deliveries.parquet
    python src/ingest.py --db-url postgresql://user:pass@host:5432/cricket   # also load to Postgres

Downloads https://cricsheet.org/downloads/ipl_json.zip (run this on your own machine).
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://cricsheet.org/downloads/ipl_json.zip"


def parse_match(match_id: str, data: dict) -> list[dict]:
    info = data["info"]
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")  # None for no result / tie
    rows = []
    teams = info["teams"]
    for inn_no, inn in enumerate(data["innings"], start=1):
        bat = inn["team"]
        bowl = [t for t in teams if t != bat][0]
        target = inn.get("target", {}).get("runs")
        for ov in inn["overs"]:
            for ball_idx, d in enumerate(ov["deliveries"]):
                extras = d.get("extras", {})
                rows.append(
                    {
                        "match_id": match_id,
                        "date": info["dates"][0],
                        "season": str(info.get("season")),
                        "venue": info.get("venue"),
                        "innings_no": inn_no,
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "over": ov["over"],
                        "ball_idx": ball_idx,
                        "batter": d["batter"],
                        "bowler": d["bowler"],
                        "runs_batter": d["runs"]["batter"],
                        "runs_extras": d["runs"]["extras"],
                        "runs_total": d["runs"]["total"],
                        "is_legal": not ("wides" in extras or "noballs" in extras),
                        "is_wicket": int("wickets" in d),
                        "target": target,
                        "winner": winner,
                    }
                )
    return rows


def load_all() -> pd.DataFrame:
    print("Downloading Cricsheet IPL data...")
    z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=120).content))
    rows = []
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        data = json.loads(z.read(name))
        if data["info"].get("gender") != "male":
            continue
        rows.extend(parse_match(Path(name).stem, data))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "match_id", "innings_no", "over", "ball_idx"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["venue", "batter", "bowler"])
    assert (df["runs_total"] == df["runs_batter"] + df["runs_extras"]).all(), "run totals inconsistent"
    print(f"Validated: {before:,} -> {len(df):,} rows, {df['match_id'].nunique():,} matches")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/deliveries.parquet")
    ap.add_argument("--db-url", default=None)
    args = ap.parse_args()

    df = validate(load_all())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")

    if args.db_url:
        from sqlalchemy import create_engine

        df.to_sql("deliveries", create_engine(args.db_url), if_exists="replace", index=False, chunksize=50_000)
        print("Loaded into Postgres table: deliveries")
