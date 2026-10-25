"""Shared feature engineering for training and serving."""
import pandas as pd

FEATURES = ["runs_left", "balls_left", "wickets_left", "crr", "rrr", "target", "venue_code"]


def add_state_features(d: pd.DataFrame, venue_map: dict) -> pd.DataFrame:
    """Expects a single-chase frame sorted by (match_id, over, ball_idx)."""
    g = d.groupby("match_id")
    d["runs_scored"] = g["runs_total"].cumsum()
    d["balls_bowled"] = g["is_legal"].cumsum()
    d["wickets_fallen"] = g["is_wicket"].cumsum()
    d["runs_left"] = d["target"] - d["runs_scored"]
    d["balls_left"] = 120 - d["balls_bowled"]
    d["wickets_left"] = 10 - d["wickets_fallen"]
    d["crr"] = d["runs_scored"] * 6 / d["balls_bowled"].clip(lower=1)
    d["rrr"] = d["runs_left"] * 6 / d["balls_left"].clip(lower=1)
    d["venue_code"] = d["venue"].map(venue_map).fillna(-1).astype(int)
    return d


def build_features(df: pd.DataFrame, venue_map: dict | None = None):
    """Training frame: 2nd-innings balls of completed matches with a result.

    Returns (frame, venue_map). Pass venue_map at serving time to keep codes stable.
    """
    d = df[(df["innings_no"] == 2) & df["winner"].notna() & df["target"].notna()].copy()
    d = d.sort_values(["match_id", "over", "ball_idx"])
    if venue_map is None:
        venue_map = {v: i for i, v in enumerate(sorted(d["venue"].dropna().unique()))}
    d = add_state_features(d, venue_map)
    d["y"] = (d["batting_team"] == d["winner"]).astype(int)
    d = d[(d["balls_left"] > 0) & (d["runs_left"] > 0) & (d["wickets_left"] > 0)]
    return d, venue_map
