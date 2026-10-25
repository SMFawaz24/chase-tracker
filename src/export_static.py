"""Export everything the dashboard needs as static files, so it can be hosted anywhere (Vercel, Netlify, GitHub Pages).

Usage (after train.py):
    python src/export_static.py

Writes to static/data/:
    index.json      seasons, match list, venues, model metrics
    model.json      the XGBoost trees in a compact form (the browser runs the model itself)
    curves/<id>.json  ball-by-ball win probability for each chase
"""
import json
import shutil
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES, add_state_features

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "data"


def export_model(model: XGBClassifier, venue_map: dict) -> dict:
    booster = model.get_booster()
    cfg = json.loads(booster.save_config())
    base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        root = json.loads(dump)
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)

        walk(root)
        size = max(nodes) + 1
        f, t, y, n = [-1] * size, [0.0] * size, [-1] * size, [-1] * size
        for nid, node in nodes.items():
            if "leaf" in node:
                t[nid] = node["leaf"]
            else:
                name = node["split"]
                f[nid] = FEATURES.index(name) if name in FEATURES else int(name.lstrip("f"))
                t[nid] = node["split_condition"]
                y[nid], n[nid] = node["yes"], node["no"]
        trees.append({"f": f, "t": t, "y": y, "n": n})
    return {
        "features": FEATURES,
        "base_margin": __import__("math").log(base / (1 - base)),
        "venues": venue_map,
        "trees": trees,
    }


def main():
    model = XGBClassifier()
    model.load_model(str(ROOT / "models" / "winprob.json"))
    venue_map = json.loads((ROOT / "models" / "venues.json").read_text())
    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    df = pd.read_parquet(ROOT / "data" / "processed" / "deliveries.parquet")
    d = df[(df["innings_no"] == 2) & df["target"].notna()].sort_values(["match_id", "over", "ball_idx"]).copy()
    d["match_id"] = d["match_id"].astype(str)
    d = add_state_features(d, venue_map)
    d["win_prob"] = model.predict_proba(d[FEATURES])[:, 1]
    # Terminal states are certain; the model is only trained on live states.
    d.loc[d["runs_left"] <= 0, "win_prob"] = 1.0
    d.loc[(d["runs_left"] > 0) & ((d["wickets_left"] <= 0) | (d["balls_left"] <= 0)), "win_prob"] = 0.0

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "curves").mkdir(parents=True)
    dump = lambda obj, path: path.write_text(json.dumps(obj, separators=(",", ":")))  # noqa: E731

    matches = []
    for mid, g in d.groupby("match_id", sort=False):
        first = g.iloc[0]
        chasing = first["batting_team"]
        winner = first["winner"] if isinstance(first["winner"], str) else None
        info = {
            "id": mid,
            "date": str(first["date"])[:10],
            "season": str(first["season"]),
            "venue": first["venue"],
            "chasing": chasing,
            "defending": first["bowling_team"],
            "target": int(first["target"]),
            "winner": winner,
        }
        matches.append(info)
        dump(
            {
                "ch": chasing, "df": info["defending"], "tg": info["target"], "wn": winner,
                "cw": winner == chasing, "v": info["venue"], "d": info["date"],
                "p": [int(round(x * 10000)) for x in g["win_prob"]],
                "r": g["runs_scored"].astype(int).tolist(),
                "w": g["wickets_fallen"].astype(int).tolist(),
                "b": g["balls_bowled"].astype(int).tolist(),
                "k": g["is_wicket"].astype(int).tolist(),
                "o": [f"{int(o)}.{int(b) + 1}" for o, b in zip(g["over"], g["ball_idx"])],
            },
            OUT / "curves" / f"{mid}.json",
        )

    matches.sort(key=lambda m: m["date"], reverse=True)
    dump(
        {
            "seasons": sorted({m["season"] for m in matches}, reverse=True),
            "matches": matches,
            "venues": sorted(venue_map),
            "metrics": metrics,
        },
        OUT / "index.json",
    )
    dump(export_model(model, venue_map), OUT / "model.json")
    size = sum(p.stat().st_size for p in OUT.rglob("*.json")) / 1e6
    print(f"Exported {len(matches)} chases to {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
