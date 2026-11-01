"""Template mode: turn verified facts into game content. No LLM is needed for anything in this file."""
import sys
from pathlib import Path

import pandas as pd

from grounding import allowed_from, name_pattern


def make_mcq(rng, qid, topic, q, correct, distractors, explain, facts, allowed_values, anchors=()):
    seen, clean = {correct}, []
    for d in distractors:
        if d not in seen:
            seen.add(d)
            clean.append(d)
    if len(clean) < 3:
        return None
    options = [correct] + clean[:3]
    rng.shuffle(options)
    return {
        "id": qid, "topic": topic, "q": q, "options": options, "answer": options.index(correct),
        "explain": explain, "source": "template", "facts": facts,
        "_allowed": sorted(allowed_from(allowed_values)), "_correct": correct, "_anchors": [a.lower() for a in anchors],
    }


def _ranked_mcq(rng, qid, topic, q, rows, unit, anchors, extra_allowed=()):
    """rows = [(label, value), ...] best first. Needs a strict winner and 3 real runners-up."""
    if len(rows) < 4 or rows[0][1] == rows[1][1]:
        return None
    facts = [f"{i + 1}. {label}: {value} {unit}" for i, (label, value) in enumerate(rows)]
    explain = f"{rows[0][0]} finished on {rows[0][1]} {unit}, ahead of {rows[1][0]} on {rows[1][1]}."
    return make_mcq(rng, qid, topic, q, rows[0][0], [r[0] for r in rows[1:4]], explain, facts,
                    [v for _, v in rows] + list(extra_allowed), anchors)


def quiz_items(data, rng):
    items = []
    words = {"runs": ("scored the most runs", "runs"), "sixes": ("hit the most sixes", "sixes"),
             "wickets": ("took the most wickets", "wickets")}
    for metric, (phrase, unit) in words.items():
        for season, rows in data.season_leaders(metric).items():
            q = f"Who {phrase} in IPL {season}?"
            items.append(_ranked_mcq(rng, f"q-{metric}-{season}", f"season_{metric}", q, rows, unit, [unit, season], [season]))
        rows = data.career_leaders(metric)
        items.append(_ranked_mcq(rng, f"q-career-{metric}", f"career_{metric}", f"Who has {phrase} in IPL history?",
                                 rows, unit, [unit, "history"]))

    totals = data.top_totals(10)
    if len(totals) >= 2 and totals[0][3] > totals[1][3]:
        team, opp, season, total = totals[0]
        others = [t[0] for t in totals[1:]] + data.teams()
        items.append(make_mcq(
            rng, "q-top-total", "records", "Which team holds the record for the highest innings total in the IPL?", team, others,
            f"{team} scored {total} against {opp} in IPL {season}.",
            [f"{t[0]}: {t[3]} vs {t[1]} ({t[2]})" for t in totals[:5]], [total, season] + [t[3] for t in totals[:5]], ["highest"]))

    chase = data.highest_chase()
    if chase:
        team, opp, season, total = chase
        distractors = [f"{total - k} runs" for k in (4, 8, 12, 17, 22)] + [f"{total + k} runs" for k in (3, 7)]
        rng.shuffle(distractors)
        items.append(make_mcq(
            rng, "q-top-chase", "records",
            "What is the highest total a team has scored while winning a chase in the IPL?", f"{total} runs", distractors,
            f"{team} chased down {opp} in IPL {season}, finishing on {total}.",
            [f"{team} chased in IPL {season}: {total} runs against {opp}"], [total, season], ["chase"]))

    venues = data.venue_top()
    if len(venues) >= 4 and venues[0][1] > venues[1][1]:
        items.append(make_mcq(
            rng, "q-top-venue", "records", "Which ground has hosted the most IPL matches?", venues[0][0],
            [v[0] for v in venues[1:4]], f"{venues[0][0]} has hosted {venues[0][1]} matches.",
            [f"{i + 1}. {v}: {c} matches" for i, (v, c) in enumerate(venues)], [c for _, c in venues], ["ipl"]))
    return [i for i in items if i]


def higher_lower_items(data, rng, n=70):
    pools = []
    for metric in ("runs", "sixes", "wickets"):
        rows = data.season_values(metric)
        if not rows:
            continue
        rows.sort(key=lambda r: -r[2])
        pools.append((metric, "season", [(name, f"IPL {s}", v) for name, s, v in rows[:200]]))
        career = {}
        for name, _, v in rows:
            career[name] = career.get(name, 0) + v
        top = sorted(career.items(), key=lambda kv: -kv[1])[:100]
        pools.append((metric, "career", [(name, "career", v) for name, v in top]))

    items, seen = [], set()
    for metric, kind, pool in pools:
        want, tries = n // 4, 0
        while want > 0 and tries < 4000 and len(pool) >= 2:
            tries += 1
            a, b = rng.sample(pool, 2)
            if a[0] == b[0] or a[2] == b[2]:
                continue
            gap = abs(a[2] - b[2]) / max(a[2], b[2])
            key = frozenset([(a[0], a[1]), (b[0], b[1])])
            if not 0.06 <= gap <= 0.45 or key in seen:
                continue
            seen.add(key)
            items.append({"id": f"hl-{len(items)}", "metric": metric,
                          "a": {"name": a[0], "ctx": a[1], "value": a[2]}, "b": {"name": b[0], "ctx": b[1], "value": b[2]}})
            want -= 1
    rng.shuffle(items)
    return items


def player_items(data, rng, top_n=120):
    items = []
    for p in data.profiles(top_n):
        n_teams = len(p["teams"])
        facts = [
            f"Has played {p['seasons']} IPL seasons, between {p['first_season']} and {p['last_season']}.",
            "Has batted for one IPL team only." if n_teams == 1 else f"Has batted for {n_teams} different IPL teams.",
            f"Career: {p['runs']:,} IPL runs and {p['sixes']} sixes at a strike rate of {p['strike_rate']}.",
            f"Best season: {p['best_season_runs']} runs in IPL {p['best_season']}; highest score {p['highest']} "
            f"against {p['highest_opp']} (IPL {p['highest_season']}).",
            f"Teams: {', '.join(p['teams'])}.",
        ]
        allowed = [p["seasons"], p["first_season"], p["last_season"], n_teams, p["runs"], p["sixes"], p["strike_rate"],
                   p["best_season"], p["best_season_runs"], p["highest"], p["highest_season"]] + p["teams"]
        items.append({"id": f"p-{len(items)}", "answer": p["name"], "clues": facts + [f"The name looks like this: {name_pattern(p['name'])}"],
                      "source": "template", "_facts": facts, "_allowed": sorted(allowed_from(allowed))})
    rng.shuffle(items)
    return items


def chase_items(df, root, rng, n=240):
    """Mid-chase situations for the Chase Predictor, scored against the project's own win-probability model."""
    import json

    from xgboost import XGBClassifier

    root = Path(root)
    model_path, venues_path = root / "models" / "winprob.json", root / "models" / "venues.json"
    if not (model_path.exists() and venues_path.exists()):
        print("  ! models/ not found: skipping the Chase Predictor (run src/train.py first)")
        return []
    sys.path.insert(0, str(root / "src"))
    from features import FEATURES, build_features

    model = XGBClassifier()
    model.load_model(str(model_path))
    d, _ = build_features(df, json.loads(venues_path.read_text()))
    d["p"] = model.predict_proba(d[FEATURES])[:, 1]

    cand = d[d["balls_bowled"].isin([36, 60, 84, 102])].groupby(["match_id", "balls_bowled"]).tail(1)
    cand = cand[cand["p"].between(0.12, 0.88)]
    seed = rng.randrange(10 ** 6)
    cand = cand.sample(frac=1, random_state=seed).groupby("match_id").head(1)
    won, lost = cand[cand["y"] == 1], cand[cand["y"] == 0]
    k = min(len(won), len(lost), n // 2)
    sel = pd.concat([won.head(k), lost.head(k)]).sample(frac=1, random_state=seed)

    fin = df[df["innings_no"] == 2].groupby("match_id").agg(
        final_runs=("runs_total", "sum"), final_wkts=("is_wicket", "sum"), final_balls=("is_legal", "sum"))
    items = []
    for r in sel.itertuples():
        if r.match_id not in fin.index:
            continue
        f = fin.loc[r.match_id]
        items.append({
            "id": str(r.match_id), "season": str(r.season), "date": str(r.date)[:10], "venue": r.venue,
            "chasing": r.batting_team, "defending": r.bowling_team, "target": int(r.target),
            "runs": int(r.runs_scored), "wickets": int(r.wickets_fallen), "balls": int(r.balls_bowled),
            "p": round(float(r.p), 3), "won": bool(r.y),
            "final_runs": int(f["final_runs"]), "final_wickets": int(f["final_wkts"]), "final_balls": int(f["final_balls"]),
        })
    return items
