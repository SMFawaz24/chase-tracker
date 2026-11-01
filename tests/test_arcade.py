"""Tests for the arcade generator. No network, no API key: the LLM is replaced by a scripted fake.

Run with:  python -m pytest -q tests/test_arcade.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "arcade"))

import llm  # noqa: E402
from builders import quiz_items  # noqa: E402
from enrich import mine_quiz, phrase_quiz, write_clues  # noqa: E402
from facts import Data  # noqa: E402
from grounding import leaks_answer, leaks_name, name_pattern, numbers_grounded  # noqa: E402
from llm import LLMClient, LLMError, MockLLM, parse_json  # noqa: E402
from sqlguard import SQLGuardError, run_sql, validate_sql  # noqa: E402


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    """Two small seasons (32 matches each) of invented ball-by-ball data."""
    rng = np.random.default_rng(1)
    teams = ["Team A", "Team B", "Team C", "Team D"]
    squads = {t: [f"{c} {s}" for c, s in zip("ABC", ("Zorvak", "Mendel", "Quillan"))] for t in teams}
    for i, t in enumerate(teams):  # unique names per team
        squads[t] = [f"{n} {t[-1]}" for n in ("A Zorvak", "B Mendel", "C Quillan", "D Tarkov")]
    venues = ["Ground 1", "Ground 1", "Ground 1", "Ground 2", "Ground 3", "Ground 4"]
    rows, mid = [], 1
    for season in ("2020", "2021"):
        for m in range(32):
            a, b = rng.choice(teams, 2, replace=False)
            venue, date = venues[m % len(venues)], pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=m)
            totals = {}
            for inn, (bat, bowl) in enumerate([(a, b), (b, a)], 1):
                total = 0
                for ball in range(24):
                    r, w = int(rng.choice([0, 1, 2, 4, 6])), int(rng.random() < 0.08)
                    total += r
                    rows.append(dict(match_id=str(mid), date=date, season=season, venue=venue, innings_no=inn, batting_team=bat,
                                     bowling_team=bowl, over=ball // 6, ball_idx=ball % 6, batter=squads[bat][ball % 4],
                                     bowler=squads[bowl][(ball + 1) % 4], runs_batter=r, runs_extras=0, runs_total=r, is_legal=True,
                                     is_wicket=w, wicket_kind=(str(rng.choice(["bowled", "run out"])) if w else None),
                                     player_out=squads[bat][ball % 4] if w else None,
                                     target=(totals[1] + 1) if inn == 2 else None, winner=None))
                totals[inn] = total
            winner = b if totals[2] > totals[1] else a
            for r in rows[-48:]:
                r["winner"] = winner
            mid += 1
    path = tmp_path_factory.mktemp("data") / "deliveries.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def data(parquet):
    return Data(parquet)


# ---------------------------------------------------------------------------------------------- grounding
def test_grounding_checks():
    assert numbers_grounded("He scored 1,057 runs in 2015.", {"1057", "2015"})
    assert not numbers_grounded("He scored 1,058 runs in 2015.", {"1057", "2015"})
    assert numbers_grounded("No numbers here", set())
    assert leaks_answer("Was it V Kohli who did it?", "v kohli")
    assert leaks_name("This opener, Kohli, is famous", "V Kohli")
    assert not leaks_name("This opener scored plenty", "V Kohli")
    assert name_pattern("V Kohli") == "V K____"
    assert name_pattern("AB de Villiers") == "AB d_ V_______"


# ---------------------------------------------------------------------------------------------- SQL guard
GOOD = "SELECT batter AS label, SUM(runs_batter) AS value FROM deliveries GROUP BY batter ORDER BY value DESC LIMIT 5"


@pytest.mark.parametrize("sql", [
    "DROP TABLE deliveries",
    "SELECT 1; DROP TABLE deliveries",
    "SELECT * FROM read_parquet('/etc/passwd')",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM other_table",
    "SELECT * FROM main.deliveries",
    "SELECT getenv('HOME')",
    "COPY deliveries TO '/tmp/x.csv'",
    "ATTACH '/tmp/x.db'",
    "INSERT INTO deliveries VALUES (1)",
    "SELECT * FROM deliveries LIMIT 1000",
    "this is not sql",
])
def test_sqlguard_rejects(sql):
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


def test_sqlguard_accepts_and_bounds():
    assert "LIMIT 5" in validate_sql(GOOD)
    assert "LIMIT 50" in validate_sql("SELECT batter FROM deliveries GROUP BY batter")  # limit added
    cte = "WITH t AS (SELECT batter, SUM(runs_batter) AS r FROM deliveries GROUP BY batter) SELECT batter AS label, r AS value FROM t ORDER BY r DESC LIMIT 5"
    assert validate_sql(cte)


def test_database_is_locked_even_if_the_guard_were_bypassed(data):
    with pytest.raises(Exception):
        data.con.execute("SELECT * FROM read_parquet('/etc/passwd')")
    with pytest.raises(Exception):
        data.con.execute("SET enable_external_access=true")


def test_slow_queries_are_interrupted(data):
    with pytest.raises(Exception):
        run_sql(data.con, "SELECT COUNT(*) FROM deliveries a, deliveries b, deliveries c", timeout=0.5)


# ---------------------------------------------------------------------------------------------- facts are true
def test_quiz_answers_match_independent_pandas_calculation(data, parquet):
    df = pd.read_parquet(parquet)
    items = quiz_items(data, random.Random(0))
    runs_items = [q for q in items if q["topic"] == "season_runs"]
    assert runs_items
    for q in runs_items:
        season = q["id"].split("-")[-1]
        truth = df[(df.season == season) & (df.innings_no <= 2)].groupby("batter")["runs_batter"].sum().idxmax()
        assert q["options"][q["answer"]] == truth
        assert len(set(q["options"])) == 4


def test_run_out_is_not_credited_to_the_bowler(data, parquet):
    df = pd.read_parquet(parquet)
    credited = df[df.is_wicket == 1]
    credited = credited[credited.wicket_kind != "run out"].groupby("bowler").size().sum()
    assert sum(v for rows in data.season_leaders("wickets", top=999).values() for _, v in rows) == credited


# ---------------------------------------------------------------------------------------------- LLM stages
def test_phrase_quiz_keeps_only_grounded_rewrites(data):
    items = [q for q in quiz_items(data, random.Random(0)) if q["q"].startswith("Who")][:4]
    assert len(items) == 4
    originals = [i["q"] for i in items]
    rewrites = [
        lambda it: {"question": it["q"].replace("Who", "Which player", 1), "fun_fact": "A big season for the winner."},  # good
        lambda it: {"question": f"Was it {it['_correct']}? " + it["q"], "fun_fact": "x" * 20},  # leaks the answer
        lambda it: {"question": it["q"][:-1] + " with over 9999 of them?", "fun_fact": "x" * 20},  # invented number
        lambda it: "not even a dict",  # malformed reply
    ]
    state = {"i": 0}
    client = MockLLM(lambda system, user: rewrites[state["i"]](items[state["i"]]))
    stats = {}
    for i, it in enumerate(items):
        state["i"] = i
        phrase_quiz(client, [it], 1, stats)
    assert [i["source"] for i in items] == ["llm", "template", "template", "template"]
    assert items[0]["q"].startswith("Which player") and "fun_fact" in items[0]
    assert [i["q"] for i in items[1:]] == originals[1:]
    assert stats["phrase"] == {"attempted": 4, "accepted": 1, "errors": 0}


def test_clues_never_see_the_name_and_bad_clues_are_rejected(data):
    from builders import player_items

    players = player_items(data, random.Random(0), 3)
    seen_prompts = []

    def handler(system, user):
        seen_prompts.append(user)
        facts = json.loads(user)["facts"]
        if len(seen_prompts) == 1:
            return {"clues": [f"Clue {'abcde'[i]}, based on this fact: {f}" for i, f in enumerate(facts)]}  # grounded
        if len(seen_prompts) == 2:
            return {"clues": ["He scored 99999 runs"] + facts[1:]}  # invented number
        return {"clues": facts[:4]}  # wrong count

    stats = {}
    write_clues(MockLLM(handler), players, 3, stats)
    for p, prompt in zip(players, seen_prompts):
        assert p["answer"] not in prompt and p["answer"].split()[1] not in prompt, "the model must never be shown the player's name"
    assert [p["source"] for p in players] == ["llm", "template", "template"]
    assert players[0]["clues"][-1].startswith("The name looks like this")
    assert stats["clues"]["accepted"] == 1


def test_mine_quiz_executes_guarded_sql_and_rejects_bad_proposals(data, parquet):
    df = pd.read_parquet(parquet)
    good = ("SELECT batter AS label, SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END) AS value FROM deliveries "
            "WHERE innings_no <= 2 AND over < 6 GROUP BY batter ORDER BY value DESC LIMIT 5")
    proposals = [
        {"title": "good", "question": "Who has hit the most sixes in the powerplay?", "sql": good},
        {"title": "drop", "question": "Which table should we drop?", "sql": "DROP TABLE deliveries"},
        {"title": "ascending", "question": "Who has scored the fewest runs overall?",
         "sql": "SELECT batter AS label, SUM(runs_batter) AS value FROM deliveries GROUP BY batter ORDER BY value ASC LIMIT 5"},
        {"title": "invented number", "question": "Who has hit more than 777 sixes in the powerplay?", "sql": good},
        {"title": "one column", "question": "Which batter is first alphabetically?", "sql": "SELECT batter FROM deliveries GROUP BY batter LIMIT 5"},
    ]
    stats = {}
    out = mine_quiz(MockLLM(lambda s, u: {"queries": proposals}), data, random.Random(0), 5, [], stats)
    assert len(out) == 1
    truth = df[(df.innings_no <= 2) & (df.over < 6) & (df.runs_batter == 6)].groupby("batter").size().sort_values(ascending=False)
    assert out[0]["options"][out[0]["answer"]] == truth.index[0]
    assert out[0]["source"] == "llm-sql" and "LIMIT 5" in out[0]["sql"]


# ---------------------------------------------------------------------------------------------- LLM client
def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! ```json\n{"a": 1}\n``` hope that helps') == {"a": 1}
    assert parse_json('Here: {"a": [1, 2]} done') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json("no json at all")


def test_client_retries_rate_limits_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.text = code, body, "limited"

        def json(self):
            return self._b

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            return Resp(429)
        return Resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    client = LLMClient("groq", cache_path=tmp_path / "c.json", delay=0)
    assert client.json("s", "u") == {"ok": True}
    assert len(calls) == 2  # one 429, one success
    assert client.json("s", "u") == {"ok": True}
    assert len(calls) == 2  # second call came from the cache
    assert LLMClient("groq", cache_path=tmp_path / "c.json", delay=0).json("s", "u") == {"ok": True}  # survives a restart


def test_client_explains_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLMClient("groq")
