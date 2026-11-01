"""LLM stages. The pattern is the same in all three: the model proposes, deterministic checks decide.

  phrase_quiz   rewrite a question in a livelier way + add a fun fact   (facts supplied, answer verified by SQL)
  mine_quiz     write new SQL for a trivia question                      (SQL guarded, executed, ranking verified)
  write_clues   turn 5 verified facts into clues                          (model never sees the player's name)

Anything that fails a check is discarded and the template version stays, so a bad model can lower
the variety of the bank but never its correctness.
"""
import json
import re

from builders import make_mcq
from grounding import allowed_from, extract_numbers, leaks_answer, leaks_name, numbers_grounded
from llm import LLMError
from sqlguard import SQLGuardError, run_sql, validate_sql

PHRASE_SYSTEM = (
    "You write cricket trivia questions. Use ONLY the facts provided. Never add any number, name or date that is "
    "not in the facts. Never reveal the answer in the question, and keep what the question asks about exactly the "
    "same as the default question. Reply as JSON: {\"question\": string ending in '?', under 25 words, "
    "\"fun_fact\": string under 30 words}.")

CLUES_SYSTEM = (
    "You write clues for a guess-the-cricketer game. You get 5 verified facts about one player. Write exactly 5 "
    "clues ordered from vaguest to most revealing, each using ONLY information from the facts, no guessing who "
    "it is, nothing invented. Each clue under 30 words. Reply as JSON: {\"clues\": [5 strings]}.")

MINE_SYSTEM = (
    "You design IPL trivia that a SQL query can verify. You get one table. Propose DIFFERENT questions of the form "
    "'which player/team/venue has the most/highest X?', each with one DuckDB SELECT query that returns exactly two "
    "columns named label (text) and value (number), ordered by value DESC, with LIMIT 5, using only the "
    "deliveries table. Be creative: powerplay, death overs, specific seasons, venues, chases, dot balls. The question "
    "must end with '?', must not contain the answer, and must match what the SQL computes. Reply as JSON: "
    "{\"queries\": [{\"title\": string, \"question\": string, \"sql\": string}]}.")


def _bump(stats, key, ok):
    s = stats.setdefault(key, {"attempted": 0, "accepted": 0, "errors": 0})
    s["attempted"] += 1
    s["accepted"] += int(ok)


def _call(client, stats, key, system, user, temperature=0.4):
    """One LLM call; returns the parsed JSON or None. Stops hammering the API after repeated failures."""
    s = stats.setdefault(key, {"attempted": 0, "accepted": 0, "errors": 0})
    if s["errors"] >= 5:
        return None
    try:
        return client.json(system, user, temperature)
    except LLMError as e:
        s["errors"] += 1
        print(f"  ! LLM error ({key}): {str(e)[:120]}")
        return None


def phrase_quiz(client, items, limit, stats):
    done = 0
    for item in items:
        if done >= limit:
            break
        done += 1
        user = json.dumps({"default_question": item["q"], "answer": item["_correct"], "facts": item["facts"]})
        r = _call(client, stats, "phrase", PHRASE_SYSTEM, user)
        ok = False
        if isinstance(r, dict) and isinstance(r.get("question"), str) and isinstance(r.get("fun_fact"), str):
            q, fun = r["question"].strip(), r["fun_fact"].strip()
            ok = (
                q.endswith("?") and 15 <= len(q) <= 220 and 5 <= len(fun) <= 240
                and not leaks_answer(q, item["_correct"])
                and all(a in q.lower() for a in item["_anchors"])
                and numbers_grounded(q + " " + fun, item["_allowed"])
            )
            if ok:
                item["q"], item["fun_fact"], item["source"] = q, fun, "llm"
        _bump(stats, "phrase", ok)
    return items


def write_clues(client, players, limit, stats):
    done = 0
    for p in players:
        if done >= limit:
            break
        done += 1
        r = _call(client, stats, "clues", CLUES_SYSTEM, json.dumps({"facts": p["_facts"]}))
        ok = False
        clues = r.get("clues") if isinstance(r, dict) else None
        if isinstance(clues, list) and len(clues) == 5 and all(isinstance(c, str) for c in clues):
            clues = [c.strip() for c in clues]
            ok = (
                len(set(clues)) == 5 and all(12 <= len(c) <= 240 for c in clues)
                and all(numbers_grounded(c, p["_allowed"]) for c in clues)
                and not any(leaks_name(c, p["answer"]) for c in clues)
            )
            if ok:
                p["clues"] = clues + [p["clues"][-1]]  # the name-pattern clue stays deterministic
                p["source"] = "llm"
        _bump(stats, "clues", ok)
    return players


def mine_quiz(client, data, rng, n_queries, existing_questions, stats):
    """Ask the model for new trivia queries; keep only those that guard, run, and rank cleanly."""
    out, seen = [], {q.lower() for q in existing_questions}
    asked, batch = 0, 8
    while asked < n_queries:
        want = min(batch, n_queries - asked)
        asked += want
        user = f"{data.schema_doc}\n\nPropose {want} different questions."
        r = _call(client, stats, "mine", MINE_SYSTEM, user, temperature=0.7)
        proposals = r.get("queries") if isinstance(r, dict) else None
        if not isinstance(proposals, list):
            _bump(stats, "mine", False)
            continue
        for prop in proposals:
            item = _verify_proposal(data, rng, prop, seen, len(out))
            _bump(stats, "mine", item is not None)
            if item:
                seen.add(item["q"].lower())
                out.append(item)
    return out


def _verify_proposal(data, rng, prop, seen, idx):
    if not isinstance(prop, dict) or not all(isinstance(prop.get(k), str) for k in ("question", "sql")):
        return None
    q = prop["question"].strip()
    try:
        sql = validate_sql(prop["sql"])
        rows = run_sql(data.con, sql, timeout=10)
    except (SQLGuardError, Exception) as e:  # noqa: BLE001 - any DB error just means "reject this proposal"
        print(f"    rejected SQL: {str(e)[:100]}")
        return None
    if len(rows) < 4 or any(len(r) != 2 for r in rows[:4]):
        return None
    try:
        rows = [(str(label), float(value)) for label, value in rows]
    except (TypeError, ValueError):
        return None
    rows = rows[:5]
    if any(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1)) or rows[0][1] == rows[1][1]:
        return None
    if len({r[0] for r in rows[:4]}) < 4 or q.lower() in seen:
        return None
    fmt = lambda v: int(v) if v == int(v) else round(v, 2)  # noqa: E731
    allowed = allowed_from([fmt(v) for _, v in rows]) | extract_numbers(sql)
    if not (q.endswith("?") and 15 <= len(q) <= 220 and numbers_grounded(q, allowed)
            and not any(leaks_answer(q, label) for label, _ in rows[:4])):
        return None
    facts = [f"{i + 1}. {label}: {fmt(v)}" for i, (label, v) in enumerate(rows)]
    item = make_mcq(rng, f"q-mined-{idx}", "mined", q, rows[0][0], [r[0] for r in rows[1:4]],
                    f"{rows[0][0]} comes out on top with {fmt(rows[0][1])}, ahead of {rows[1][0]} on {fmt(rows[1][1])}.",
                    facts, [], [])
    if item:
        item["source"], item["sql"] = "llm-sql", sql
    return item
