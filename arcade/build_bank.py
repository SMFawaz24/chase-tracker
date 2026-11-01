"""Build the question bank that the static game page reads.

    python arcade/build_bank.py                                  # template mode: no LLM, no key, ready to ship
    python arcade/build_bank.py --mode llm --provider groq       # add LLM phrasing, SQL mining and clues

The page itself only reads static/arcade/bank.json, so the live site has no API costs and no keys.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from builders import chase_items, higher_lower_items, player_items, quiz_items  # noqa: E402
from facts import Data  # noqa: E402


def clean(item):
    return {k: v for k, v in item.items() if not k.startswith("_")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "deliveries.parquet"))
    ap.add_argument("--out", default=str(ROOT / "static" / "arcade" / "bank.json"))
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--provider", default="groq", help="groq | gemini | openai | ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--phrase", type=int, default=60, help="max questions to rephrase with the LLM")
    ap.add_argument("--clues", type=int, default=40, help="max players to write clues for")
    ap.add_argument("--mine", type=int, default=24, help="how many new SQL trivia queries to ask the LLM for")
    ap.add_argument("--max-calls", type=int, default=150)
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between LLM calls (free tiers rate-limit)")
    ap.add_argument("--players", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    print("Loading data...")
    data = Data(a.data)
    df = pd.read_parquet(a.data)

    print("Building verified content...")
    quiz = quiz_items(data, rng)
    hl = higher_lower_items(data, rng)
    players = player_items(data, rng, a.players)
    chase = chase_items(df, ROOT, rng)
    if not data.has_wickets:
        print("  ! no wicket columns in the data: bowling questions skipped (re-run src/ingest.py to add them)")

    stats, llm_info = {}, None
    if a.mode == "llm":
        from enrich import mine_quiz, phrase_quiz, write_clues
        from llm import LLMClient, LLMError

        try:
            client = LLMClient(a.provider, a.model, cache_path=HERE / ".cache" / "llm_cache.json", delay=a.delay, max_calls=a.max_calls)
        except LLMError as e:
            raise SystemExit(f"LLM setup failed: {e}")
        llm_info = {"provider": a.provider, "model": client.model}
        print(f"LLM mode: {a.provider} / {client.model}")
        print("  rephrasing questions...")
        phrase_quiz(client, quiz, a.phrase, stats)
        print("  writing clues...")
        write_clues(client, players, a.clues, stats)
        print("  mining new questions with SQL...")
        quiz += mine_quiz(client, data, rng, a.mine, [q["q"] for q in quiz], stats)

    rng.shuffle(quiz)
    names = [r[0] for r in data.rows("SELECT batter FROM deliveries GROUP BY batter ORDER BY SUM(runs_batter) DESC LIMIT 400")]
    bank = {
        "meta": {"generated": time.strftime("%Y-%m-%d"), "mode": a.mode, "llm": llm_info, "validation": stats,
                 "counts": {"quiz": len(quiz), "higher_lower": len(hl), "players": len(players), "chase": len(chase)}},
        "quiz": [clean(q) for q in quiz], "higher_lower": hl, "players": [clean(p) for p in players],
        "names": names, "chase": chase,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, separators=(",", ":")), encoding="utf-8")

    print(f"\nWrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("Counts:", bank["meta"]["counts"])
    by_source = {}
    for q in bank["quiz"]:
        by_source[q["source"]] = by_source.get(q["source"], 0) + 1
    print("Quiz questions by source:", by_source)
    for k, v in stats.items():
        rate = 100 * v["accepted"] / v["attempted"] if v["attempted"] else 0
        print(f"LLM {k}: {v['accepted']}/{v['attempted']} passed validation ({rate:.0f}%), {v['errors']} API errors")
    return bank


if __name__ == "__main__":
    main()
