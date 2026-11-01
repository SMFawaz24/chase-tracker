# Cricket Arcade

Four mini-games (Quick quiz, Higher or lower, Guess the player, Chase predictor) built from IPL ball-by-ball data.
Live page: `/arcade/` on your chase-tracker site.

## The design: the LLM proposes, SQL verifies, validators gate

An LLM never states a fact that reaches a player. Every answer comes from a DuckDB query over `deliveries.parquet`.

| Stage | What the LLM does | What checks it |
|---|---|---|
| Rephrase | Rewrites a template question and adds a fun fact | Every number must appear in the verified facts; the answer must not leak; the question must still ask the same thing |
| Clues | Turns 5 verified facts into clues (never sees the player's name) | Numbers grounded in the facts; the name must not appear in any clue |
| Mine | Writes new SQL for a trivia question | sqlglot guard (one SELECT, only the `deliveries` table, no file or system functions), DuckDB locked down, 10s timeout, strict ranking, 4 distinct answers, question grounded |

Anything rejected is discarded and the template version stays. A bad model lowers variety, never correctness.
The bank is built offline into `static/arcade/bank.json`, so the live site has no API cost and no keys.

## Setup

```powershell
pip install -r arcade/requirements.txt
python src/ingest.py          # re-run once: adds wicket_kind and player_out columns
python arcade/build_bank.py   # template mode, no key needed
```

Commit `static/arcade/bank.json` and push; Vercel redeploys and the page is live at `/arcade/`.

## LLM mode (needs a free key)

1. Get a key: Groq (console.groq.com) or Google AI Studio (aistudio.google.com).
2. PowerShell, current window only:
```powershell
$env:GROQ_API_KEY = "your-key"        # or $env:GEMINI_API_KEY
python arcade/build_bank.py --mode llm --provider groq
```
3. The run prints validation pass rates, for example `LLM mine: 14/24 passed validation`. Replies are cached in `arcade/.cache`, so re-runs are free. Add `.cache/` to `.gitignore` and never commit the key.

## Tests

```powershell
pytest tests/test_arcade.py
```
24 tests cover grounding, the SQL guard (DROP, multi-statement, file reads, timeouts), quiz answers against independent pandas, and the LLM stages with a mock model.

## Resume bullet (fill in your real numbers after the llm run)

> Built Cricket Arcade, an LLM-powered IPL trivia app over 295K deliveries: LLM-generated SQL guarded by an AST validator and a locked-down DuckDB, with grounding checks that discarded X% of model output, so every served answer is verified against the data.
