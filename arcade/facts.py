"""Verified facts: every number and name in the games comes from a query in this file (or a guarded
LLM-written query), never from a model's memory."""
import duckdb
import pandas as pd

BOWLER_WICKETS = "('bowled','caught','lbw','stumped','caught and bowled','hit wicket')"

SCHEMA_DOC = """Table deliveries: one row per ball bowled in men's IPL matches.
  match_id TEXT, date DATE, season TEXT (e.g. '2016', '2020/21'), venue TEXT,
  innings_no INT (use 1 and 2 only), batting_team TEXT, bowling_team TEXT,
  over INT (0 to 19; the powerplay is over 0 to 5), ball_idx INT,
  batter TEXT, bowler TEXT, runs_batter INT, runs_extras INT, runs_total INT,
  is_legal BOOL, is_wicket INT (1 when a wicket fell on this ball)"""
SCHEMA_DOC_WICKETS = """,
  wicket_kind TEXT (e.g. 'bowled','caught','lbw','run out'), player_out TEXT"""
SCHEMA_DOC_END = """,
  target INT (the chase target, only on innings 2), winner TEXT (team name; null if no result)
Names look like 'V Kohli' or 'MS Dhoni'. A six is runs_batter = 6."""


class Data:
    def __init__(self, parquet_path):
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE deliveries AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
        cols = {r[0] for r in self.con.execute("DESCRIBE deliveries").fetchall()}
        self.has_wickets = {"wicket_kind", "player_out"} <= cols
        # From here on nothing (including LLM-written SQL) can read files or change settings.
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")

    @property
    def schema_doc(self):
        return SCHEMA_DOC + (SCHEMA_DOC_WICKETS if self.has_wickets else "") + SCHEMA_DOC_END

    def rows(self, sql, params=None):
        return self.con.execute(sql, params or []).fetchall()

    def df(self, sql, params=None):
        return self.con.execute(sql, params or []).df()

    # ---- season / career leaders --------------------------------------------------------------------
    def _metric_sql(self, metric):
        if metric == "runs":
            return "SELECT {g}, batter AS label, SUM(runs_batter)::INT AS value FROM deliveries WHERE innings_no <= 2 GROUP BY ALL"
        if metric == "sixes":
            return ("SELECT {g}, batter AS label, SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END)::INT AS value "
                    "FROM deliveries WHERE innings_no <= 2 GROUP BY ALL")
        if metric == "wickets" and self.has_wickets:
            return (f"SELECT {{g}}, bowler AS label, COUNT(*)::INT AS value FROM deliveries WHERE innings_no <= 2 "
                    f"AND is_wicket = 1 AND wicket_kind IN {BOWLER_WICKETS} GROUP BY ALL")
        return None

    def full_seasons(self, min_matches=30):
        return [s for s, n in self.rows("SELECT season, COUNT(DISTINCT match_id) FROM deliveries GROUP BY season ORDER BY MIN(date)")
                if n >= min_matches]

    def season_leaders(self, metric, top=5):
        """{season: [(label, value), ...]} best first, for seasons with a (nearly) full schedule."""
        sql = self._metric_sql(metric)
        if not sql:
            return {}
        keep = set(self.full_seasons())
        out = {}
        for season, label, value in self.rows(sql.format(g="season")):
            if season in keep:
                out.setdefault(season, []).append((label, value))
        return {s: sorted(v, key=lambda r: -r[1])[:top] for s, v in out.items()}

    def season_values(self, metric):
        """Flat [(label, season, value)] for Higher or Lower."""
        sql = self._metric_sql(metric)
        if not sql:
            return []
        keep = set(self.full_seasons())
        return [(label, season, value) for season, label, value in self.rows(sql.format(g="season")) if season in keep]

    def career_leaders(self, metric, top=5):
        sql = self._metric_sql(metric)
        if not sql:
            return []
        return self.rows(f"SELECT label, SUM(value)::INT AS value FROM ({sql.format(g='season')}) GROUP BY label ORDER BY value DESC LIMIT {top}")

    # ---- team and venue records ---------------------------------------------------------------------
    def top_totals(self, n=10):
        return self.rows(
            "SELECT batting_team, bowling_team, season, SUM(runs_total)::INT AS total FROM deliveries "
            "WHERE innings_no <= 2 GROUP BY match_id, innings_no, batting_team, bowling_team, season "
            f"ORDER BY total DESC LIMIT {n}")

    def highest_chase(self):
        rows = self.rows(
            "SELECT batting_team, bowling_team, season, SUM(runs_total)::INT AS total FROM deliveries "
            "WHERE innings_no = 2 AND winner = batting_team GROUP BY match_id, batting_team, bowling_team, season "
            "ORDER BY total DESC LIMIT 1")
        return rows[0] if rows else None

    def venue_top(self, n=6):
        return self.rows(f"SELECT venue, COUNT(DISTINCT match_id)::INT AS c FROM deliveries GROUP BY venue ORDER BY c DESC LIMIT {n}")

    def teams(self):
        return [r[0] for r in self.rows("SELECT batting_team FROM deliveries GROUP BY batting_team ORDER BY COUNT(DISTINCT match_id) DESC")]

    # ---- player profiles ----------------------------------------------------------------------------
    def profiles(self, top_n=120):
        inn = self.df(
            "SELECT match_id, season, batter, ANY_VALUE(bowling_team) AS opp, MIN(date) AS date, "
            "SUM(runs_batter)::INT AS runs, COUNT(*) FILTER (WHERE is_legal)::INT AS balls, "
            "SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END)::INT AS sixes "
            "FROM deliveries WHERE innings_no <= 2 GROUP BY match_id, season, batter")
        keep = set(self.full_seasons())
        teams = self.df("SELECT batter, batting_team, MIN(date) AS d FROM deliveries WHERE innings_no <= 2 "
                        "GROUP BY batter, batting_team ORDER BY d")
        out = []
        for name, g in inn.groupby("batter"):
            runs, balls = int(g["runs"].sum()), int(g["balls"].sum())
            by_season = g.groupby("season")["runs"].sum()
            by_season = by_season[by_season.index.isin(keep)] if keep else by_season
            if by_season.empty or balls < 50:
                continue
            best_season = by_season.idxmax()
            hs = g.loc[g["runs"].idxmax()]
            seasons = sorted(g["season"].unique())
            out.append({
                "name": name, "runs": runs, "sixes": int(g["sixes"].sum()),
                "strike_rate": round(runs * 100 / balls, 1), "seasons": len(seasons),
                "first_season": seasons[0], "last_season": seasons[-1],
                "best_season": best_season, "best_season_runs": int(by_season.max()),
                "highest": int(hs["runs"]), "highest_opp": hs["opp"], "highest_season": hs["season"],
                "teams": teams.loc[teams["batter"] == name, "batting_team"].tolist(),
            })
        out.sort(key=lambda p: -p["runs"])
        return out[:top_n]
