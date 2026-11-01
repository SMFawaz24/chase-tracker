"""Guard for SQL written by an LLM: read-only, one statement, one table, bounded.

Three layers, so no single mistake is enough to cause harm:
  1. This parser check (SELECT only, allowed tables only, no file or system functions, LIMIT enforced).
  2. DuckDB runs with external access disabled and its configuration locked (see facts.Data).
  3. A timeout interrupts any query that runs too long.
"""
import threading

import sqlglot
from sqlglot import exp

FORBIDDEN_NODES = {
    "Insert", "Update", "Delete", "Create", "Drop", "Alter", "Command", "Copy", "Attach", "Detach",
    "Pragma", "Set", "Use", "Transaction", "Commit", "Rollback", "Merge", "TruncateTable", "Export",
}
FORBIDDEN_FUNCS = {"glob", "getenv", "current_setting", "query", "query_table", "sniff_csv", "which_secret"}
FORBIDDEN_PREFIXES = ("read_", "duckdb_", "pragma_", "parquet_", "iceberg_", "delta_", "json_read")


class SQLGuardError(ValueError):
    pass


def validate_sql(sql, allowed_tables=("deliveries",), max_limit=50):
    """Return a normalised, bounded SELECT statement or raise SQLGuardError."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise SQLGuardError(f"cannot parse SQL: {str(e)[:120]}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLGuardError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SQLGuardError("only SELECT statements are allowed")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | ctes
    for node in tree.walk():
        if type(node).__name__ in FORBIDDEN_NODES:
            raise SQLGuardError(f"{type(node).__name__} is not allowed")
        if isinstance(node, exp.Table):
            if isinstance(node.this, exp.Func) or not isinstance(node.this, exp.Identifier):
                raise SQLGuardError("table functions are not allowed")
            if node.args.get("db") or node.args.get("catalog"):
                raise SQLGuardError("qualified table names are not allowed")
            if node.name.lower() not in allowed:
                raise SQLGuardError(f"table '{node.name}' is not allowed")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCS or name.startswith(FORBIDDEN_PREFIXES):
                raise SQLGuardError(f"function '{name}' is not allowed")

    if isinstance(tree, exp.Select):
        limit = tree.args.get("limit")
        if limit is None:
            tree = tree.limit(max_limit)
        else:
            try:
                n = int(limit.expression.name)
            except (ValueError, AttributeError):
                raise SQLGuardError("LIMIT must be a plain number") from None
            if n > max_limit:
                raise SQLGuardError(f"LIMIT must be at most {max_limit}")
    return tree.sql(dialect="duckdb")


def run_sql(con, sql, timeout=10.0):
    """Execute on a private cursor; interrupt the query if it exceeds `timeout` seconds."""
    cur = con.cursor()
    timer = threading.Timer(timeout, cur.interrupt)
    timer.start()
    try:
        return cur.execute(sql).fetchall()
    finally:
        timer.cancel()
        cur.close()
