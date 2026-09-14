"""
Runs the TPC-H or TPC-DS queries through the Python engine against local CSV data.

Reads a directory of ``<table>.csv.gz`` files laid out like the fixtures under
``tests/fixtures/optimizer`` - either those directories themselves, or one
written by ``benchmarks.tpch_datagen``:

    python -m benchmarks.tpc_run                      # TPC-H, on the fixtures
    python -m benchmarks.tpc_run --suite tpc-ds       # TPC-DS, on the fixtures
    python -m benchmarks.tpc_run /tmp/tpch            # TPC-H, on generated data

With duckdb installed each result is also checked against duckdb over the same
data; without it, queries are only run and timed. The duckdb reference is the
optimized form of each query, which is what the engine runs: several TPC-DS
queries lean on implicit string/date coercion in the raw text that duckdb
rejects but `canonicalize` makes explicit.
"""

from __future__ import annotations

import argparse
import csv
import decimal
import gzip
import os
import time
import typing as t

from sqlglot import find_tables, parse_one, transpile
from sqlglot.executor import execute
from sqlglot.executor.table import Table
from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs

SUITES = {
    "tpc-h": (TPCH_SCHEMA, "optimizer/tpc-h/tpc-h.sql"),
    "tpc-ds": (TPCDS_SCHEMA, "optimizer/tpc-ds/tpc-ds.sql"),
}

# duckdb hands back Decimal for DECIMAL columns; compare it numerically.
NUMERIC = (int, float, decimal.Decimal)

INT_TYPES = {"int", "integer", "bigint", "smallint", "tinyint"}
FLOAT_TYPES = {"double", "float", "real", "decimal"}


def converter(type_: str) -> t.Callable:
    name = type_.split("(")[0].strip().lower()

    if name in INT_TYPES:
        return lambda v: int(float(v))
    if name in FLOAT_TYPES:
        return float

    return str


def open_file(path: str) -> t.IO:
    with open(path, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"

    if gzipped:
        return gzip.open(path, "rt", newline="")

    return open(path, encoding="utf-8", newline="")


def table_path(directory: str, table: str) -> str:
    path = os.path.join(directory, f"{table}.csv.gz")
    return path if os.path.exists(path) else os.path.join(directory, f"{table}.csv")


def load(directory: str, schema: dict) -> dict[str, Table]:
    """Reads each table, converting columns by the schema rather than by guessing."""
    tables = {}

    for table, columns in schema.items():
        reader = csv.reader(open_file(table_path(directory, table)), delimiter="|")
        ctypes = [converter(type_) for type_ in columns.values()]
        next(reader)
        rows = [tuple(None if v == "" else c(v) for c, v in zip(ctypes, row)) for row in reader]
        tables[table] = Table(columns=columns, rows=rows)

    return tables


def duckdb_connection(directory: str, schema: dict):
    try:
        import duckdb
    except ImportError:
        return None

    conn = duckdb.connect()

    for table, columns in schema.items():
        path = table_path(directory, table)
        conn.execute(
            f"CREATE VIEW {table} AS SELECT * FROM "
            f"READ_CSV('{path}', delim='|', header=True, columns={columns})"
        )

    return conn


def run(directory: str, suite: str = "tpc-h", only: set | None = None) -> int:
    """Runs every query in the suite, returning the number that failed."""
    schema, fixture = SUITES[suite]
    tables = load(directory, schema)
    conn = duckdb_connection(directory, schema)
    queries = [
        (i, sql, optimized)
        for i, (_, sql, optimized) in enumerate(load_sql_fixture_pairs(fixture), 1)
        if not only or i in only
    ]
    failures = 0

    for i, sql, optimized in queries:
        start = time.time()

        try:
            used = {t.name: tables[t.name] for t in find_tables(parse_one(sql))}
            result = execute(parse_one(sql), schema=schema, tables=used)
        except Exception as e:
            print(f"q{i:<3} FAILED  {type(e).__name__}: {e}")
            failures += 1
            continue

        elapsed = time.time() - start
        status = "ok"

        if conn is not None:
            # The raw text leans on implicit string/date coercion duckdb
            # rejects; the optimized form, which is what the engine runs,
            # carries the casts `canonicalize` added.
            try:
                expected = conn.execute(transpile(optimized, write="duckdb")[0]).fetchall()
            except Exception:
                expected = conn.execute(transpile(sql, write="duckdb")[0]).fetchall()

            if len(expected) != len(result.rows):
                status = f"MISMATCH duckdb={len(expected)} rows, engine={len(result.rows)}"
                failures += 1
            elif not rows_match(expected, result.rows):
                status = "MISMATCH values differ"
                failures += 1

        print(f"q{i:<3} {len(result.rows):>6} rows  {elapsed:7.3f}s  {status}")

    print(f"\n{len(queries) - failures}/{len(queries)} queries ok")
    return failures


def rows_match(expected: list, actual: list, tolerance: float = 1e-6) -> bool:
    for want, got in zip(expected, actual):
        if len(want) != len(got):
            return False

        for a, b in zip(want, got):
            if a is None or b is None:
                if a is not None or b is not None:
                    return False
            elif isinstance(a, NUMERIC) and isinstance(b, NUMERIC):
                if abs(float(a) - float(b)) > tolerance * max(1.0, abs(float(a))):
                    return False
            elif str(a) != str(b):
                return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", help="directory of <table>.csv.gz files")
    parser.add_argument("--suite", choices=sorted(SUITES), default="tpc-h")
    parser.add_argument("--queries", help="comma-separated query numbers to run")
    args = parser.parse_args()
    directory = args.directory or os.path.join(FIXTURES_DIR, "optimizer", args.suite)
    only = {int(q) for q in args.queries.split(",")} if args.queries else None
    raise SystemExit(1 if run(directory, args.suite, only) else 0)


if __name__ == "__main__":
    main()
