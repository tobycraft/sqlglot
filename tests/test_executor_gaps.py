"""Regression tests for sqlglot's Python executor itself (as opposed to gaps
in sqlcov's own coverage tracking - see test_sqlcov_gaps.py), found by
stress-testing sqlcov against a large production Athena query (not included
in this repo).

Still-open gaps here would be marked ``xfail(strict=True)``, asserting the
*correct* result sqlglot cannot yet produce; XPASS then fails the suite
(dropping the marker, at that point, is also the cue to check whether
sqlcov's own coverage surface - predicates/CASE arms - should widen to use
the newly-supported construct). There are none open right now - every gap
found during that survey has been fixed in the local sqlglot checkout
(tracked via ``[tool.uv.sources]`` in pyproject.toml while it's under active
development); the tests below are plain regression guards.
"""

from __future__ import annotations

import datetime

import pytest
import sqlglot
from sqlglot.executor import execute
from sqlglot.executor.python import PythonExecutor
from sqlglot.executor.table import ensure_tables
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify
from sqlglot.planner import Plan

# --- Fixed: an expression wrapping a window function, in the SAME select ---
# When a window function's result was consumed by another expression *within
# the same SELECT that computes it* (comparison, arithmetic, CASE, even a
# plain scalar function call), the wrapping expression used to be silently
# discarded, and the projection returned the raw, un-wrapped window value
# instead - no error, just a wrong number. This was distinct from
# test_date_trunc_over_window_aggregate below, which wraps a window result
# that was already materialized as a plain column by an *earlier* CTE/step -
# that case always worked; only same-select wrapping was affected. Found this
# way stress-testing sqlcov: a CASE guarding on `SUM(x) OVER (PARTITION BY
# ...)` in the same CTE that computes the sum silently returned the sum
# itself (always truthy), miscounting every arm. Fixed upstream; kept as a
# regression guard.

_WINDOW_WRAPPED_CASES = [
    pytest.param(
        "SUM(b) OVER (PARTITION BY a) + 1",
        [(1, 16), (1, 16), (2, 6), (2, 6)],
        id="arithmetic",
    ),
    pytest.param(
        "SUM(b) OVER (PARTITION BY a) > 10",
        [(1, True), (1, True), (2, False), (2, False)],
        id="comparison",
    ),
    pytest.param(
        "CASE WHEN SUM(b) OVER (PARTITION BY a) > 10 THEN 'big' ELSE 'small' END",
        [(1, "big"), (1, "big"), (2, "small"), (2, "small")],
        id="case",
    ),
]


@pytest.mark.parametrize("expr, expected", _WINDOW_WRAPPED_CASES)
def test_expression_wrapping_window_in_same_select(expr, expected):
    tables = {"t": [{"a": 1, "b": 6}, {"a": 1, "b": 9}, {"a": 2, "b": 3}, {"a": 2, "b": 2}]}
    schema = {"t": {"a": "BIGINT", "b": "BIGINT"}}
    res = execute(f"SELECT a, {expr} AS x FROM t", schema=schema, tables=tables, dialect="presto")
    assert sorted(res.rows) == sorted(expected)


# --- Fixed: DATE_TRUNC on a window-aggregate result -------------------------
# DATE_TRUNC(unit, MIN(d) OVER (...)) - i.e. applied to a window aggregate's
# result rather than a plain column - used to get canonicalized to
# exp.TimestampTrunc instead of exp.DateTrunc, even when the underlying
# column is schema-typed DATE, and sqlglot.executor.env.ENV had no
# "TIMESTAMPTRUNC" entry (only "DATETRUNC"). Found stress-testing sqlcov
# against a real Athena CTAS: `DATE_TRUNC('month', MIN(...) OVER (...))`.
# Fixed upstream; kept as a regression guard.


def test_date_trunc_over_window_aggregate():
    sql = """
WITH a AS (
  SELECT g, MIN(d) OVER (PARTITION BY g) AS min_d FROM t
)
SELECT DATE_TRUNC('month', min_d) AS x FROM a
"""
    tables = {
        "t": [
            {"g": 1, "d": datetime.date(2025, 1, 31)},
            {"g": 1, "d": datetime.date(2025, 2, 10)},
        ]
    }
    schema = {"t": {"g": "BIGINT", "d": "DATE"}}
    res = execute(sql, schema=schema, tables=tables, dialect="athena")
    assert sorted(res.rows) == [(datetime.date(2025, 1, 1),), (datetime.date(2025, 1, 1),)]


# --- Fixed: bare DATE(...) constructor under the athena dialect ------------
# Under "athena", DATE(x) parses to its own exp.Date node (rather than
# normalizing to CAST(x AS DATE), the way it does under "presto"), and
# sqlglot.executor.env.ENV used to have no "DATE" entry - raising NameError
# the moment a row was evaluated. Found stress-testing sqlcov against a real
# Athena CTAS: `DATE(('2025-11-30'))` in a CASE guard. Fixed upstream; kept
# as a regression guard.


def test_date_constructor_under_athena_dialect():
    res = execute(
        "SELECT DATE(('2025-11-30')) AS x FROM t", tables={"t": [{"a": 1}]}, dialect="athena"
    )
    assert list(res.rows) == [(datetime.date(2025, 11, 30),)]


# --- Fixed: window functions --------------------------------------------------
# exp.Window used to have no PythonGenerator transform, so it fell through to
# the default, dialect-agnostic SQL generator, which emitted the literal
# "FUNC(...) OVER (...)" text - not valid Python, so every window function
# failed the same way, regardless of which one it was or whether it was an
# aggregate reused as one. Fixed upstream; kept as a regression guard.

_WINDOW_CASES = [
    pytest.param(
        "SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn FROM t",
        {"t": [{"a": 2}, {"a": 1}, {"a": 3}]},
        [(1, 1), (2, 2), (3, 3)],
        id="row_number",
    ),
    pytest.param(
        "SELECT a, RANK() OVER (ORDER BY a) AS rk FROM t",
        {"t": [{"a": 1}, {"a": 1}, {"a": 2}]},
        [(1, 1), (1, 1), (2, 3)],
        id="rank",
    ),
    pytest.param(
        "SELECT a, LAG(a) OVER (ORDER BY a) AS lg, LEAD(a) OVER (ORDER BY a) AS ld FROM t",
        {"t": [{"a": 1}, {"a": 2}, {"a": 3}]},
        [(1, None, 2), (2, 1, 3), (3, 2, None)],
        id="lag_lead",
    ),
    pytest.param(
        "SELECT a, SUM(b) OVER (PARTITION BY a) AS s FROM t",
        {"t": [{"a": 1, "b": 10}, {"a": 1, "b": 5}, {"a": 2, "b": 7}]},
        [(1, 15), (1, 15), (2, 7)],
        id="aggregate_over_partition",
    ),
]


@pytest.mark.parametrize("sql, tables, expected", _WINDOW_CASES)
def test_window_function(sql, tables, expected):
    res = execute(sql, tables=tables, dialect="presto")
    assert sorted(res.rows) == sorted(expected)


# --- Fixed: aggregate FILTER clause -----------------------------------------
# Used to have no PythonGenerator transform for exp.Filter, so it fell
# through to the raw "FILTER (...)" SQL text, which isn't valid Python.
# Fixed upstream; kept as a regression guard.


def test_aggregate_filter_clause():
    res = execute(
        "SELECT COUNT(*) FILTER (WHERE a > 1) AS c FROM t",
        tables={"t": [{"a": 1}, {"a": 2}, {"a": 3}]},
        dialect="presto",
    )
    assert list(res.rows) == [(2,)]


# --- Fixed: UNNEST in FROM ---------------------------------------------------
# Used to fail before any function lookup happened - the planner's join step
# assumed every FROM-clause source was a real table and choked on
# `exp.Unnest` itself (`'Unnest' object has no attribute 'parts'`). Fixed
# upstream; kept as a regression guard.


def test_unnest_in_from_clause():
    res = execute(
        "SELECT x FROM t CROSS JOIN UNNEST(ARRAY[1, 2, 3]) AS u(x)",
        tables={"t": [{"a": 1}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1,), (2,), (3,)]


# --- Fixed: parameterized CAST types ----------------------------------------
# Used to stringify the target type directly into `exp.DType.<TYPE>`, which
# breaks for a parameterized type like DECIMAL(18, 2) - its precision/scale
# rendered as call arguments, `exp.DType.DECIMAL(18, 2)`, and DType members
# aren't callable. Fixed upstream; kept as a regression guard.

_PARAMETERIZED_CAST_CASES = [
    pytest.param("CAST(a AS DECIMAL(18, 2))", {"a": 1.256}, 1.26, id="decimal_with_precision"),
    pytest.param("CAST(s AS VARCHAR(5))", {"s": "hello world"}, "hello", id="varchar_with_length"),
]


@pytest.mark.parametrize("expr, row, expected", _PARAMETERIZED_CAST_CASES)
def test_cast_to_parameterized_type(expr, row, expected):
    res = execute(f"SELECT {expr} AS x FROM t", tables={"t": [row]}, dialect="presto")
    assert list(res.rows) == [(expected,)]


# --- Fixed: bare (unparameterized) DECIMAL cast -----------------------------
# Used to silently truncate to an integer instead of preserving fractional
# precision - no exception, just a wrong number. Fixed upstream; kept as a
# regression guard.


def test_cast_to_bare_decimal_preserves_fraction():
    res = execute(
        "SELECT CAST(a AS DECIMAL) AS x FROM t", tables={"t": [{"a": 1.5}]}, dialect="presto"
    )
    assert list(res.rows) == [(1.5,)]


# --- Fixed: DATE_DIFF honoring its unit argument ----------------------------
# Used to always return a day count regardless of the requested unit
# ('month', 'hour', ...) - silently wrong, not a crash. Fixed upstream; kept
# as a regression guard.


def test_date_diff_honors_unit():
    res = execute(
        "SELECT DATE_DIFF('month', CAST(d1 AS DATE), CAST(d2 AS DATE)) AS x FROM t",
        tables={"t": [{"d1": "2024-01-01", "d2": "2024-03-01"}]},
        dialect="presto",
    )
    assert list(res.rows) == [(2,)]


# --- Fixed: previously-missing scalar/array functions -----------------------
# Each of these used to compile fine (the generator just emits a call to an
# ALL_CAPS name) but raise NameError at row-eval time because
# sqlglot.executor.env.ENV had no entry for it. All now registered; kept as a
# regression guard grouped under one parametrized test.

_PREVIOUSLY_MISSING_FUNCTION_CASES = [
    pytest.param(
        "DATE_TRUNC('month', CAST(d AS DATE))",
        {"d": "2024-03-15"},
        datetime.date(2024, 3, 1),
        id="date_trunc",
    ),
    pytest.param(
        "DATE_ADD('day', 1, CAST(d AS DATE))",
        {"d": "2024-03-15"},
        datetime.date(2024, 3, 16),
        id="date_add",
    ),
    pytest.param(
        "LAST_DAY_OF_MONTH(CAST(d AS DATE))",
        {"d": "2024-03-15"},
        datetime.date(2024, 3, 31),
        id="last_day_of_month",
    ),
    pytest.param(
        "DAY_OF_WEEK(CAST(d AS DATE))",
        {"d": "2024-03-15"},  # a Friday; Presto DAY_OF_WEEK is ISO (Mon=1..Sun=7)
        5,
        id="day_of_week",
    ),
    pytest.param("ARRAY_SORT(ARRAY[3, 1, 2])", {}, [1, 2, 3], id="array_sort"),
    pytest.param("ARRAY_DISTINCT(ARRAY[1, 1, 2])", {}, [1, 2], id="array_distinct"),
    pytest.param("ARRAYS_OVERLAP(ARRAY[1, 2], ARRAY[2, 3])", {}, True, id="arrays_overlap"),
    pytest.param("ARRAY_MIN(ARRAY[3, 1, 2])", {}, 1, id="array_min"),
    pytest.param("CARDINALITY(ARRAY[1, 2, 3])", {}, 3, id="cardinality"),
    pytest.param("FLATTEN(ARRAY[ARRAY[1, 2], ARRAY[3]])", {}, [1, 2, 3], id="flatten"),
    pytest.param("SIGN(b)", {"b": -5}, -1, id="sign"),
    pytest.param("IS_NAN(CAST(b AS DOUBLE))", {"b": 5.0}, False, id="is_nan"),
    pytest.param("CONTAINS(ARRAY[1, 2], a)", {"a": 1}, True, id="contains"),
    pytest.param("TO_UTF8(s)", {"s": "hello"}, b"hello", id="to_utf8"),
    pytest.param("SPLIT_PART(s, 'l', 1)", {"s": "hello"}, "he", id="split_part"),
    pytest.param(
        "TRY_CAST(s AS INT)", {"s": "not a number"}, None, id="try_cast_returns_null_on_failure"
    ),
    pytest.param("GREATEST(a, b)", {"a": 1, "b": 10}, 10, id="greatest"),
    pytest.param("LEAST(a, b)", {"a": 1, "b": 10}, 1, id="least"),
]


@pytest.mark.parametrize("expr, row, expected", _PREVIOUSLY_MISSING_FUNCTION_CASES)
def test_previously_missing_function(expr, row, expected):
    res = execute(f"SELECT {expr} AS x FROM t", tables={"t": [row]}, dialect="presto")
    assert list(res.rows) == [(expected,)]


# --- Open: FIRST_VALUE window function is not implemented -------------------
# Found stress-testing sqlcov against a real Athena CTAS pipeline: a dedup CTE
# that carries the first-seen value forward via
# `FIRST_VALUE(x) OVER (PARTITION BY ... ORDER BY ...)`. Every window function
# in _WINDOW_CASES above works; FIRST_VALUE specifically raises
# `Window function not supported: FIRST_VALUE(...)` - PythonExecutor's window
# dispatch has no case for it.


@pytest.mark.xfail(strict=True, reason="PythonExecutor: FIRST_VALUE window function not supported")
def test_first_value_window_function():
    res = execute(
        "SELECT a, FIRST_VALUE(b) OVER (PARTITION BY a ORDER BY b) AS fv FROM t",
        tables={"t": [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 5}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1, 10), (1, 10), (2, 5)]


# --- Fixed: `||` string concatenation has no Python codegen -----------------
# Found stress-testing sqlcov against a real Athena CTAS: a derived-columns
# CTE builds a timestamp string via
# `DATE_PARSE(CAST(d AS VARCHAR) || SUBSTR(t, 12), ...)`. The `||` operator
# (exp.DPipe under presto/athena) compiles to a call to a Python-env function
# named "DPIPE", which sqlglot.executor.env.ENV had no entry for - a bare
# NameError at row-eval time, not a parse or plan error. Fixed upstream; kept
# as a regression guard.


def test_string_concat_operator():
    res = execute(
        "SELECT a || b AS x FROM t", tables={"t": [{"a": "foo", "b": "bar"}]}, dialect="presto"
    )
    assert list(res.rows) == [("foobar",)]


# --- Open: date + INTERVAL 'n' MONTH arithmetic -----------------------------
# Found stress-testing sqlcov against a real Athena CTAS computing a
# schedule's end date as `start_date + INTERVAL '1' MONTH + INTERVAL '-1' DAY`.
# A DAY interval works fine (it's a fixed duration), but a MONTH interval
# compiles to `datetime.timedelta(months=...)` - and `timedelta` has no
# `months` parameter (months aren't a fixed number of days), raising
# `TypeError: 'months' is an invalid keyword argument for __new__()`.


@pytest.mark.xfail(
    strict=True, reason="PythonExecutor: INTERVAL MONTH arithmetic misuses datetime.timedelta"
)
def test_date_plus_interval_month():
    res = execute(
        "SELECT d + INTERVAL '1' MONTH AS x FROM t",
        tables={"t": [{"d": "2024-01-15"}]},
        schema={"t": {"d": "DATE"}},
        dialect="presto",
    )
    assert list(res.rows) == [(datetime.date(2024, 2, 15),)]


# --- Open: an un-merged nested derived table breaks the planner ------------
# sqlcov deliberately runs only `qualify()` + `annotate_types()` before
# planning (not the full `optimize()` pipeline `execute()` uses under the
# hood) - see coverage.py's module docstring - specifically so a predicate
# like `x IN (1, 2)` isn't rewritten before it's tracked. `optimize()`'s
# "merge subqueries" rule normally collapses a pass-through wrapper like
# `SELECT * FROM (SELECT a FROM t)` into one Scan; skip that rule (as sqlcov's
# pipeline does) and the *same* SQL, planned and executed the exact way
# sqlcov does it, dies with a KeyError naming the wrapper's auto-generated
# alias (e.g. "_0") - the Scan step for the outer SELECT looks the inner
# derived table up in a tables map that was never populated for it. Found
# stress-testing sqlcov against a real Athena CTAS whose CTEs nest several
# unaliased derived tables (`SELECT * FROM (SELECT * FROM (...))`), which
# Athena's own DDL exporter routinely emits and qualify() names `_innerN`.
# `execute()`'s own tests all pass because `execute()` calls full `optimize()`
# first, which erases the wrapper before this ever matters - this test uses
# the qualify/annotate_types/Plan/PythonExecutor pipeline directly, mirroring
# coverage.py, to exercise the code path sqlcov actually runs.


@pytest.mark.xfail(
    strict=True,
    reason="PythonExecutor: KeyError scanning an un-merged nested derived table "
    "(only hit when optimize()'s merge-subqueries rule is skipped, as sqlcov does)",
)
def test_nested_derived_table_without_subquery_merging():
    schema = {"t": {"code": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one("SELECT * FROM (SELECT code FROM t)", dialect="presto"),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"code": 1}, {"code": 2}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1,), (2,)]
