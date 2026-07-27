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
development); the tests below are plain regression guards. That includes one
genuine regression (not a new gap): a since-landed fix for a narrow
nested-derived-table case briefly broke plain chained CTEs (see
test_chained_ctes_regressed_by_nested_derived_table_fix) before being fixed
for real.
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


# --- Fixed: FIRST_VALUE window function was not implemented -----------------
# Found stress-testing sqlcov against a real Athena CTAS pipeline: a dedup CTE
# that carries the first-seen value forward via
# `FIRST_VALUE(x) OVER (PARTITION BY ... ORDER BY ...)`. Every window function
# in _WINDOW_CASES above works; FIRST_VALUE specifically raised
# `Window function not supported: FIRST_VALUE(...)` - PythonExecutor's window
# dispatch had no case for it. Fixed upstream; kept as a regression guard.


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
# (exp.DPipe under presto/athena) compiled to a call to a Python-env function
# named "DPIPE", which sqlglot.executor.env.ENV had no entry for - a bare
# NameError at row-eval time, not a parse or plan error. Fixed upstream; kept
# as a regression guard.


def test_string_concat_operator():
    res = execute(
        "SELECT a || b AS x FROM t", tables={"t": [{"a": "foo", "b": "bar"}]}, dialect="presto"
    )
    assert list(res.rows) == [("foobar",)]


# --- Fixed: date + INTERVAL 'n' MONTH arithmetic ----------------------------
# Found stress-testing sqlcov against a real Athena CTAS computing a
# schedule's end date as `start_date + INTERVAL '1' MONTH + INTERVAL '-1' DAY`.
# A DAY interval works fine (it's a fixed duration), but a MONTH interval used
# to compile to `datetime.timedelta(months=...)` - and `timedelta` has no
# `months` parameter (months aren't a fixed number of days), raising
# `TypeError: 'months' is an invalid keyword argument for __new__()`. Fixed
# upstream; kept as a regression guard.


def test_date_plus_interval_month():
    res = execute(
        "SELECT d + INTERVAL '1' MONTH AS x FROM t",
        tables={"t": [{"d": "2024-01-15"}]},
        schema={"t": {"d": "DATE"}},
        dialect="presto",
    )
    assert list(res.rows) == [(datetime.date(2024, 2, 15),)]


# --- Fixed: an un-merged nested derived table broke the planner -------------
# sqlcov deliberately runs only `qualify()` + `annotate_types()` before
# planning (not the full `optimize()` pipeline `execute()` uses under the
# hood) - see coverage.py's module docstring - specifically so a predicate
# like `x IN (1, 2)` isn't rewritten before it's tracked. `optimize()`'s
# "merge subqueries" rule normally collapses a pass-through wrapper like
# `SELECT * FROM (SELECT a FROM t)` into one Scan; skip that rule (as sqlcov's
# pipeline does) and the *same* SQL, planned and executed the exact way
# sqlcov does it, used to die with a KeyError naming the wrapper's
# auto-generated alias (e.g. "_0") - the Scan step for the outer SELECT looked
# the inner derived table up in a tables map that was never populated for it.
# Found stress-testing sqlcov against a real Athena CTAS whose CTEs nest
# several unaliased derived tables (`SELECT * FROM (SELECT * FROM (...))`),
# which Athena's own DDL exporter routinely emits and qualify() names
# `_innerN`. `execute()`'s own tests all passed throughout because `execute()`
# calls full `optimize()` first, which erases the wrapper before this ever
# mattered - this test uses the qualify/annotate_types/Plan/PythonExecutor
# pipeline directly, mirroring coverage.py, to exercise the code path sqlcov
# actually runs. Fixed upstream; kept as a regression guard.


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


# --- Fixed: LAST_VALUE window function was not implemented ------------------
# Found stress-testing sqlcov against a real Athena CTAS pipeline (once an
# earlier statement's failure stopped masking this one): a dedup CTE tracks
# the *last*-seen value in a partition via
# `LAST_VALUE(x) OVER (PARTITION BY ... ORDER BY ...)`. Same gap as the
# already-fixed FIRST_VALUE above, but that fix didn't cover LAST_VALUE -
# PythonExecutor's window dispatch had no case for it, raising
# `Window function not supported: LAST_VALUE(...)`. Fixed upstream; kept as a
# regression guard.


def test_last_value_window_function():
    res = execute(
        "SELECT a, LAST_VALUE(b) OVER (PARTITION BY a ORDER BY b) AS lv FROM t",
        tables={"t": [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 5}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1, 20), (1, 20), (2, 5)]


# --- Fixed: a nested derived table used as one side of a JOIN ---------------
# The already-fixed test_nested_derived_table_without_subquery_merging above
# covers an un-merged nested derived table (`SELECT * FROM (SELECT ...)`) as
# the query's *sole* top-level FROM. Found stress-testing sqlcov against a
# real Athena CTAS: the same shape of nested derived table, but used as one
# side of a JOIN instead (`FROM t AS base LEFT JOIN (SELECT * FROM (SELECT
# ... FROM t)) AS sub ON ...`) - a CTE wrapping a dedup subquery, then joined
# to a base table, which is exactly how the production query's CTEs
# (code_revised, sub_code_revised) are consumed. That variant used to raise a
# `KeyError` naming the wrapper's auto-generated alias (e.g. "_0") - the
# earlier fix for the sole-top-level-FROM case didn't reach this one. Neither
# a single level of nesting joined directly, nor the double-nested form
# outside a JOIN, reproduced this - it took both the double nesting *and*
# the JOIN together. Fixed upstream; kept as a regression guard.


def test_nested_derived_table_joined_to_another_table():
    schema = {"t": {"code": "BIGINT", "n": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one(
            "SELECT base.code, sub.n AS a FROM t AS base "
            "LEFT JOIN (SELECT * FROM (SELECT code, n FROM t)) AS sub ON base.code = sub.code",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"code": 1, "n": 1}, {"code": 2, "n": 1}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1, 1), (2, 1)]


# --- Fixed: an aggregate feeding a window, under a nested wrapper, joined ---
# A deeper variant of the same bug class as the two nested-derived-table
# tests above, neither of which reached it. Found stress-testing sqlcov
# against a real Athena CTAS: a dedup CTE shaped
# `WITH x AS (SELECT * FROM (SELECT ..., ROW_NUMBER() OVER (...) AS rn FROM
# (SELECT ..., COUNT(*) AS n FROM t GROUP BY ...)) WHERE rn = 1)`, then
# LEFT JOINed to a base table - i.e. a GROUP BY (`Aggregate` step) feeding a
# `ROW_NUMBER()` window, itself wrapped in another `SELECT * FROM (...)
# WHERE rn = 1`, then joined. Drop the GROUP BY and it's exactly
# test_nested_derived_table_joined_to_another_table above (already fixed);
# drop the outer `WHERE rn = 1` wrapper and it works fine even with the
# GROUP BY in place - it took the aggregate-under-window-under-wrapper-
# under-JOIN combination together to raise a `KeyError` naming an
# auto-generated alias (e.g. "_1"). Fixed upstream (repointing the stale
# alias in a doubly-nested derived table join); kept as a regression guard.
#
# NOTE: that same fix commit introduced a severe, unrelated regression -
# see test_chained_ctes_regressed_by_nested_derived_table_fix below.


def test_aggregate_fed_window_under_nested_wrapper_joined_to_another_table():
    schema = {"t": {"code": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one(
            "WITH code_revised AS ("
            "  SELECT * FROM ("
            "    SELECT code, ROW_NUMBER() OVER (PARTITION BY code ORDER BY n DESC) AS rn"
            "    FROM (SELECT code, COUNT(*) AS n FROM t GROUP BY code)"
            "  ) WHERE rn = 1"
            ") "
            "SELECT base.code, code_revised.rn AS a FROM t AS base "
            "LEFT JOIN code_revised ON base.code = code_revised.code",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"code": 1}, {"code": 1}, {"code": 2}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1, 1), (1, 1), (2, 1)]


# --- Fixed: REGRESSION - plain chained CTEs, no nesting at all --------------
# Not a new gap - a severe regression, introduced by the very commit that
# fixed test_aggregate_fed_window_under_nested_wrapper_joined_to_another_table
# above ("repoint stale alias in doubly-nested derived table joins"). Found
# stress-testing sqlcov against a real Athena CTAS pipeline: a plain chain of
# CTEs, each just `SELECT ... FROM <previous CTE>` - no un-merged nested
# derived table, no window, no aggregate, nothing exotic. Two CTEs chained
# was already enough (`WITH a AS (...), b AS (SELECT a.x FROM a) SELECT *
# FROM b`); one CTE alone was fine. This was about as common a SQL shape as
# exists, so this regression was far more damaging than the narrow case the
# commit fixed - it also took down sqlcov's own
# test_sqlcov_gaps.py::test_case_tracked_through_chained_and_diamond_ctes,
# which is now un-xfailed too. Fixed upstream for real this time; kept as a
# regression guard.


def test_chained_ctes_regressed_by_nested_derived_table_fix():
    schema = {"t": {"code": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one(
            "WITH a AS (SELECT code FROM t), b AS (SELECT a.code FROM a) SELECT * FROM b",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"code": 1}, {"code": 2}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1,), (2,)]


# --- Fixed: an un-merged derived table that renames a column ----------------
# The true root cause behind every "nested derived table" KeyError above and
# the "'flag_last_row'"/"'flag_rows_to_remove'" crashes found stress-testing
# sqlcov against a real Athena CTAS - none of those fixes actually reached
# it. `Scan.from_expression` in planner.py handled `SELECT ... FROM
# (subquery)` by reusing the subquery's own inner Step wholesale: it renamed
# the inner Step to the subquery's alias and returned it directly. But
# `Step.from_expression` (the caller) then built the *enclosing* SELECT's own
# projections from its own expression list and unconditionally overwrote
# `step.projections` with them - discarding the inner Step's real, computed
# projections. That's invisible when the derived table is a bare passthrough
# (`SELECT code FROM t`, no rename): the outer's projections reference the
# same column name the raw table already has, so evaluation limps along by
# coincidence. The instant the derived table renames or computes a column
# (`SELECT code AS x FROM t`), the outer's projections reference a column
# ("x") that was never actually computed by anything, anywhere - a bare
# `KeyError` on that name at execution time, regardless of nesting depth
# (one level was already enough) or whether a JOIN/window/aggregate was
# layered on top (none of those were necessary either - they just happened
# to be what the earlier, narrower fixes above targeted).
#
# Fixed by no longer renaming/reusing the inner Step in place: `Scan.
# from_expression` now wires it in as a genuine dependency, exactly the way
# a CTE reference already worked correctly (see the `with_` handling and the
# plain-`exp.Table` branch in the same function) - the executor's existing
# "scan a by-name dependency" path (`PythonExecutor.scan`) then evaluates the
# outer SELECT's projections against the inner Step's real, already-computed
# output, the same mechanism that made CTE renames work all along.
#
# Found stress-testing sqlcov against a real Athena CTAS pipeline: an SCD
# dedup chain of several `SELECT ... FROM (SELECT ... FROM (...))` layers,
# each renaming/computing a column the next layer depends on.


def test_renamed_column_in_unmerged_derived_table():
    schema = {"t": {"code": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one("SELECT * FROM (SELECT code AS x FROM t)", dialect="presto"),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"code": 1}, {"code": 2}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1,), (2,)]


# --- Fixed: a UNION as the sole content of a derived table -------------------
# A fresh gap surfaced by the fix above: wiring the inner Step in as a
# dependency via `step.source = inner.name` assumed `inner` always has a
# name. It doesn't when the derived table's body is a `UNION`/`INTERSECT`/
# `EXCEPT` - `SetOperation.from_expression` names its `left`/`right`
# dependencies but never names the SetOperation Step itself, since it's
# normally consumed directly by whatever references it by its own name. Left
# unnamed, `step.source` ended up `None`, and `PythonExecutor.scan` reads a
# `None` source as "no FROM at all", building an empty static context - the
# derived table's Scan step then found zero tables to read a column list
# from: `IndexError: list index out of range`.
#
# Fixed by giving the inner SetOperation Step a fallback name (its own
# derived-table alias) when it doesn't already have one, mirroring this
# file's existing `left.name = left.name or "left"` idiom.
#
# Found stress-testing sqlcov against a real Athena CTAS: a derived table
# merging two differently-sourced result sets row-wise (`FROM (SELECT ...
# UNION ALL SELECT ...) AS base`), immediately after the previous fix landed.


def test_union_as_sole_content_of_derived_table():
    schema = {"t": {"code": "BIGINT"}, "t2": {"code": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one(
            "SELECT base.code FROM (SELECT code FROM t UNION ALL SELECT code FROM t2) AS base",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables(
        {"t": [{"code": 1}, {"code": 2}], "t2": [{"code": 3}]}, dialect="presto"
    )
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(1,), (2,), (3,)]


# --- Fixed: comparing a `datetime.date` against a `datetime.datetime` -------
# Found stress-testing sqlcov against a real Athena CTAS, in two directions:
# a date-dimension join filters on `sticky_date_dim_latest.first_day_of_month
# >= CAST('2022-11-01' AS TIMESTAMP)` (a DATE column - loader.py only ever
# produces `datetime.date`, never `datetime.datetime`, for a fixture column -
# against a literal explicitly cast to TIMESTAMP); a later CASE guards on
# `_derived_cols._creation_timestamp < CAST('2023-08-03' AS DATE)` (the
# reverse: a computed `datetime.datetime`, from an earlier DATE_PARSE, against
# a literal cast to DATE). Athena/Presto coerce a DATE/TIMESTAMP comparison
# implicitly either way (a DATE reads as midnight on that day); the
# executor's comparison operators (env.py's LT/GT/GE/LE, thin wrappers around
# Python's own `<`/`>`/etc.) used to not, so both directions raised the
# identical `TypeError: can't compare datetime.datetime to datetime.date`
# instead of resolving the comparison. Fixed upstream; kept parametrized as a
# regression guard covering both directions.

_DATE_DATETIME_COMPARISON_CASES = [
    pytest.param(
        "SELECT d FROM t WHERE d >= CAST('2022-11-01' AS TIMESTAMP)",
        {"t": {"d": "DATE"}},
        [{"d": datetime.date(2022, 12, 1)}, {"d": datetime.date(2022, 1, 1)}],
        [(datetime.date(2022, 12, 1),)],
        id="date_column_ge_timestamp_cast_literal",
    ),
    pytest.param(
        "SELECT CASE WHEN d < CAST('2023-08-03' AS DATE) THEN 0 ELSE 1 END AS x FROM t",
        None,
        [
            {"d": datetime.datetime(2023, 8, 1, 10, 0)},
            {"d": datetime.datetime(2023, 9, 1, 10, 0)},
        ],
        [(0,), (1,)],
        id="computed_datetime_lt_date_cast_literal",
    ),
]


@pytest.mark.parametrize("sql, schema, rows, expected", _DATE_DATETIME_COMPARISON_CASES)
def test_date_datetime_comparison_does_not_coerce(sql, schema, rows, expected):
    res = execute(sql, tables={"t": rows}, schema=schema, dialect="athena")
    assert list(res.rows) == expected


# --- Fixed: a "simple CASE with a subject" generated invalid Python ---------
# sqlcov deliberately runs only `qualify()` + `annotate_types()` before
# planning (not the full `optimize()` pipeline `execute()` uses under the
# hood) - see coverage.py's module docstring. `optimize()`'s canonicalize
# step normally rewrites a "simple CASE" (`CASE x WHEN v1 THEN r1 WHEN v2
# THEN r2 END`, comparing a subject expression against each WHEN value) into
# the equivalent "searched CASE" form (`CASE WHEN x = v1 THEN r1 ...  END`);
# skip that rewrite (as sqlcov's pipeline does) and the raw simple-CASE node
# used to generate each WHEN comparison using a literal `=` instead of
# Python's `==` - a bare `SyntaxError` ("expected 'else' after 'if'
# expression") at row-eval time, not a plan or execution error. Neither an
# explicit ELSE nor a JOIN was needed to trigger it - a bare SELECT-list
# simple CASE was enough. `execute()`'s own tests didn't hit this because
# `execute()` calls full `optimize()` first, rewriting the simple CASE away
# before this ever mattered - this test uses the qualify/annotate_types/
# Plan/PythonExecutor pipeline directly, mirroring coverage.py, to exercise
# the code path sqlcov actually runs.
#
# Found stress-testing sqlcov against a real Athena CTAS: a JOIN condition
# matching `ipd2.ipdt_product = CASE seed.product WHEN 'svr' THEN 'stvr'
# WHEN 'bvr' THEN 'btb' END` - a bare SELECT-list simple CASE reproduces the
# identical failure with no JOIN needed at all. Fixed upstream (the
# PythonGenerator's `_case_sql` now emits `==` for the subject comparison);
# kept as a regression guard.


def test_simple_case_with_subject():
    schema = {"seed": {"product": "VARCHAR"}}
    tree = qualify(
        sqlglot.parse_one(
            "SELECT CASE seed.product WHEN 'svr' THEN 'stvr' WHEN 'bvr' THEN 'btb' END AS x "
            "FROM seed",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables(
        {"seed": [{"product": "svr"}, {"product": "bvr"}, {"product": "other"}]},
        dialect="presto",
    )
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows, key=lambda r: (r[0] is None, r[0])) == [
        ("btb",),
        ("stvr",),
        (None,),
    ]


# --- Fixed: CROSS JOIN UNNEST of a column reference (not a literal array) ---
# The already-fixed test_unnest_in_from_clause above covers `UNNEST(ARRAY[1,
# 2, 3])` - a literal array needing no row context to evaluate.
# `PythonExecutor.scan_unnest` evaluated the UNNEST expression against a
# brand-new, completely empty static context - fine for a literal, but a real
# column reference like `UNNEST(t.arr)` needs the *joined* row's own context
# to resolve `t`, which was never wired in: a bare `KeyError` naming the outer
# table, not a wrong answer. The array to explode differs per outer row, so
# it can't be scanned as an independent Step the way a real table can -
# planner.Join.from_joins now stashes the raw Unnest node on the join info
# instead of adding a Scan dependency for it, and PythonExecutor.join
# evaluates it directly against each row of the join accumulated so far
# (`lateral_unnest_join`), the way a LATERAL join would.
#
# Found stress-testing sqlcov against a real Athena CTAS: `FROM _opened_loans
# CROSS JOIN UNNEST(_opened_loans.customer_set_latest) AS u(cust_id)` -
# exploding an array-typed column from the joined CTE itself, not a literal.


def test_unnest_of_column_reference():
    res = execute(
        "SELECT t.id, u.x FROM t CROSS JOIN UNNEST(t.arr) AS u(x)",
        tables={"t": [{"id": 1, "arr": [10, 20]}, {"id": 2, "arr": [30]}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1, 10), (1, 20), (2, 30)]
