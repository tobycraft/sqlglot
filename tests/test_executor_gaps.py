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
development); the tests below are plain regression guards. That includes
one genuine regression (not a new gap): a since-landed fix for a narrow
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


# --- Fixed: a "simple CASE with a subject" generated invalid Python --------
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
# simple CASE was enough. `execute()`'s own tests never hit this because
# `execute()` calls full `optimize()` first, rewriting the simple CASE away
# before this ever mattered - this test uses the qualify/annotate_types/
# Plan/PythonExecutor pipeline directly, mirroring coverage.py, to exercise
# the code path sqlcov actually runs.
#
# Found stress-testing sqlcov against a real Athena CTAS: a JOIN condition
# matching `ipd2.ipdt_product = CASE seed.product WHEN 'svr' THEN 'stvr'
# WHEN 'bvr' THEN 'btb' END` - a bare SELECT-list simple CASE reproduces the
# identical failure with no JOIN needed at all. Fixed upstream; kept as a
# regression guard.


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
# `PythonExecutor.scan_unnest` used to evaluate the UNNEST expression against
# a brand-new, completely empty static context - fine for a literal, but a
# real column reference like `UNNEST(t.arr)` needs the *joined* row's own
# context to resolve `t`, which was never wired in: a bare `KeyError` naming
# the outer table, not a wrong answer.
#
# Found stress-testing sqlcov against a real Athena CTAS: `FROM _opened_loans
# CROSS JOIN UNNEST(_opened_loans.customer_set_latest) AS u(cust_id)` -
# exploding an array-typed column from the joined CTE itself, not a literal.
# Fixed upstream; kept as a regression guard.


def test_unnest_of_column_reference():
    res = execute(
        "SELECT t.id, u.x FROM t CROSS JOIN UNNEST(t.arr) AS u(x)",
        tables={"t": [{"id": 1, "arr": [10, 20]}, {"id": 2, "arr": [30]}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1, 10), (1, 20), (2, 30)]


# --- Fixed: WITH RECURSIVE was not supported --------------------------------
# planner.py/PythonExecutor used to have no concept of a self-referential
# CTE at all - the whole design is a single-pass DAG of Steps, each
# depending only on already-built ones, which was fundamentally incompatible
# with a recursive term that reads from the very CTE it's still defining.
# When the recursive branch's `FROM walk AS curr` got planned, "walk" wasn't
# a registered table anywhere yet, so `self.tables.find(step.source)`
# returned `None`, and the first attempt to read a column off it raised a
# bare `AttributeError: 'NoneType' object has no attribute 'range_reader'` -
# unlike most other gaps in this file, this wasn't a small codegen/wiring
# fix; it needed genuine iterative evaluation (repeatedly running the
# recursive term against the previous iteration's rows until it stops
# producing new ones), a different execution model than the rest of this
# executor. Fixed upstream; kept as a regression guard.
#
# Found stress-testing sqlcov against a real Athena CTAS: a `WITH RECURSIVE
# walk(...)` chain walking a linked list of account-merge edges to its root.


def test_recursive_cte():
    res = execute(
        """
        WITH RECURSIVE walk(current_id, next_id) AS (
          SELECT base.current_id, base.next_id FROM base
          UNION ALL
          SELECT curr.next_id, nxt.next_id
          FROM walk AS curr
          INNER JOIN base AS nxt ON curr.next_id = nxt.current_id
        )
        SELECT * FROM walk
        """,
        tables={"base": [{"current_id": 1, "next_id": 2}, {"current_id": 2, "next_id": 3}]},
        dialect="presto",
    )
    # base rows (1, 2) and (2, 3), plus one recursive round: from (1, 2),
    # next_id=2 joins base.current_id=2 -> adds (2, 3) again (UNION ALL, not
    # DISTINCT); from (2, 3), next_id=3 has no match in base, so recursion
    # stops there.
    assert sorted(res.rows) == [(1, 2), (2, 3), (2, 3)]


# --- Open: a correlated NOT EXISTS subquery generates invalid Python -------
# sqlcov deliberately runs only `qualify()` + `annotate_types()` before
# planning (not the full `optimize()` pipeline `execute()` uses under the
# hood) - see coverage.py's module docstring. `optimize()` normally
# decorrelates an `EXISTS`/`NOT EXISTS` subquery into a semi-join the planner
# can execute; skip that rewrite (as sqlcov's pipeline does) and the raw
# `exp.Exists` node used to fall through to a codegen path with no real
# Python transform for it - it emitted the subquery's own SQL text verbatim
# (`not EXISTS(SELECT 1 FROM "u" AS "u" WHERE ...)`), which isn't valid
# Python, raising a bare `SyntaxError` at row-eval time. `execute()`'s own
# tests never hit this because `execute()` calls full `optimize()` first,
# decorrelating the subquery away before this ever mattered - this test
# uses the qualify/annotate_types/Plan/PythonExecutor pipeline directly,
# mirroring coverage.py, to exercise the code path sqlcov actually runs.
#
# Found stress-testing sqlcov against a real Athena CTAS: `WHERE NOT
# EXISTS(SELECT 1 FROM h_inrr_tmp AS sub WHERE CONTAINS(sub.visited_ids,
# inrr_tmp.next_id))` - filtering out rows already reachable from another
# row's visited-id history. Fixed upstream; kept as a regression guard.
# NOTE: the bare correlated case here is fixed, but the production query's
# exact shape still fails differently as of this writing - see
# test_correlated_not_exists_with_array_contains below.


def test_correlated_not_exists_subquery():
    schema = {"t": {"id": "BIGINT"}, "u": {"id": "BIGINT"}}
    tree = qualify(
        sqlglot.parse_one(
            "SELECT t.id FROM t WHERE NOT EXISTS(SELECT 1 FROM u WHERE u.id = t.id)",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"id": 1}, {"id": 2}], "u": [{"id": 1}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(2,)]


# --- Fixed: a correlated NOT EXISTS subquery whose condition is CONTAINS ---
# A more specific variant of the fixed test_correlated_not_exists_subquery
# above, not covered by that fix at the time: the bare-equality correlated
# NOT EXISTS compiled fine, but the production query's exact shape - the
# subquery's WHERE is `CONTAINS(array_column, outer_column)`, not a plain
# equality - still failed the same way: `not EXISTS(SELECT 1 FROM "u" AS "u"
# WHERE ARRAYCONTAINS(scope["u"]["arr"], scope["t"]["id"])), line 1` was
# emitted as literal SQL/pseudo-Python text rather than valid Python,
# raising the identical `SyntaxError`. Fixed upstream; kept as a regression
# guard alongside the plain-equality case.
#
# Found stress-testing sqlcov against a real Athena CTAS: `WHERE NOT
# EXISTS(SELECT 1 FROM h_inrr_tmp AS sub WHERE CONTAINS(sub.visited_ids,
# inrr_tmp.next_id))` - the exact production shape.


def test_correlated_not_exists_with_array_contains():
    schema = {"t": {"id": "BIGINT"}, "u": {"arr": "ARRAY<BIGINT>"}}
    tree = qualify(
        sqlglot.parse_one(
            "SELECT t.id FROM t WHERE NOT EXISTS(SELECT 1 FROM u WHERE CONTAINS(u.arr, t.id))",
            dialect="presto",
        ),
        schema=schema,
        dialect="presto",
    )
    tree = annotate_types(tree, schema=schema, dialect="presto")
    tables = ensure_tables({"t": [{"id": 1}, {"id": 2}], "u": [{"arr": [1, 3]}]}, dialect="presto")
    result = PythonExecutor(tables=tables).execute(Plan(tree))
    assert sorted(result.rows) == [(2,)]


# --- Open: a CASE expression used as another CASE's WHEN condition ---------
# `_case_sql` in sqlglot/generators/python.py splices a WHEN's condition into
# the ternary chain unparenthesized: `chain = f"{true} if {condition} else
# ({chain})"`. When `condition` itself compiles to a Python ternary - i.e.
# the WHEN test is itself a CASE expression - the emitted text is a bare,
# unparenthesized "X if Y else Z" sitting where Python's grammar expects a
# plain (non-conditional) expression, e.g. `1 if True if a else False else
# 0`. `compile()` rejects that with "expected 'else' after 'if'
# expression" - a SyntaxError, not a runtime error, so it fails regardless of
# the row data. Found stress-testing sqlcov against a real Athena CTAS whose
# WHEN condition tested another CASE's result directly. Fixed by
# parenthesizing the WHEN condition in `_case_sql`.


def test_case_as_when_condition():
    sql = (
        "SELECT CASE WHEN (CASE WHEN a = 1 THEN TRUE ELSE FALSE END) AND b = 2 "
        "THEN 1 ELSE 0 END AS x FROM t"
    )
    res = execute(sql, tables={"t": [{"a": 1, "b": 2}]})
    assert list(res.rows) == [(1,)]


# --- Fixed: an explicit ROWS BETWEEN frame was silently ignored -----------
# `PythonExecutor._window_values` (sqlglot/executor/python.py) computed a
# generic aggregate window function (SUM, MIN, MAX, COUNT, AVG, ...) as
# `[agg(operands)] * width` - the aggregate over *every* row in the
# partition, broadcast unchanged to every row - regardless of any `ROWS
# BETWEEN ... AND ...` frame clause on the window. A running/cumulative
# frame like `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (extremely
# common for running totals, "has this ever happened up to this point"
# flags, etc.) instead silently returned the same whole-partition value for
# every row - wrong data, not an error, so it was easy to miss. Found
# stress-testing sqlcov against a real Athena CTAS that computed a
# cumulative SUM per account ordered by month to detect a balance that had
# gone to zero and stayed there; every row got the account's overall sum
# instead of a running one. Fixed by resolving each row's `ROWS` frame to a
# start/end position (via the new `_frame_bound` helper) and slicing
# `operands` to that window before calling `agg`, instead of always using
# the whole partition; `RANGE`/`GROUPS` frames still fall back to the old
# whole-partition behavior, since they need peer-row grouping by ORDER BY
# value rather than a plain positional slice.
def test_rows_between_frame_is_respected():
    sql = (
        "SELECT a, b, "
        "SUM(b) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_sum, "
        "MIN(b) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_min, "
        "MIN(b) OVER (PARTITION BY a ORDER BY b DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_min_desc "
        "FROM t"
    )
    res = execute(
        sql,
        tables={"t": [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 1, "b": 5}]},
        dialect="presto",
    )
    rows = {row[1]: row for row in res.rows}
    # ordered by b ASC: 5, 10, 20 -> running sums 5, 15, 30; running mins (ASC) 5, 5, 5
    # ordered by b DESC: 20, 10, 5 -> running mins (DESC) 20, 10, 5
    assert rows[5][2:] == (5, 5, 5)
    assert rows[10][2:] == (15, 5, 10)
    assert rows[20][2:] == (35, 5, 20)


# --- Fixed: a window function collapsed two same-named joined columns -----
# `PythonExecutor.window` (sqlglot/executor/python.py) rebuilds a fresh,
# range-less `Table` after computing a window function, then pointed *every*
# table alias in scope - not just the window step's own name - at that same
# unrestricted table. That's fine when every alias already shared one
# physical row (the common case), but when the preceding step joined two
# *different* aliases of the same underlying table (e.g. two lookups against
# one shared reference table, both contributing identically-named columns to
# the merged column list), a range-less `RowReader` builds its name->index
# map over *every* column - so the duplicate name resolves to whichever
# alias's column happened to land last, silently returning the wrong
# alias's value (e.g. `b.tgt` for both `a.tgt` and `b.tgt`) instead of
# raising. Only surfaced when some *other* projection in the same SELECT
# also used a window function - otherwise the planner drops the Window step
# as dead code, so a query needs both to fail. Found stress-testing sqlcov
# against a real Athena CTAS that joined a shared code/description lookup
# table twice (once per source column) inside a CTE that also computed a
# DENSE_RANK() - both lookup values ended up equal to the second join's
# result. Fixed by giving every alias other than the window step's own name
# a new `_AliasedRowReader`, resolving column names through that alias's own
# pre-window `column_range` while always reading the live current row off
# the shared table - preserving both per-alias disambiguation and the
# existing invariant that advancing iteration through any one alias name
# keeps every alias in lockstep.
def test_window_function_preserves_distinct_joined_aliases():
    sql = (
        "SELECT m.id, DENSE_RANK() OVER (PARTITION BY m.id ORDER BY m.id) AS rnk, "
        "la.tgt AS a_val, lb.tgt AS b_val "
        "FROM m "
        "LEFT JOIN lookup AS la ON m.code_a = la.src AND la.col = 'alpha' "
        "LEFT JOIN lookup AS lb ON m.code_b = lb.src AND lb.col = 'beta'"
    )
    tables = {
        "m": [{"id": 1, "code_a": "A1", "code_b": "B1"}, {"id": 2, "code_a": "A2", "code_b": "B2"}],
        "lookup": [
            {"src": "A1", "col": "alpha", "tgt": "fixed"},
            {"src": "A2", "col": "alpha", "tgt": "variable"},
            {"src": "B1", "col": "beta", "tgt": "io"},
            {"src": "B2", "col": "beta", "tgt": "pi"},
        ],
    }
    res = execute(sql, tables=tables, dialect="presto")
    rows = {row[0]: row for row in res.rows}
    assert rows[1][1:] == (1, "fixed", "io")
    assert rows[2][1:] == (1, "variable", "pi")


# --- Fixed: a LEFT/RIGHT JOIN with a non-equality residual ON condition ----
# used to drop unmatched outer rows entirely instead of NULL-padding them.
# `PythonExecutor.hash_join` (sqlglot/executor/python.py) buckets rows purely
# by the equi-join key extracted by `join_condition`, NULL-pads any bucket
# whose other side is empty (correct outer-join behavior for the KEY alone),
# then `PythonExecutor.join` applied whatever residual ON-clause survived key
# extraction (any non-equality conjunct, e.g. `b.upd <= a.closure`, or an
# equality that couldn't be pulled into the hash key) as a blanket
# `Context.filter` over the *entire* joined table - including the rows that
# were just NULL-padded because there was no key match at all. That filter
# evaluated the residual against the NULL-padded columns, got NULL/false,
# and threw the outer row away - so a LEFT JOIN with any extra ON condition
# beyond a plain equality silently behaved like an INNER JOIN whenever the
# left row had no match. Found stress-testing sqlcov against a real Athena
# CTAS joining a corrections/refinance-reason table via
# `LEFT JOIN fmsr ON mrg.id = fmsr.id AND (fmsr.upd <= mrg.closure + INTERVAL
# '90' DAY OR fmsr.upd <= DATE '2025-10-31')` - adding new fixture rows with
# no matching fmsr counterpart made them vanish from the join output entirely
# instead of appearing with the fmsr columns NULL. Fixed by having
# `hash_join` test the residual itself, per candidate `(a_row, b_row)` pair
# via the new `_residual_match_fn` helper, *before* deciding whether a source
# row has any match at all - not as a filter over the already-decided,
# already NULL-padded result.
def test_left_join_residual_condition_preserves_unmatched_rows():
    sql = (
        "SELECT a.id, b.upd "
        "FROM a "
        "LEFT JOIN b ON a.id = b.id AND b.upd <= a.closure"
    )
    tables = {
        "a": [{"id": 1, "closure": 5}, {"id": 2, "closure": 5}, {"id": 3, "closure": 5}],
        "b": [{"id": 1, "upd": 5}],
    }
    res = execute(sql, tables=tables, dialect="presto")
    rows = {row[0]: row for row in res.rows}
    assert rows[1] == (1, 5)
    assert rows[2] == (2, None)
    assert rows[3] == (3, None)


# --- Fixed: DOUBLE / DOUBLE division truncated to an integer under a -------
# TYPED_DIVISION dialect (Presto/Athena/Postgres/...). `_div_sql`
# (sqlglot/generators/python.py) picked the executor's `TYPEDDIV` op - which
# does `int(e / this)` (sqlglot/executor/env.py) - purely because
# `e.args.get("typed")` is set on the `Div` node, and every division parses
# with `typed=True` under these dialects (`Dialect.TYPED_DIVISION`), since
# that flag only records "this dialect's `/` follows typed-division rules",
# not "this particular division is between two integers". The optimizer's
# own `annotate_types._annotate_div` gets this right - it only assigns the
# division an integer result type when `typed` AND *both* operands are
# already integer-typed - but the code generator never consulted that, so a
# DOUBLE/DOUBLE (or DECIMAL/INT, etc.) division still got truncated to an
# int, silently producing 0 instead of a fraction. Found stress-testing
# sqlcov against a real Athena CTAS computing an interest-rate discount as
# `CAST((-base_rate + rate) AS DECIMAL(20, 5)) / 100` - every row came out
# as exactly 0 regardless of the actual rates. Fixed by having `_div_sql`
# check both operands' annotated types (mirroring `_annotate_div`'s own
# condition) before picking `TYPEDDIV` over the plain `DIV`.
def test_typed_division_dialect_preserves_fractional_result():
    res = execute(
        "SELECT (-base + rate) / 100 AS discount FROM t",
        tables={"t": [{"base": 50.0, "rate": 100.0}]},
        dialect="presto",
    )
    assert res.rows == [(0.5,)]


# --- Fixed: a LEFT/RIGHT JOIN with no equi-join key dropped unmatched rows -
# `PythonExecutor.nested_loop_join` (sqlglot/executor/python.py) - used when
# an ON-clause has no extractable equi-join key at all, e.g. every conjunct
# is a non-equality like a date-range check - had no join-condition or
# LEFT/RIGHT handling whatsoever: it built a bare, unconditional cross
# product of every (source, join) row pair, leaving `join()`'s caller to
# apply the *entire* ON-clause as a blanket post-join filter. For an INNER
# join that's equivalent to a real inner join (filter a cross product), so
# it went unnoticed; for a LEFT/RIGHT join it's exactly the same bug already
# fixed for `hash_join` in `test_left_join_residual_condition_preserves_unmatched_rows`
# above - the filter evaluates the condition against rows that were never
# matched at all, gets NULL/false, and drops the outer row entirely instead
# of NULL-padding it. Found stress-testing sqlcov against a real Athena CTAS
# joining two derived tables (one filtered to "most recent row per account",
# one to "earliest row per account" via a separate CTE) via `LEFT JOIN ...
# ON new.open_date BETWEEN main.closure_date AND main.closure_date +
# INTERVAL '5' DAY AND main.id <> new.id` - no equi-key at all, so every
# account whose closure never lined up with another account's opening
# vanished from the output entirely, instead of surviving with the joined
# columns NULL. Fixed by giving `nested_loop_join` the same LEFT/RIGHT
# NULL-padding treatment as `hash_join` (via the same `_residual_match_fn`
# helper), testing the full condition per `(a_row, b_row)` pair directly
# instead of bucketing by a key that doesn't exist here.
def test_nested_loop_join_preserves_unmatched_outer_rows():
    sql = (
        "SELECT a.id AS a_id, b.id AS b_id "
        "FROM (SELECT id, closure_date FROM a) AS a "
        "LEFT JOIN (SELECT id, open_date FROM b) AS b "
        "ON b.open_date > a.closure_date"
    )
    tables = {
        "a": [{"id": 1, "closure_date": 10}, {"id": 2, "closure_date": 30}],
        "b": [{"id": 3, "open_date": 20}],
    }
    res = execute(sql, tables=tables, dialect="presto")
    rows = {row[0]: row for row in res.rows}
    assert rows[1] == (1, 3)
    assert rows[2] == (2, None)


# --- Fixed: a window function alongside GROUP BY ran before aggregation ----
# `planner.py` used to build the `Aggregate` step depending on a preceding
# `Window` step whenever the SELECT list had both a GROUP BY and a window
# function - regardless of whether the window function's arguments were
# themselves GROUP BY keys (or otherwise already available
# post-aggregation). That's backwards for this case: standard SQL evaluates
# a window function *after* GROUP BY/aggregation - operating over the
# query's grouped result set - not over the raw pre-aggregation rows.
# Concretely, `PythonExecutor.aggregate` (sqlglot/executor/python.py) ended
# up trying to read the window column (here `rn`) directly off its own
# per-*row* input during `_project_and_filter`, but the column was computed
# by the *preceding* Window step over ungrouped rows and never survived into
# the Aggregate step's own output columns - `KeyError` on the window
# column's name, not a wrong-value bug. Found stress-testing sqlcov against
# a real Athena CTAS: a self-join aggregated with `ARRAY_AGG`/`GROUP BY 1, 2,
# 3` that also computed `ROW_NUMBER() OVER (ORDER BY <the same three GROUP
# BY key expressions>) AS group_no` in the same SELECT - this crashed
# outright (not merely undercounted) the moment upstream fixture data made
# the self-join actually produce rows, having gone unnoticed while that
# join's input was always empty. Fixed by building `Aggregate` from the
# pre-window step and `Window` *after* it (reversing the dependency) whenever
# a GROUP BY or aggregation is present, rewriting each window body's column
# references through the same GROUP-BY-key-to-aggregate-alias `intermediate`
# map already applied to `projections`/`aggregate.condition` - so e.g.
# `ORDER BY <group-by key expression>` resolves against the aggregated
# output instead of rows that no longer exist by the time Window runs.
def test_window_function_over_group_by_keys_runs_after_aggregation():
    res = execute(
        "SELECT t.a, COUNT(*) AS c, ROW_NUMBER() OVER (ORDER BY t.a) AS rn FROM t GROUP BY 1",
        tables={"t": [{"a": 1}, {"a": 1}, {"a": 2}]},
        dialect="presto",
    )
    rows = {row[0]: row for row in res.rows}
    assert rows[1] == (1, 2, 1)
    assert rows[2] == (2, 1, 2)


# --- Fixed: a window ORDER BY column with two or more NULLs crashed --------
# `PythonExecutor.window` (sqlglot/executor/python.py) sorted a partition's
# rows for its ORDER BY using the idiom `(v is None, v)` per column - meant
# to keep NULLs from ever being compared against a real value (`5 < None`
# has no defined ordering in Python). But when two rows both have `v is
# None` for the same column, their `(True, None)` keys are only *fully*
# equal - `True == True` and `None == None` - so tuple comparison never
# needs to fall back to `<`; in practice, though, `sort()`'s internal
# comparisons don't always hit that fully-equal shortcut before trying `<`
# on a still-tied prefix, so a partition with more than one NULL in the same
# ORDER BY column intermittently raised `TypeError: '<' not supported
# between instances of 'NoneType' and 'NoneType'`. A DESC column had it
# worse: its value arrives already wrapped in `reverse_key` (used to invert
# `<`), whose own `__lt__` did `other.obj < self.obj` unconditionally - no
# NULL check at all, so it broke on *any* NULL, not just two. Found
# stress-testing sqlcov against a real Athena CTAS: a Hogan-side account
# table with genuinely unset dates being ranked with `DENSE_RANK() OVER
# (ORDER BY closure_date)` - one NULL closure date was already a crash
# (via `reverse_key`), and this surfaced only once upstream fixture changes
# made that step run over real, non-empty, NULL-containing data for the
# first time. Fixed by replacing the tuple idiom with a dedicated
# `_NullSafeKey` class (NULLS LAST, matching common defaults like Trino's)
# that never compares two `None`s via `<`, and by making `reverse_key` itself
# NULL-aware (NULLS FIRST, the mirror image for a reversed/DESC column)
# instead of assuming its wrapped value is never `None`.
def test_window_order_by_with_multiple_nulls_does_not_crash():
    tables = {"t": [{"g": 1, "a": None}, {"g": 1, "a": None}, {"g": 1, "a": 5}]}

    asc = execute(
        "SELECT a, ROW_NUMBER() OVER (PARTITION BY g ORDER BY a) AS rn FROM t",
        tables=tables,
        dialect="presto",
    )
    asc_by_rn = {row[1]: row[0] for row in asc.rows}
    assert asc_by_rn[1] == 5
    assert asc_by_rn[2] is None
    assert asc_by_rn[3] is None

    desc = execute(
        "SELECT a, ROW_NUMBER() OVER (PARTITION BY g ORDER BY a DESC) AS rn FROM t",
        tables=tables,
        dialect="presto",
    )
    desc_by_rn = {row[1]: row[0] for row in desc.rows}
    assert desc_by_rn[1] is None
    assert desc_by_rn[2] is None
    assert desc_by_rn[3] == 5
