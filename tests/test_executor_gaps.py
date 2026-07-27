"""Regression canaries for gaps in sqlglot's Python executor itself (as
opposed to gaps in sqlcov's own coverage tracking - see test_sqlcov_gaps.py) -
found by stress-testing sqlcov against a large production Athena query (not
included in this repo).

Every test asserts the *correct*, desired result. Today each one xfails: some
because sqlglot raises (a missing function, an unhandled AST node), one
because sqlglot silently returns a *wrong* value with no error at all. If a
sqlglot upgrade closes a gap, that test XPASSes and - because these are marked
``strict=True`` - the suite fails. That is the intended signal: drop the
marker, note the fix in docs/design.md, and check whether sqlcov's own
coverage surface (predicates/CASE arms) should widen to make use of it.

Confirmed against sqlglot 30.13.0.
"""

from __future__ import annotations

import datetime

import pytest
from sqlglot.executor import execute

# --- Window functions: entirely unsupported ---------------------------------
# exp.Window has no PythonGenerator transform, so it falls through to the
# default, dialect-agnostic SQL generator, which emits the literal
# "FUNC(...) OVER (...)" text. That text is then compiled as Python source,
# which isn't valid Python - every window function fails the same way,
# regardless of which one it is or whether it's an aggregate reused as one.

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
@pytest.mark.xfail(
    strict=True, reason="exp.Window has no PythonGenerator transform: SyntaxError on OVER (...)"
)
def test_window_function(sql, tables, expected):
    res = execute(sql, tables=tables, dialect="presto")
    assert sorted(res.rows) == sorted(expected)


# --- Aggregate FILTER clause -------------------------------------------------
# Fixed: the planner now rewrites `AGG(x) FILTER (WHERE cond)` into
# `AGG(CASE WHEN cond THEN x END)` before operand extraction, since every ENV
# aggregator already ignores `None`s and exp.Filter itself has no
# PythonGenerator transform.


def test_aggregate_filter_clause():
    res = execute(
        "SELECT COUNT(*) FILTER (WHERE a > 1) AS c FROM t",
        tables={"t": [{"a": 1}, {"a": 2}, {"a": 3}]},
        dialect="presto",
    )
    assert list(res.rows) == [(2,)]


# --- Parameterized CAST types ------------------------------------------------
# Fixed: PythonGenerator.TRANSFORMS[exp.Cast] now emits the type's
# precision/scale or length as extra positional args to `CAST(...)` instead of
# stringifying them into the type name, and env.py's `cast()` uses them to
# round DECIMAL-family values and truncate TEXT-family values.

_PARAMETERIZED_CAST_CASES = [
    pytest.param("CAST(a AS DECIMAL(18, 2))", {"a": 1.256}, 1.26, id="decimal_with_precision"),
    pytest.param("CAST(s AS VARCHAR(5))", {"s": "hello world"}, "hello", id="varchar_with_length"),
]


@pytest.mark.parametrize("expr, row, expected", _PARAMETERIZED_CAST_CASES)
def test_cast_to_parameterized_type(expr, row, expected):
    res = execute(f"SELECT {expr} AS x FROM t", tables={"t": [row]}, dialect="presto")
    assert list(res.rows) == [(expected,)]


# --- Bare (unparameterized) DECIMAL cast ------------------------------------
# Fixed: env.py's `cast()` now treats the whole DECIMAL family as a float type
# (like FLOAT/DOUBLE) instead of falling into the NUMERIC_TYPES branch that
# truncates via `int(this)`.


def test_cast_to_bare_decimal_preserves_fraction():
    res = execute(
        "SELECT CAST(a AS DECIMAL) AS x FROM t", tables={"t": [{"a": 1.5}]}, dialect="presto"
    )
    assert list(res.rows) == [(1.5,)]


# --- DATE_DIFF: no error, but the unit argument is silently ignored --------
# ENV["DATEDIFF"] is `lambda this, expression, *_: (this - expression).days`
# - the requested unit ('month', 'hour', ...) is swallowed by `*_` and the
# result is always a day count. A predicate like `DATE_DIFF('month', a, b) > 3`
# will silently evaluate against the wrong number.


def test_date_diff_honors_unit():
    res = execute(
        "SELECT DATE_DIFF('month', CAST(d1 AS DATE), CAST(d2 AS DATE)) AS x FROM t",
        tables={"t": [{"d1": "2024-01-01", "d2": "2024-03-01"}]},
        dialect="presto",
    )
    assert list(res.rows) == [(2,)]


# --- UNNEST in FROM: a planner bug, not just a missing function ------------
# This fails before any function lookup happens - the planner's join step
# assumes every FROM-clause source is a real table and chokes on `exp.Unnest`
# itself (`'Unnest' object has no attribute 'parts'`). Distinct from the
# NameError-class gaps below because it's structural: no scalar function
# workaround fixes it.


@pytest.mark.xfail(
    strict=True, reason="UNNEST in FROM is not handled by the planner: AttributeError on exp.Unnest"
)
def test_unnest_in_from_clause():
    res = execute(
        "SELECT x FROM t CROSS JOIN UNNEST(ARRAY[1, 2, 3]) AS u(x)",
        tables={"t": [{"a": 1}]},
        dialect="presto",
    )
    assert sorted(res.rows) == [(1,), (2,), (3,)]


# --- Missing scalar functions: fixed by registering them in ENV -----------
# Fixed: sqlglot.executor.env.ENV now implements DATETRUNC, DATEADD,
# DAYOFWEEKISO/LASTDAY, SIGN, ISNAN, ARRAYCONTAINS, ENCODE, SPLITPART,
# TRYCAST, GREATEST and LEAST (plus PythonGenerator.TRANSFORMS entries for
# exp.DateAdd and exp.TryCast, which previously emitted unquoted unit names
# or fell through to the wrong TRANSFORMS entry).

_FIXED_FUNCTION_CASES = [
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


@pytest.mark.parametrize("expr, row, expected", _FIXED_FUNCTION_CASES)
def test_missing_function(expr, row, expected):
    res = execute(f"SELECT {expr} AS x FROM t", tables={"t": [row]}, dialect="presto")
    assert list(res.rows) == [(expected,)]


# --- Missing array functions: no longer blocked by the empty table schema --
# ENV now has ARRAYSORT, ARRAYDISTINCT, ARRAYS_OVERLAP, ARRAYMIN, ARRAYSIZE
# and FLATTEN, but these cases take no columns from `t`, so the row is `{}`.
# Fixed: execute() now registers a zero-column table in the inferred schema
# with a placeholder column (rather than omitting the table entirely), so the
# schema's `supported_table_args` lines up with the literal table's instead of
# collapsing to `()` and tripping the "Tables must support the same table args
# as schema" check before the plan ever runs.

_MISSING_FUNCTION_CASES = [
    pytest.param("ARRAY_SORT(ARRAY[3, 1, 2])", {}, [1, 2, 3], id="array_sort"),
    pytest.param("ARRAY_DISTINCT(ARRAY[1, 1, 2])", {}, [1, 2], id="array_distinct"),
    pytest.param("ARRAYS_OVERLAP(ARRAY[1, 2], ARRAY[2, 3])", {}, True, id="arrays_overlap"),
    pytest.param("ARRAY_MIN(ARRAY[3, 1, 2])", {}, 1, id="array_min"),
    pytest.param("CARDINALITY(ARRAY[1, 2, 3])", {}, 3, id="cardinality"),
    pytest.param("FLATTEN(ARRAY[ARRAY[1, 2], ARRAY[3]])", {}, [1, 2, 3], id="flatten"),
]


@pytest.mark.parametrize("expr, row, expected", _MISSING_FUNCTION_CASES)
def test_missing_function_blocked_by_empty_table_schema(expr, row, expected):
    res = execute(f"SELECT {expr} AS x FROM t", tables={"t": [row]}, dialect="presto")
    assert list(res.rows) == [(expected,)]
