import calendar
import datetime
import inspect
import math
import re
import statistics
from functools import wraps

from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get


_XXH64_P1 = 0x9E3779B185EBCA87
_XXH64_P2 = 0xC2B2AE3D27D4EB4F
_XXH64_P3 = 0x165667B19E3779F9
_XXH64_P4 = 0x85EBCA77C2B2AE63
_XXH64_P5 = 0x27D4EB2F165667C5
_XXH64_MASK = 0xFFFFFFFFFFFFFFFF


def _xxh64_rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _XXH64_MASK


def xxhash64(data: bytes, seed: int = 0) -> int:
    """Pure-Python xxHash64 (https://github.com/Cyan4973/xxHash), no third-party dependency."""
    length = len(data)
    i = 0

    if length >= 32:
        v1 = (seed + _XXH64_P1 + _XXH64_P2) & _XXH64_MASK
        v2 = (seed + _XXH64_P2) & _XXH64_MASK
        v3 = seed & _XXH64_MASK
        v4 = (seed - _XXH64_P1) & _XXH64_MASK

        while i <= length - 32:
            for j in range(4):
                lane = int.from_bytes(data[i : i + 8], "little")
                v = (v1, v2, v3, v4)[j]
                v = (v + lane * _XXH64_P2) & _XXH64_MASK
                v = _xxh64_rotl(v, 31)
                v = (v * _XXH64_P1) & _XXH64_MASK
                v1, v2, v3, v4 = (v if k == j else prev for k, prev in enumerate((v1, v2, v3, v4)))
                i += 8

        h = (
            _xxh64_rotl(v1, 1)
            + _xxh64_rotl(v2, 7)
            + _xxh64_rotl(v3, 12)
            + _xxh64_rotl(v4, 18)
        ) & _XXH64_MASK

        for v in (v1, v2, v3, v4):
            v = (v * _XXH64_P2) & _XXH64_MASK
            v = _xxh64_rotl(v, 31)
            v = (v * _XXH64_P1) & _XXH64_MASK
            h = ((h ^ v) * _XXH64_P1 + _XXH64_P4) & _XXH64_MASK
    else:
        h = (seed + _XXH64_P5) & _XXH64_MASK

    h = (h + length) & _XXH64_MASK

    while i <= length - 8:
        k1 = int.from_bytes(data[i : i + 8], "little")
        k1 = (k1 * _XXH64_P2) & _XXH64_MASK
        k1 = _xxh64_rotl(k1, 31)
        k1 = (k1 * _XXH64_P1) & _XXH64_MASK
        h = (_xxh64_rotl(h ^ k1, 27) * _XXH64_P1 + _XXH64_P4) & _XXH64_MASK
        i += 8

    if i <= length - 4:
        k1 = int.from_bytes(data[i : i + 4], "little")
        h = (_xxh64_rotl(h ^ ((k1 * _XXH64_P1) & _XXH64_MASK), 23) * _XXH64_P2 + _XXH64_P3) & _XXH64_MASK
        i += 4

    while i < length:
        h = (_xxh64_rotl(h ^ (data[i] * _XXH64_P5), 11) * _XXH64_P1) & _XXH64_MASK
        i += 1

    h ^= h >> 33
    h = (h * _XXH64_P2) & _XXH64_MASK
    h ^= h >> 29
    h = (h * _XXH64_P3) & _XXH64_MASK
    h ^= h >> 32
    return h


class reverse_key:
    """Sort key for a DESC ORDER BY column: reverses the wrapped value's
    natural ordering. NULLS FIRST for a descending column (matching the
    common default, e.g. Trino's) - the mirror image of ascending's NULLS
    LAST - so ``None`` needs its own comparison branch here rather than
    falling into ``other.obj < self.obj``, which raises the moment either
    side is ``None`` (`'<' not supported between instances of ... and
    'NoneType'`), not just when both are.
    """

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        if self.obj is None:
            return other.obj is not None
        if other.obj is None:
            return False
        return other.obj < self.obj


def filter_nulls(func, empty_null=True):
    @wraps(func)
    def _func(values):
        filtered = tuple(v for v in values if v is not None)
        if not filtered and empty_null:
            return None
        return func(filtered)

    return _func


def coerce_date_and_datetime(func):
    """
    Decorator for binary comparison functions that lets a `datetime.date` compare
    against a `datetime.datetime` by widening the date to midnight on that day, the
    same implicit coercion Presto/Athena apply to DATE/TIMESTAMP comparisons.
    """

    @wraps(func)
    def _func(this, e):
        if (
            isinstance(this, datetime.date)
            and not isinstance(this, datetime.datetime)
            and isinstance(e, datetime.datetime)
        ):
            this = datetime.datetime(this.year, this.month, this.day)
        elif (
            isinstance(e, datetime.date)
            and not isinstance(e, datetime.datetime)
            and isinstance(this, datetime.datetime)
        ):
            e = datetime.datetime(e.year, e.month, e.day)
        return func(this, e)

    return _func


def null_if_any(*required):
    """
    Decorator that makes a function return `None` if any of the `required` arguments are `None`.

    This also supports decoration with no arguments, e.g.:

        @null_if_any
        def foo(a, b): ...

    In which case all arguments are required.
    """
    f = None
    if len(required) == 1 and callable(required[0]):
        f = required[0]
        required = ()

    def decorator(func):
        if required:
            required_indices = [
                i for i, param in enumerate(inspect.signature(func).parameters) if param in required
            ]

            def predicate(*args):
                return any(args[i] is None for i in required_indices)

        else:

            def predicate(*args):
                return any(a is None for a in args)

        @wraps(func)
        def _func(*args):
            if predicate(*args):
                return None
            return func(*args)

        return _func

    if f:
        return decorator(f)

    return decorator


@null_if_any("this", "substr")
def str_position(this, substr, position=None):
    position = position - 1 if position is not None else position
    return this.find(substr, position) + 1


@null_if_any("this")
def substring(this, start=None, length=None):
    if start is None:
        return this
    elif start == 0:
        return ""
    elif start < 0:
        start = len(this) + start
    else:
        start -= 1

    end = None if length is None else start + length

    return this[start:end]


@null_if_any("this", "delimiter", "part")
def split_part(this, delimiter, part):
    parts = this.split(delimiter)
    index = part - 1 if part > 0 else part
    if -len(parts) <= index < len(parts):
        return parts[index]
    return ""


DECIMAL_TYPES = {
    exp.DType.DECIMAL,
    exp.DType.DECIMAL32,
    exp.DType.DECIMAL64,
    exp.DType.DECIMAL128,
    exp.DType.DECIMAL256,
    exp.DType.BIGDECIMAL,
    exp.DType.UDECIMAL,
    exp.DType.MONEY,
    exp.DType.SMALLMONEY,
    exp.DType.DECFLOAT,
}


@null_if_any("this", "to")
def cast(this, to, *params):
    if to == exp.DType.DATE:
        if isinstance(this, datetime.datetime):
            return this.date()
        if isinstance(this, datetime.date):
            return this
        if isinstance(this, str):
            return datetime.date.fromisoformat(this)
    if to == exp.DType.TIME:
        if isinstance(this, datetime.datetime):
            return this.time()
        if isinstance(this, datetime.time):
            return this
        if isinstance(this, str):
            return datetime.time.fromisoformat(this)
    if to in (exp.DType.DATETIME, exp.DType.TIMESTAMP):
        if isinstance(this, datetime.datetime):
            return this
        if isinstance(this, datetime.date):
            return datetime.datetime(this.year, this.month, this.day)
        if isinstance(this, str):
            return datetime.datetime.fromisoformat(this)
    if to == exp.DType.BOOLEAN:
        return bool(this)
    if to in exp.DataType.TEXT_TYPES:
        this = str(this)
        if params:
            this = this[: int(params[0])]
        return this
    if to in {exp.DType.FLOAT, exp.DType.DOUBLE} | DECIMAL_TYPES:
        this = float(this)
        if params:
            this = round(this, int(params[-1]))
        return this
    if to in exp.DataType.NUMERIC_TYPES:
        return int(this)
    raise NotImplementedError(f"Casting {this} to '{to}' not implemented.")


@null_if_any("this", "to")
def try_cast(this, to, *params):
    try:
        return cast(this, to, *params)
    except (NotImplementedError, TypeError, ValueError):
        return None


def date_(this=None):
    if this is None:
        return datetime.date.today()
    if isinstance(this, datetime.datetime):
        return this.date()
    if isinstance(this, datetime.date):
        return this
    if isinstance(this, str):
        return datetime.date.fromisoformat(this)
    raise NotImplementedError(f"DATE does not support argument '{this}'.")


def ordered(this, desc, nulls_first):
    if desc:
        return reverse_key(this)
    return this


class _MonthsDelta:
    """A `timedelta`-like duration for MONTH/QUARTER/YEAR intervals.

    These units aren't a fixed number of days (`datetime.timedelta` has no
    `months` parameter), so adding/subtracting one has to shift the calendar
    month-by-month (via `_add_months`) instead.
    """

    __slots__ = ("months",)

    def __init__(self, months):
        self.months = months

    def __neg__(self):
        return _MonthsDelta(-self.months)

    def __add__(self, other):
        return _as_date(other, _add_months(_as_datetime(other), self.months))

    __radd__ = __add__

    def __rsub__(self, other):
        return _as_date(other, _add_months(_as_datetime(other), -self.months))


@null_if_any
def interval(this, unit):
    unit = unit.lower()
    if unit in ("year", "quarter", "month"):
        months = int(float(this)) * (12 if unit == "year" else 3 if unit == "quarter" else 1)
        return _MonthsDelta(months)

    plural = (unit + "s").upper()
    if plural in Generator.TIME_PART_SINGULARS:
        unit = plural
    return datetime.timedelta(**{unit.lower(): float(this)})


def _as_datetime(value):
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        return datetime.datetime.fromisoformat(value)
    return datetime.datetime(value.year, value.month, value.day)


def _month_diff(start, end):
    start_dt = _as_datetime(start)
    end_dt = _as_datetime(end)
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

    if (end_dt.day, end_dt.hour, end_dt.minute, end_dt.second, end_dt.microsecond) < (
        start_dt.day,
        start_dt.hour,
        start_dt.minute,
        start_dt.second,
        start_dt.microsecond,
    ):
        months -= 1

    return months


_DATEDIFF_UNIT_SECONDS = {
    "week": 604800,
    "day": 86400,
    "hour": 3600,
    "minute": 60,
    "second": 1,
    "millisecond": 0.001,
    "microsecond": 0.000001,
}


@null_if_any("this", "expression")
def datediff(this, expression, unit="day"):
    unit = unit.lower()

    if unit in ("year", "quarter", "month"):
        months = _month_diff(expression, this)
        if unit == "year":
            return int(months / 12)
        if unit == "quarter":
            return int(months / 3)
        return months

    unit_seconds = _DATEDIFF_UNIT_SECONDS.get(unit)
    if unit_seconds is None:
        raise NotImplementedError(f"DATEDIFF does not support unit '{unit}'.")

    seconds = (_as_datetime(this) - _as_datetime(expression)).total_seconds()
    return int(seconds / unit_seconds)


def _as_date(this, result):
    if isinstance(this, str):
        return result.date() if len(this) <= len("YYYY-MM-DD") else result
    if isinstance(this, datetime.date) and not isinstance(this, datetime.datetime):
        return result.date()
    return result


def _add_months(dt, months):
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


@null_if_any("unit", "this")
def datetrunc(unit, this):
    unit = unit.lower()
    dt = _as_datetime(this)

    if unit == "year":
        result = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == "quarter":
        quarter_month = ((dt.month - 1) // 3) * 3 + 1
        result = dt.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == "month":
        result = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif unit == "week":
        result = (dt - datetime.timedelta(days=dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif unit == "day":
        result = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif unit == "hour":
        result = dt.replace(minute=0, second=0, microsecond=0)
    elif unit == "minute":
        result = dt.replace(second=0, microsecond=0)
    elif unit == "second":
        result = dt.replace(microsecond=0)
    else:
        raise NotImplementedError(f"DATE_TRUNC does not support unit '{unit}'.")

    return _as_date(this, result)


@null_if_any("this", "unit")
def timestamptrunc(this, unit):
    return datetrunc(unit, this)


@null_if_any("this", "expression")
def dateadd(this, expression, unit="day"):
    unit = unit.lower()
    dt = _as_datetime(this)

    if unit in ("year", "quarter", "month"):
        months = expression * (12 if unit == "year" else 3 if unit == "quarter" else 1)
        result = _add_months(dt, months)
    else:
        unit_seconds = _DATEDIFF_UNIT_SECONDS.get(unit)
        if unit_seconds is None:
            raise NotImplementedError(f"DATE_ADD does not support unit '{unit}'.")
        result = dt + datetime.timedelta(seconds=unit_seconds * expression)

    return _as_date(this, result)


@null_if_any("this")
def lastday(this, unit=None):
    dt = _as_datetime(this)
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return _as_date(this, dt.replace(day=last_day))


@null_if_any("this", "expression")
def arraytostring(this, expression, null=None):
    return expression.join(x for x in (x if x is not None else null for x in this) if x is not None)


@null_if_any("this", "expression")
def jsonextract(this, expression):
    for path_segment in expression:
        if isinstance(this, dict):
            this = this.get(path_segment)
        elif isinstance(this, list) and is_int(path_segment):
            this = seq_get(this, int(path_segment))
        else:
            raise NotImplementedError(f"Unable to extract value for {this} at {path_segment}.")

        if this is None:
            break

    return this


ENV = {
    "exp": exp,
    # aggs
    "ARRAYAGG": list,
    "ARRAYUNIQUEAGG": filter_nulls(lambda acc: list(set(acc))),
    "AVG": filter_nulls(statistics.fmean if PYTHON_VERSION >= (3, 8) else statistics.mean),  # type: ignore
    "COUNT": filter_nulls(lambda acc: sum(1 for _ in acc), False),
    "MAX": filter_nulls(max),
    "MIN": filter_nulls(min),
    "SUM": filter_nulls(sum),
    # scalar functions
    "ABS": null_if_any(lambda this: abs(this)),
    "ADD": null_if_any(lambda e, this: e + this),
    "ARRAYANY": null_if_any(lambda arr, func: any(func(e) for e in arr)),
    "ARRAYCONTAINS": null_if_any(lambda arr, e: e in arr),
    "ARRAYDISTINCT": null_if_any(lambda arr: list(dict.fromkeys(arr))),
    "ARRAYFILTER": null_if_any(lambda arr, func: [e for e in arr if func(e)]),
    "ARRAYMIN": null_if_any(min),
    "ARRAYSIZE": null_if_any(lambda arr, *_: len(arr)),
    "ARRAYSORT": null_if_any(lambda arr, *_: sorted(arr)),
    "ARRAYS_OVERLAP": null_if_any(lambda a, b: bool(set(a) & set(b))),
    "ARRAYTOSTRING": arraytostring,
    "BETWEEN": null_if_any(lambda this, low, high: low <= this and this <= high),
    "BITWISEAND": null_if_any(lambda this, e: this & e),
    "BITWISELEFTSHIFT": null_if_any(lambda this, e: this << e),
    "BITWISEOR": null_if_any(lambda this, e: this | e),
    "BITWISERIGHTSHIFT": null_if_any(lambda this, e: this >> e),
    "BITWISEXOR": null_if_any(lambda this, e: this ^ e),
    "CAST": cast,
    "COALESCE": lambda *args: next((a for a in args if a is not None), None),
    "CONCAT": null_if_any(lambda *args: "".join(args)),
    "SAFECONCAT": null_if_any(lambda *args: "".join(str(arg) for arg in args)),
    "CONCATWS": null_if_any(lambda this, *args: this.join(args)),
    "DATE": date_,
    "DATEADD": dateadd,
    "DATEDIFF": datediff,
    "DATESTRTODATE": null_if_any(lambda arg: datetime.date.fromisoformat(arg)),
    "DATETRUNC": datetrunc,
    "TIMESTAMPTRUNC": timestamptrunc,
    "DAYOFWEEKISO": null_if_any(lambda arg: arg.isoweekday()),
    "DIV": null_if_any(lambda e, this: e / this),
    "TYPEDDIV": null_if_any(lambda e, this: int(e / this)),
    "DOT": null_if_any(lambda e, this: e[this]),
    "DPIPE": null_if_any(lambda this, e: this + e),
    "ENCODE": null_if_any(lambda this, charset="utf-8": this.encode(charset)),
    "FROM_BIG_ENDIAN_64": null_if_any(lambda this: int.from_bytes(this, "big", signed=True)),
    "EQ": null_if_any(lambda this, e: this == e),
    "EXTRACT": null_if_any(lambda this, e: getattr(e, this)),
    "FLATTEN": null_if_any(lambda arr: [x for sub in arr for x in sub]),
    "GREATEST": null_if_any(lambda *args: max(args)),
    "GT": null_if_any(coerce_date_and_datetime(lambda this, e: this > e)),
    "GTE": null_if_any(coerce_date_and_datetime(lambda this, e: this >= e)),
    "IF": lambda predicate, true, false: true if predicate else false,
    "INTDIV": null_if_any(lambda e, this: e // this),
    "INTERVAL": interval,
    "ISNAN": null_if_any(math.isnan),
    "JSONEXTRACT": jsonextract,
    "LASTDAY": lastday,
    "LEAST": null_if_any(lambda *args: min(args)),
    "LEFT": null_if_any(lambda this, e: this[:e]),
    "LIKE": null_if_any(
        lambda this, e: bool(re.match(e.replace("_", ".").replace("%", ".*"), this))
    ),
    "LOWER": null_if_any(lambda arg: arg.lower()),
    "LT": null_if_any(coerce_date_and_datetime(lambda this, e: this < e)),
    "LTE": null_if_any(coerce_date_and_datetime(lambda this, e: this <= e)),
    "MAP": null_if_any(lambda *args: dict(zip(*args))),  # type: ignore
    "MOD": null_if_any(lambda e, this: e % this),
    "MUL": null_if_any(lambda e, this: e * this),
    "NEQ": null_if_any(lambda this, e: this != e),
    "ORD": null_if_any(ord),
    "ORDERED": ordered,
    "POW": pow,
    "RIGHT": null_if_any(lambda this, e: this[-e:]),
    "ROUND": null_if_any(lambda this, decimals=None, truncate=None: round(this, ndigits=decimals)),
    "SIGN": null_if_any(lambda this: (this > 0) - (this < 0)),
    "SPLITPART": split_part,
    "STRPOSITION": str_position,
    "SUB": null_if_any(lambda e, this: e - this),
    "SUBSTRING": substring,
    "TIMESTRTOTIME": null_if_any(lambda arg: datetime.datetime.fromisoformat(arg)),
    "TRYCAST": try_cast,
    "UPPER": null_if_any(lambda arg: arg.upper()),
    "XXHASH64": null_if_any(lambda this: xxhash64(this).to_bytes(8, "big")),
    "YEAR": null_if_any(lambda arg: arg.year),
    "MONTH": null_if_any(lambda arg: arg.month),
    "DAY": null_if_any(lambda arg: arg.day),
    "CURRENTDATETIME": datetime.datetime.now,
    "CURRENTTIMESTAMP": datetime.datetime.now,
    "CURRENTTIME": datetime.datetime.now,
    "CURRENTDATE": datetime.date.today,
    "STRFTIME": null_if_any(lambda fmt, arg: datetime.datetime.fromisoformat(arg).strftime(fmt)),
    "STRTOTIME": null_if_any(lambda arg, format: datetime.datetime.strptime(arg, format)),
    "TRIM": null_if_any(lambda this, e=None: this.strip(e)),
    "STRUCT": lambda *args: {
        args[x]: args[x + 1]
        for x in range(0, len(args), 2)
        if (args[x + 1] is not None and args[x] is not None)
    },
    "UNIXTOTIME": null_if_any(
        lambda arg: datetime.datetime.fromtimestamp(arg, datetime.timezone.utc)
    ),
}
