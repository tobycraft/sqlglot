import collections
import itertools
import math

from sqlglot import exp, planner, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import ExecuteError
from sqlglot.executor.context import Context
from sqlglot.executor.env import ENV
from sqlglot.executor.table import RowReader, Table
from sqlglot.generators.python import PythonGenerator


class _NullSafeKey:
    """A sort key for an ascending ORDER BY column - NULLS LAST, matching the
    common default (e.g. Trino's) - without ever calling the wrapped value's
    own comparison against another ``None``. Unlike the once-used ``(v is
    None, v)`` idiom, which still compares `v` itself as a tuple-equality
    tiebreaker whenever two rows' `v is None` flags match, so two `None`s (a
    common case: multiple rows sharing an unset ORDER BY column) raised
    `TypeError: '<' not supported between instances of 'NoneType' and
    'NoneType'`. See ``PythonExecutor.window``.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        if self.value is None:
            return False
        if other.value is None:
            return True
        return self.value < other.value


class _AliasedRowReader:
    """A read-only view of another `RowReader`'s *current* row, scoped to just
    one alias' own `column_range` - lets two aliases of the same underlying
    columns (e.g. two joins of the same reference table) each resolve a
    same-named column to their own value, while `row` always reflects
    whatever row `source` is currently on (never a frozen copy), so it stays
    correct as the driving iteration advances - see `PythonExecutor.window`.
    """

    def __init__(self, source, columns, column_range):
        self._source = source
        self.columns = {
            column: i for i, column in enumerate(columns) if not column_range or i in column_range
        }

    @property
    def row(self):
        return self._source.row

    def __getitem__(self, column):
        return self.row[self.columns[column]]


def _hashable_key(value):
    """A join/group key may legitimately be an ARRAY (Presto/Athena allow array
    equality, e.g. in an equi-join), which the executor represents as a Python
    ``list`` - not hashable, so it can't be used directly as a dict key. Recurse
    into lists (and tuples, for nested arrays) and convert them to tuples so the
    resulting key is always hashable, without changing equality semantics."""
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_key(v) for v in value)
    return value


class PythonExecutor:
    def __init__(self, env=None, tables=None, recursion_limit=10_000):
        self.generator = Python().generator(identify=True, comments=False)
        self.env = {**ENV, **(env or {})}
        self.tables = tables or {}
        self.recursion_limit = recursion_limit

    def execute(self, plan):
        return self._run(plan.root).tables[plan.root.name]

    def _run(self, root, seed=None):
        """
        Runs the Step DAG rooted at `root` to completion and returns its Context.

        `seed` pre-populates `contexts` for specific Step objects (e.g. a
        RecursiveCTE's own working-table placeholder, or an outer CTE it
        depends on) - those nodes are treated as already finished and are
        never dispatched generically, which lets `recursive_cte` re-run the
        same `body` subtree against a different table on each iteration.
        """
        dag = {}
        nodes = {root}

        while nodes:
            node = nodes.pop()
            dag[node] = set(node.dependencies)
            nodes.update(node.dependencies)

        contexts = dict(seed or {})
        finished = set(contexts)
        queue = {
            node
            for node, deps in dag.items()
            if node not in finished and all(d in contexts for d in deps)
        }

        while queue:
            node = queue.pop()
            try:
                if node in contexts:
                    pass
                else:
                    context = self.context(
                        {
                            name: table
                            for dep in node.dependencies
                            for name, table in contexts[dep].tables.items()
                        }
                    )

                    if isinstance(node, planner.Scan):
                        contexts[node] = self.scan(node, context)
                    elif isinstance(node, planner.Window):
                        contexts[node] = self.window(node, context)
                    elif isinstance(node, planner.Aggregate):
                        contexts[node] = self.aggregate(node, context)
                    elif isinstance(node, planner.Join):
                        contexts[node] = self.join(node, context)
                    elif isinstance(node, planner.Sort):
                        contexts[node] = self.sort(node, context)
                    elif isinstance(node, planner.SetOperation):
                        contexts[node] = self.set_operation(node, context)
                    elif isinstance(node, planner.RecursiveCTE):
                        contexts[node] = self.recursive_cte(
                            node, {dep: contexts[dep] for dep in node.dependencies}
                        )
                    else:
                        raise NotImplementedError

                finished.add(node)

                for dep in node.dependents:
                    if dep in dag and all(d in contexts for d in dep.dependencies):
                        queue.add(dep)

                for dep in node.dependencies:
                    if dep not in (seed or {}) and all(d in finished for d in dep.dependents):
                        contexts.pop(dep)
            except Exception as e:
                raise ExecuteError(f"Step '{node.id}' failed: {e}") from e

        return contexts[root]

    def recursive_cte(self, step, foreign_seed):
        accumulated = self._run(step.anchor, seed=foreign_seed).tables[step.anchor.name]
        working = accumulated

        for _ in range(self.recursion_limit):
            seed = {**foreign_seed, step.ref: self.context({step.name: working})}
            new_rows = self._run(step.body, seed=seed).tables[step.body.name]

            rows = new_rows.rows
            if step.distinct:
                seen = set(accumulated.rows)
                rows = [row for row in rows if row not in seen]

            if not rows:
                break

            accumulated.rows.extend(rows)
            working = Table(new_rows.columns, rows)
        else:
            raise ExecuteError(
                f"Recursive CTE '{step.name}' exceeded {self.recursion_limit} iterations"
            )

        return self.context({step.name: accumulated})

    def generate(self, expression):
        """Convert a SQL expression into literal Python code and compile it into bytecode."""
        if not expression:
            return None

        sql = self.generator.generate(expression)
        return compile(sql, sql, "eval", optimize=2)

    def generate_tuple(self, expressions):
        """Convert an array of SQL expressions into tuple of Python byte code."""
        if not expressions:
            return tuple()
        return tuple(self.generate(expression) for expression in expressions)

    def context(self, tables):
        return Context(tables, env=self.env)

    def table(self, expressions):
        return Table(
            expression.alias_or_name if isinstance(expression, exp.Expr) else expression
            for expression in expressions
        )

    def scan(self, step, context):
        source = step.source

        if source and isinstance(source, exp.Expr):
            source = source.name or source.alias

        if source is None:
            context, table_iter = self.static()
        elif source in context:
            if not step.projections and not step.condition:
                return self.context({step.name: context.tables[source]})
            if step.name and step.name != source and step.name not in context.tables:
                # a CTE-level rename (e.g. a CTE that just re-selects from another
                # CTE by name) can leave step.name out of sync with the dependency
                # name (source); alias it onto the same table so projections/
                # condition qualified with the new name still resolve
                context = self.context({step.name: context.tables[source], **context.tables})
            table_iter = context.table_iter(source)
        else:
            context, table_iter = self.scan_table(step)

        return self.context({step.name: self._project_and_filter(context, step, table_iter)})

    def _project_and_filter(self, context, step, table_iter):
        sink = self.table(step.projections if step.projections else context.columns)
        condition = self.generate(step.condition)
        projections = self.generate_tuple(step.projections)

        for reader in table_iter:
            if len(sink) >= step.limit:
                break

            if condition and not context.eval(condition):
                continue

            if projections:
                sink.append(context.eval_tuple(projections))
            else:
                sink.append(reader.row)

        return sink

    def static(self):
        return self.context({}), [RowReader(())]

    def scan_table(self, step):
        if isinstance(step.source, exp.Unnest):
            return self.scan_unnest(step)

        table = self.tables.find(step.source)
        tables = {step.source.alias_or_name: table}
        if step.name and step.name != step.source.alias_or_name:
            # an un-merged derived table reuses its innermost physical Scan step,
            # renamed to the derived table's own alias, so its projections/condition
            # may be qualified with either the physical table's alias or this one
            tables[step.name] = table
        context = self.context(tables)
        return context, iter(table)

    def scan_unnest(self, step):
        unnest = step.source
        static_context = self.context({})
        arrays = [static_context.eval(self.generate(expression)) for expression in unnest.expressions]

        offset = unnest.args.get("offset")
        columns = [column.name for column in unnest.selects] or [
            f"_col_{i}" for i in range(len(arrays) + bool(offset))
        ]

        table = Table(columns)
        for i, values in enumerate(itertools.zip_longest(*arrays)):
            table.append(values + (i + 1,) if offset else values)

        context = self.context({step.name: table})
        return context, iter(table)

    def join(self, step, context):
        source = step.source_name

        source_table = context.tables[source]
        source_context = self.context({source: source_table})
        column_ranges = {source: range(0, len(source_table.columns))}

        for name, join in step.joins.items():
            start = max(r.stop for r in column_ranges.values())

            unnest = join.get("unnest")
            already_filtered = False
            if unnest is not None:
                table = self.lateral_unnest_join(unnest, source_context)
                column_ranges[name] = range(start, len(table.columns))
            else:
                table = context.tables[name]
                column_ranges[name] = range(start, len(table.columns) + start)
                join_context = self.context({name: table})

                if join.get("source_key"):
                    # hash_join already applies join["condition"] itself (see
                    # its docstring) - correctly, for LEFT/RIGHT, *before*
                    # deciding whether a source row has any match at all,
                    # rather than as a blanket filter afterward that would
                    # wrongly strip out the NULL-padded rows it just added.
                    table = self.hash_join(join, source_context, join_context)
                    already_filtered = True
                else:
                    # No extractable equi-join key - nested_loop_join applies
                    # the *entire* condition itself, same reasoning as above.
                    table = self.nested_loop_join(join, source_context, join_context)
                    already_filtered = True

            source_context = self.context(
                {
                    name: Table(table.columns, table.rows, column_range)
                    for name, column_range in column_ranges.items()
                }
            )
            if not already_filtered:
                condition = self.generate(join["condition"])
                if condition:
                    source_context.filter(condition)

        if step.name and step.name != source and step.name not in source_context.tables:
            # a CTE-level rename (a pass-through CTE body whose outermost step is
            # itself a Join, e.g. one with no real joins) can leave step.name out of
            # sync with source_name; alias it onto the same range so projections/
            # condition qualified with the new name still resolve
            source_context = self.context(
                {step.name: source_context.tables[source], **source_context.tables}
            )

        if not step.condition and not step.projections:
            return source_context

        sink = self._project_and_filter(
            source_context,
            step,
            (reader for reader, _ in iter(source_context)),
        )

        if step.projections:
            return self.context({step.name: sink})
        else:
            return self.context(
                {
                    name: Table(table.columns, sink.rows, table.column_range)
                    for name, table in source_context.tables.items()
                }
            )

    def lateral_unnest_join(self, unnest, source_context):
        """Explode an UNNEST(...) that may reference a column of an already-joined
        source (e.g. `CROSS JOIN UNNEST(t.arr)`) against each row of that source,
        the way a LATERAL join would - a plain nested-loop/hash join can't work here
        since the array to explode differs per outer row rather than being a single,
        independently-scannable table.
        """
        offset = unnest.args.get("offset")
        columns = [column.name for column in unnest.selects] or [
            f"_col_{i}" for i in range(len(unnest.expressions) + bool(offset))
        ]
        exprs = self.generate_tuple(unnest.expressions)

        table = Table(source_context.columns + tuple(columns))
        for reader, ctx in source_context:
            # A NULL array (as opposed to an empty one) explodes to zero rows,
            # same as Presto/Trino's UNNEST - without this, zip_longest(*arrays)
            # blows up trying to iterate a None.
            arrays = [ctx.eval(expr) or [] for expr in exprs]
            for i, values in enumerate(itertools.zip_longest(*arrays)):
                table.append(reader.row + (values + (i + 1,) if offset else values))

        return table

    def nested_loop_join(self, join, source_context, join_context):
        """A join with no extractable equi-join key at all (e.g. every ON-clause
        conjunct is a non-equality, like a date-range check) - the entire
        condition is the "residual" `hash_join` would otherwise apply after
        key-bucketing. With no key to bucket by, every source row is tested
        against every join row directly; LEFT/RIGHT NULL-padding follows the
        same rule as `hash_join`: a source row with zero matches still gets
        one output row, padded with NULLs on the other side, instead of
        vanishing - the same regression class `hash_join`'s residual matching
        already guards against (see its docstring), just with no key to skip
        the O(n*m) comparison.
        """
        a_group = [reader.row for reader, _ in source_context]
        b_group = [reader.row for reader, _ in join_context]

        left = join.get("side") == "LEFT"
        right = join.get("side") == "RIGHT"

        table = Table(source_context.columns + join_context.columns)
        null = (None,) * len(join_context.columns if left else source_context.columns)

        condition = join.get("condition")
        matches = None
        if condition is not None and self.generate(condition) != "True":
            matches = self._residual_match_fn(condition, source_context, join_context)

        if left:
            for a_row in a_group:
                real = [a_row + b_row for b_row in b_group if matches is None or matches(a_row, b_row)]
                table.rows.extend(real if real else [a_row + null])
        elif right:
            for b_row in b_group:
                real = [a_row + b_row for a_row in a_group if matches is None or matches(a_row, b_row)]
                table.rows.extend(real if real else [null + b_row])
        else:
            for a_row, b_row in itertools.product(a_group, b_group):
                if matches is None or matches(a_row, b_row):
                    table.append(a_row + b_row)

        return table

    def hash_join(self, join, source_context, join_context):
        source_key = self.generate_tuple(join["source_key"])
        join_key = self.generate_tuple(join["join_key"])
        left = join.get("side") == "LEFT"
        right = join.get("side") == "RIGHT"

        results = collections.defaultdict(lambda: ([], []))

        for reader, ctx in source_context:
            results[_hashable_key(ctx.eval_tuple(source_key))][0].append(reader.row)
        for reader, ctx in join_context:
            results[_hashable_key(ctx.eval_tuple(join_key))][1].append(reader.row)

        table = Table(source_context.columns + join_context.columns)
        nulls = [(None,) * len(join_context.columns if left else source_context.columns)]

        # Any ON-clause conjunct `join_condition` couldn't fold into the hash
        # key above (a non-equality, e.g. a date-range check, or an equality
        # it couldn't isolate to one side) is left over as `join["condition"]`
        # - see `join()`'s docstring-less comment at the call site for why that
        # can't just be a blanket post-join filter for LEFT/RIGHT: a bucket
        # here is already correctly NULL-padded (or not) based on the key
        # alone, and a residual test needs to run *before* that NULL-padding
        # decision, not after, or it ends up testing the padding itself.
        residual = join.get("condition")
        residual_matches = None
        if residual is not None and self.generate(residual) != "True":
            residual_matches = self._residual_match_fn(residual, source_context, join_context)

        for a_group, b_group in results.values():
            if residual_matches is not None:
                if left:
                    for a_row in a_group:
                        real = [a_row + b_row for b_row in b_group if residual_matches(a_row, b_row)]
                        table.rows.extend(real if real else [a_row + nulls[0]])
                elif right:
                    for b_row in b_group:
                        real = [a_row + b_row for a_row in a_group if residual_matches(a_row, b_row)]
                        table.rows.extend(real if real else [nulls[0] + b_row])
                else:
                    for a_row, b_row in itertools.product(a_group, b_group):
                        if residual_matches(a_row, b_row):
                            table.append(a_row + b_row)
                continue

            if left:
                b_group = b_group or nulls
            elif right:
                a_group = a_group or nulls

            for a_row, b_row in itertools.product(a_group, b_group):
                table.append(a_row + b_row)

        return table

    def _residual_match_fn(self, condition, source_context, join_context):
        """Build a per-`(a_row, b_row)` test for a JOIN's residual ON-clause
        condition (whatever `join_condition` left after extracting the hash
        key), evaluated the same way a real query row would see it - via a
        merged context spanning both sides, so column references on either
        alias resolve correctly. Used by `hash_join` to test LEFT/RIGHT
        candidate pairs *before* deciding whether a source row has any match
        at all, instead of filtering the (already NULL-padded) joined table
        afterward.
        """
        residual_code = self.generate(condition)
        n_source = len(source_context.columns)
        combined_columns = source_context.columns + join_context.columns

        # Reused as-is: each existing source-side Table already has `columns`
        # set to its own full (pre-join) combined row and a `column_range`
        # that's still valid unchanged - we're only appending columns after
        # it, not renumbering what's already there.
        tables = dict(source_context.tables)
        for name, jtable in join_context.tables.items():
            start = n_source + (jtable.column_range.start if jtable.column_range else 0)
            width = len(jtable.column_range) if jtable.column_range else len(jtable.columns)
            tables[name] = Table(combined_columns, [], range(start, start + width))

        residual_ctx = self.context(tables)

        def matches(a_row, b_row):
            residual_ctx.set_row(a_row + b_row)
            return bool(residual_ctx.eval(residual_code))

        return matches

    def window(self, step, context):
        rows = [reader.row for reader, _ in context]
        n = len(rows)
        names = list(step.windows)

        results = {name: [None] * n for name in names}

        for name, window in step.windows.items():
            func = window.this
            partition_by = self.generate_tuple(window.args.get("partition_by"))
            order = window.args.get("order")
            order_by = self.generate_tuple(order.expressions if order else None)

            partitions = collections.defaultdict(list)
            for i in range(n):
                context.set_index(i)
                partitions[_hashable_key(context.eval_tuple(partition_by))].append(i)

            for indices in partitions.values():
                keys = None

                if order_by:
                    evaluated = []
                    for i in indices:
                        context.set_index(i)
                        evaluated.append((i, context.eval_tuple(order_by)))
                    evaluated.sort(key=lambda pair: tuple(_NullSafeKey(v) for v in pair[1]))
                    indices = [i for i, _ in evaluated]
                    keys = [key for _, key in evaluated]

                spec = window.args.get("spec")
                for i, value in zip(
                    indices, self._window_values(func, indices, keys, context, spec)
                ):
                    results[name][i] = value

        table = Table(list(context.columns) + names)
        for i in range(n):
            table.append(rows[i] + tuple(results[name][i] for name in names))

        # Every pre-existing alias keeps pointing at the *same* `table` object
        # (not a copy) - `context.table_iter`/`set_index` only ever advance
        # whichever one Table object `step.source` names, so every alias must
        # be that identical object for its row to stay in lockstep once
        # iteration resumes (this is also how the pre-fix code behaved, for
        # any case where every alias already shared one physical Scan table).
        #
        # But two joined aliases of the same underlying table (e.g. two
        # lookups against one shared reference table) contribute identically
        # -named columns to the merged column list; naively giving every
        # alias key an unrestricted (column_range-less) reader onto that
        # shared `table` would make `RowReader` build its name->index map
        # over *all* columns, so a duplicate name resolves to whichever
        # alias's column happened to land last - e.g. `b.tgt` shadowing
        # `a.tgt` - silently returning the wrong alias's value instead of
        # raising. So every alias other than `step.name` instead gets an
        # `_AliasedRowReader`: it resolves column names through that alias's
        # own pre-window `column_range` (set up by `join()`), while always
        # reading the *current* row live off the shared table's own reader -
        # never a frozen copy - so it stays correct as iteration advances.
        orig_ranges = {name: t.column_range for name, t in context.tables.items()}
        tables = {name: table for name in context.tables}
        tables[step.name] = table
        context = self.context(tables)
        for name, column_range in orig_ranges.items():
            if name != step.name and column_range:
                context.row_readers[name] = _AliasedRowReader(
                    table.reader, table.columns, column_range
                )
        context.env["scope"] = context.row_readers

        if step.projections or step.condition:
            return self.scan(step, context)
        return context

    def _frame_bound(self, value, side, pos, width, context):
        """Resolve one end (start or end) of a ``ROWS BETWEEN`` frame to a row
        position within the partition, clamped to ``[0, width - 1]``.

        ``value``/``side`` come straight from the parsed ``WindowSpec``: `value`
        is the literal string ``"UNBOUNDED"``, the literal string ``"CURRENT
        ROW"``, an expression evaluating to a non-negative row-offset, or
        ``None`` (meaning this end of the frame was omitted, e.g. no ``AND ...``
        clause - defaults to ``CURRENT ROW``); `side` is ``"PRECEDING"``,
        ``"FOLLOWING"``, or ``None`` (only meaningful together with a numeric
        `value`).
        """
        if value is None or value == "CURRENT ROW":
            bound = pos
        elif value == "UNBOUNDED":
            bound = 0 if side == "PRECEDING" else width - 1
        else:
            offset = context.eval(self.generate(value)) if isinstance(value, exp.Expr) else value
            bound = pos - offset if side == "PRECEDING" else pos + offset
        return max(0, min(width - 1, bound))

    def _window_values(self, func, indices, keys, context, spec=None):
        """Compute the values of a single window function across one partition.

        `indices` are the row indices of the partition, already sorted by the window's
        ORDER BY (if any); `keys` are the corresponding evaluated ORDER BY tuples.

        `spec` is the window's ``WindowSpec`` (the ``ROWS BETWEEN ...`` clause),
        or ``None`` if the window has no explicit frame - in which case a
        generic aggregate below is computed over the whole partition, matching
        this executor's long-standing (frame-less) behavior. An explicit
        ``ROWS`` frame is honored per-row; ``RANGE``/``GROUPS`` frames (which
        need peer-row grouping by ORDER BY value, not just position) fall back
        to the same whole-partition behavior rather than risk a wrong slice.
        """
        ignore_nulls = isinstance(func, exp.IgnoreNulls)
        if ignore_nulls or isinstance(func, exp.RespectNulls):
            func = func.this

        width = len(indices)

        if isinstance(func, exp.RowNumber):
            return range(1, width + 1)

        if isinstance(func, (exp.Rank, exp.DenseRank)):
            dense = isinstance(func, exp.DenseRank)
            values = []
            prev_key = object()
            rank = 0
            for pos, key in enumerate(keys or [()] * width, start=1):
                if key != prev_key:
                    rank = rank + 1 if dense else pos
                    prev_key = key
                values.append(rank)
            return values

        if isinstance(func, (exp.Lag, exp.Lead)):
            sign = -1 if isinstance(func, exp.Lag) else 1

            if indices:
                context.set_index(indices[0])
            offset_expr = func.args.get("offset")
            offset = context.eval(self.generate(offset_expr)) if offset_expr else 1
            default_expr = func.args.get("default")
            default = context.eval(self.generate(default_expr)) if default_expr else None

            this = self.generate(func.this)
            partition_values = []
            for i in indices:
                context.set_index(i)
                partition_values.append(context.eval(this))

            values = []
            for pos in range(width):
                src = pos + sign * offset
                values.append(partition_values[src] if 0 <= src < width else default)
            return values

        if isinstance(func, (exp.FirstValue, exp.LastValue)):
            this = self.generate(func.this)
            candidates = indices
            if ignore_nulls:
                non_null = []
                for i in indices:
                    context.set_index(i)
                    if context.eval(this) is not None:
                        non_null.append(i)
                candidates = non_null
            if candidates:
                index = candidates[0] if isinstance(func, exp.FirstValue) else candidates[-1]
                context.set_index(index)
                value = context.eval(this)
            else:
                value = None
            return [value] * width

        agg = self.env.get(func.__class__.__name__.upper())
        if agg is None:
            raise NotImplementedError(f"Window function not supported: {func.sql()}")

        if isinstance(func.this, exp.Star):
            operands = [1] * width
        else:
            this = self.generate(func.this)
            operands = []
            for i in indices:
                context.set_index(i)
                operands.append(context.eval(this))

        if spec is not None and spec.args.get("kind") == "ROWS":
            values = []
            for pos in range(width):
                lo = self._frame_bound(spec.args.get("start"), spec.args.get("start_side"), pos, width, context)
                hi = self._frame_bound(spec.args.get("end"), spec.args.get("end_side"), pos, width, context)
                values.append(agg(operands[lo : hi + 1]) if lo <= hi else None)
            return values

        return [agg(operands)] * width

    def aggregate(self, step, context):
        group_by = self.generate_tuple(step.group.values())
        aggregations = self.generate_tuple(step.aggregations)
        operands = self.generate_tuple(step.operands)

        if operands:
            operand_table = Table(self.table(step.operands).columns)

            for reader, ctx in context:
                operand_table.append(ctx.eval_tuple(operands))

            for i, (a, b) in enumerate(zip(context.table.rows, operand_table.rows)):
                context.table.rows[i] = a + b

            width = len(context.columns)
            context.add_columns(*operand_table.columns)

            operand_table = Table(
                context.columns,
                context.table.rows,
                range(width, width + len(operand_table.columns)),
            )

            context = self.context(
                {
                    None: operand_table,
                    **context.tables,
                }
            )

        context.sort(group_by)

        group = None
        start = 0
        end = 1
        length = len(context.table)
        table = self.table(list(step.group) + step.aggregations)

        def add_row():
            table.append(group + context.eval_tuple(aggregations))

        if length:
            for i in range(length):
                context.set_index(i)
                key = context.eval_tuple(group_by)
                group = key if group is None else group
                end += 1
                if key != group:
                    context.set_range(start, end - 2)
                    add_row()
                    group = key
                    start = end - 2
                if len(table.rows) >= step.limit:
                    break
                if i == length - 1:
                    context.set_range(start, end - 1)
                    add_row()
        elif step.limit > 0 and not group_by:
            context.set_range(0, 0)
            table.append(context.eval_tuple(aggregations))

        context = self.context({step.name: table, **{name: table for name in context.tables}})

        if step.projections or step.condition:
            return self.scan(step, context)
        return context

    def sort(self, step, context):
        projections = self.generate_tuple(step.projections)
        projection_columns = [p.alias_or_name for p in step.projections]
        all_columns = list(context.columns) + projection_columns
        sink = self.table(all_columns)
        for reader, ctx in context:
            sink.append(reader.row + ctx.eval_tuple(projections))

        sort_ctx = self.context(
            {
                None: sink,
                **{table: sink for table in context.tables},
            }
        )
        sort_ctx.sort(self.generate_tuple(step.key))

        if not math.isinf(step.limit):
            sort_ctx.table.rows = sort_ctx.table.rows[0 : step.limit]

        output = Table(
            projection_columns,
            rows=[r[len(context.columns) : len(all_columns)] for r in sort_ctx.table.rows],
        )
        return self.context({step.name: output})

    def set_operation(self, step, context):
        left = context.tables[step.left]
        right = context.tables[step.right]

        sink = self.table(left.columns)

        if issubclass(step.op, exp.Intersect):
            sink.rows = list(set(left.rows).intersection(set(right.rows)))
        elif issubclass(step.op, exp.Except):
            sink.rows = list(set(left.rows).difference(set(right.rows)))
        elif issubclass(step.op, exp.Union) and step.distinct:
            sink.rows = list(set(left.rows).union(set(right.rows)))
        else:
            sink.rows = left.rows + right.rows

        if not math.isinf(step.limit):
            sink.rows = sink.rows[0 : step.limit]

        return self.context({step.name: sink})


class Python(Dialect):
    class Tokenizer(tokens.Tokenizer):
        STRING_ESCAPES = ["\\"]

    Generator = PythonGenerator
