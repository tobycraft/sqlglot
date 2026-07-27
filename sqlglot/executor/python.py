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


class PythonExecutor:
    def __init__(self, env=None, tables=None):
        self.generator = Python().generator(identify=True, comments=False)
        self.env = {**ENV, **(env or {})}
        self.tables = tables or {}

    def execute(self, plan):
        finished = set()
        queue = set(plan.leaves)
        contexts = {}

        while queue:
            node = queue.pop()
            try:
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
                else:
                    raise NotImplementedError

                finished.add(node)

                for dep in node.dependents:
                    if all(d in contexts for d in dep.dependencies):
                        queue.add(dep)

                for dep in node.dependencies:
                    if all(d in finished for d in dep.dependents):
                        contexts.pop(dep)
            except Exception as e:
                raise ExecuteError(f"Step '{node.id}' failed: {e}") from e

        root = plan.root
        return contexts[root].tables[root.name]

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
            table = context.tables[name]
            start = max(r.stop for r in column_ranges.values())
            column_ranges[name] = range(start, len(table.columns) + start)
            join_context = self.context({name: table})

            if join.get("source_key"):
                table = self.hash_join(join, source_context, join_context)
            else:
                table = self.nested_loop_join(join, source_context, join_context)

            source_context = self.context(
                {
                    name: Table(table.columns, table.rows, column_range)
                    for name, column_range in column_ranges.items()
                }
            )
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

    def nested_loop_join(self, _join, source_context, join_context):
        table = Table(source_context.columns + join_context.columns)

        for reader_a, _ in source_context:
            for reader_b, _ in join_context:
                table.append(reader_a.row + reader_b.row)

        return table

    def hash_join(self, join, source_context, join_context):
        source_key = self.generate_tuple(join["source_key"])
        join_key = self.generate_tuple(join["join_key"])
        left = join.get("side") == "LEFT"
        right = join.get("side") == "RIGHT"

        results = collections.defaultdict(lambda: ([], []))

        for reader, ctx in source_context:
            results[ctx.eval_tuple(source_key)][0].append(reader.row)
        for reader, ctx in join_context:
            results[ctx.eval_tuple(join_key)][1].append(reader.row)

        table = Table(source_context.columns + join_context.columns)
        nulls = [(None,) * len(join_context.columns if left else source_context.columns)]

        for a_group, b_group in results.values():
            if left:
                b_group = b_group or nulls
            elif right:
                a_group = a_group or nulls

            for a_row, b_row in itertools.product(a_group, b_group):
                table.append(a_row + b_row)

        return table

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
                partitions[context.eval_tuple(partition_by)].append(i)

            for indices in partitions.values():
                keys = None

                if order_by:
                    evaluated = []
                    for i in indices:
                        context.set_index(i)
                        evaluated.append((i, context.eval_tuple(order_by)))
                    evaluated.sort(key=lambda pair: tuple((v is None, v) for v in pair[1]))
                    indices = [i for i, _ in evaluated]
                    keys = [key for _, key in evaluated]

                for i, value in zip(indices, self._window_values(func, indices, keys, context)):
                    results[name][i] = value

        table = Table(list(context.columns) + names)
        for i in range(n):
            table.append(rows[i] + tuple(results[name][i] for name in names))

        context = self.context({step.name: table, **{name: table for name in context.tables}})

        if step.projections or step.condition:
            return self.scan(step, context)
        return context

    def _window_values(self, func, indices, keys, context):
        """Compute the values of a single window function across one partition.

        `indices` are the row indices of the partition, already sorted by the window's
        ORDER BY (if any); `keys` are the corresponding evaluated ORDER BY tuples.
        """
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
            if indices:
                index = indices[0] if isinstance(func, exp.FirstValue) else indices[-1]
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
