from __future__ import annotations

import itertools
import math
import typing as t

from sqlglot import alias, exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.eliminate_joins import join_condition
from sqlglot.optimizer.scope import find_all_in_scope, find_in_scope
from collections.abc import Iterator, Sequence, Iterable


def _expand_grouping_sets(group: exp.Group) -> list[list[exp.Expr]] | None:
    """
    Expands ROLLUP / CUBE / GROUPING SETS into the explicit key lists they stand for.

    Returns `None` when the GROUP BY is a plain key list, which the executor
    aggregates in a single pass. Several such constructs in one GROUP BY
    multiply out, per standard SQL, and plain keys belong to every set.
    """
    plain: list[exp.Expr] = []
    factors: list[list[list[exp.Expr]]] = []

    for key in group.expressions:
        if isinstance(key, exp.Rollup):
            keys = key.expressions
            factors.append([keys[:i] for i in range(len(keys), -1, -1)])
        elif isinstance(key, exp.Cube):
            keys = key.expressions
            factors.append(
                [
                    list(combo)
                    for size in range(len(keys), -1, -1)
                    for combo in itertools.combinations(keys, size)
                ]
            )
        elif isinstance(key, exp.GroupingSets):
            sets = []

            for entry in key.expressions:
                if isinstance(entry, (exp.Tuple, exp.Array)):
                    sets.append(list(entry.expressions))
                elif isinstance(entry, exp.Paren):
                    sets.append([entry.this])
                else:
                    sets.append([entry])

            factors.append(sets)
        else:
            plain.append(key)

    if not factors:
        return None

    combinations = [plain]

    for factor in factors:
        combinations = [keys + extra for keys in combinations for extra in factor]

    return combinations


class Plan:
    def __init__(self, expression: exp.Expr) -> None:
        self.expression: exp.Expr = expression.copy()
        with_: exp.With | None = self.expression.args.get("with_")
        self.ctes: exp.With | None = with_.copy() if with_ is not None else None
        self.root: Step = Step.from_expression(self.expression)
        self._dag: dict[Step, set[Step]] = {}

    @property
    def dag(self) -> dict[Step, set[Step]]:
        if not self._dag:
            dag: dict[Step, set[Step]] = {}
            nodes = {self.root}

            while nodes:
                node = nodes.pop()
                dag[node] = set()

                for dep in node.dependencies:
                    dag[node].add(dep)
                    nodes.add(dep)

            self._dag = dag

        return self._dag

    @property
    def leaves(self) -> Iterator[Step]:
        return (node for node, deps in self.dag.items() if not deps)

    def __repr__(self) -> str:
        return f"Plan\n----\n{repr(self.root)}"


class Step:
    @classmethod
    def from_expression(cls, expression: exp.Expr, ctes: dict[str, Step] | None = None) -> Step:
        """
        Builds a DAG of Steps from a SQL expression so that it's easier to execute in an engine.
        Note: the expression's tables and subqueries must be aliased for this method to work. For
        example, given the following expression:

        SELECT
          x.a,
          SUM(x.b)
        FROM x AS x
        JOIN y AS y
          ON x.a = y.a
        GROUP BY x.a

        the following DAG is produced (the expression IDs might differ per execution):

        - Aggregate: x (4347984624)
            Context:
              Aggregations:
                - SUM(x.b)
              Group:
                - x.a
            Projections:
              - x.a
              - "x".""
            Dependencies:
            - Join: x (4347985296)
              Context:
                y:
                On: x.a = y.a
              Projections:
              Dependencies:
              - Scan: x (4347983136)
                Context:
                  Source: x AS x
                Projections:
              - Scan: y (4343416624)
                Context:
                  Source: y AS y
                Projections:

        Args:
            expression: the expression to build the DAG from.
            ctes: a dictionary that maps CTEs to their corresponding Step DAG by name.

        Returns:
            A Step DAG corresponding to `expression`.
        """
        ctes = ctes or {}
        expression = expression.unnest()
        with_: exp.With | None = expression.args.get("with_")

        # CTEs break the mold of scope and introduce themselves to all in the context.
        if with_ is not None:
            ctes = ctes.copy()
            for cte in with_.expressions:
                step = Step.from_expression(cte.this, ctes)
                step.name = cte.alias
                ctes[step.name] = step  # type: ignore

        from_ = expression.args.get("from_")

        if isinstance(expression, exp.Select) and from_:
            step = Scan.from_expression(from_.this, ctes)
        elif isinstance(expression, exp.SetOperation):
            step = SetOperation.from_expression(expression, ctes)
        else:
            step = Scan()

        joins: list[exp.Join] | None = expression.args.get("joins")

        if joins is not None:
            join = Join.from_joins(joins, ctes)
            join.name = step.name
            join.source_name = step.name
            join.add_dependency(step)
            step = join
        # final selects in this chain of steps representing a select
        projections: list[exp.Expr] = []
        # intermediate computations of agg funcs eg x + 1 in SUM(x + 1)
        operands: dict[exp.Expr, str] = {}
        aggregations: dict[exp.Expr, None] = {}
        next_operand_name = name_sequence("_a_")
        next_hoisted_name = name_sequence("_hw_")
        windows: dict[str, exp.Window] = {}
        next_window_name = name_sequence("_w_")

        def extract_agg_operands(expression: exp.Expr) -> bool:
            agg_funcs = tuple(find_all_in_scope(expression, exp.AggFunc))
            if agg_funcs:
                aggregations[expression] = None

            for agg in agg_funcs:
                for operand in agg.unnest_operands():
                    # DISTINCT dedupes across the whole group, so it has to stay
                    # in the aggregation; only what it wraps is a per-row operand.
                    targets = (
                        operand.expressions if isinstance(operand, exp.Distinct) else [operand]
                    )

                    for target in targets:
                        if isinstance(target, exp.Column):
                            continue
                        if target not in operands:
                            operands[target] = next_operand_name()

                        target.replace(exp.column(operands[target], quoted=True))

            return bool(agg_funcs)

        def set_ops_and_aggs(step) -> None:
            step.operands = tuple(alias(operand, alias_) for operand, alias_ in operands.items())
            step.aggregations = list(aggregations)

        for e in expression.expressions:
            windows_in_e = list(find_all_in_scope(e, exp.Window))
            if windows_in_e:
                bare = e.this if isinstance(e, exp.Alias) else e
                if bare in windows_in_e:
                    # the projection is a (possibly aliased) window with nothing
                    # else wrapping it, e.g. `SUM(b) OVER (...) AS s`
                    name = e.alias_or_name or next_window_name()
                    windows[name] = bare
                    projections.append(exp.column(name, step.name, quoted=True))
                else:
                    # the window's result is consumed by an enclosing expression
                    # in the same select, e.g. `SUM(b) OVER (...) + 1` or a CASE
                    # guarding on it; compute the window(s) as usual but keep the
                    # wrapping expression, rewriting each window reference into a
                    # column pointing at its value
                    for window in windows_in_e:
                        name = next_window_name()
                        windows[name] = window
                        window.replace(exp.column(name, step.name, quoted=True))
                    projections.append(e)
            elif find_in_scope(e, exp.AggFunc):
                projections.append(exp.column(e.alias_or_name, step.name, quoted=True))
                extract_agg_operands(e)
            else:
                projections.append(e)

        where: exp.Where | None = expression.args.get("where")

        if where is not None:
            step.condition = where.this

        group: exp.Group | None = expression.args.get("group")

        # A window function alongside a GROUP BY operates on the query's
        # *grouped* result set, per standard SQL - never on pre-aggregation rows
        # - so Aggregate must run before Window here, the reverse of the
        # windows-but-no-GROUP-BY case below. `intermediate`'s rewrite of a GROUP
        # BY key reference into its aggregate-group alias is applied to each
        # window body too, so e.g. `ROW_NUMBER() OVER (ORDER BY <group-by key>)`
        # resolves against the aggregated output rather than the raw rows.
        defer_windows = bool(windows) and (group is not None or aggregations)

        if windows and not defer_windows:
            window_step = Window()
            window_step.source = step.name
            window_step.name = step.name
            window_step.windows = windows
            window_step.add_dependency(step)
            step = window_step

        if group is not None or aggregations:
            aggregate = Aggregate()
            aggregate.source = step.name
            aggregate.name = step.name

            having: exp.Having | None = expression.args.get("having")

            if having is not None:
                if extract_agg_operands(exp.alias_(having.this, "_h", quoted=True)):
                    aggregate.condition = exp.column("_h", step.name, quoted=True)
                else:
                    aggregate.condition = having.this

            if defer_windows:
                # A deferred window runs on the *aggregated* rows, so an
                # aggregate inside its frame has to be computed as one of this
                # step's aggregations and read back as a column. GROUPING()
                # included: it varies per row, each row coming from one grouping
                # set, so it can't be folded in the window itself. `window.this`
                # is excluded - that's the window function, which is computed
                # over these rows rather than alongside them.
                hoisted: dict[str, str] = {}

                for window in windows.values():
                    frame = list(window.args.get("partition_by") or [])
                    window_order = window.args.get("order")

                    if window_order:
                        frame.append(window_order)

                    for part in frame:
                        for node in list(part.find_all(exp.AggFunc)):
                            key = node.sql()
                            name = hoisted.get(key)

                            if name is None:
                                name = next_hoisted_name()
                                hoisted[key] = name
                                extract_agg_operands(exp.alias_(node.copy(), name, quoted=True))

                            node.replace(exp.column(name, step.name, quoted=True))

            set_ops_and_aggs(aggregate)

            # give aggregates names and replace projections with references to them
            grouping_sets = _expand_grouping_sets(group) if group else None

            if grouping_sets is None:
                keys = list(group.expressions) if group else []
            else:
                # Every key any set mentions has to be computed; each set then
                # aggregates by the subset it actually names.
                keys = []
                for grouping_set in grouping_sets:
                    for key in grouping_set:
                        if not any(key == existing for existing in keys):
                            keys.append(key)

            aggregate.group = {f"_g{i}": e for i, e in enumerate(keys)}

            if grouping_sets is not None:
                names = {e: name for name, e in aggregate.group.items()}
                aggregate.grouping_sets = [
                    [names[key] for key in grouping_set] for grouping_set in grouping_sets
                ]

            intermediate: dict[str | exp.Expr, str] = {}
            for k, v in aggregate.group.items():
                intermediate[v] = k
                if isinstance(v, exp.Column):
                    intermediate[v.name] = k

            for projection in projections:
                for node in projection.walk():
                    name = intermediate.get(node)
                    if name:
                        node.replace(exp.column(name, step.name))

            if defer_windows:
                for window in windows.values():
                    for node in window.walk():
                        name = intermediate.get(node) or intermediate.get(node.name)
                        if name:
                            node.replace(exp.column(name, step.name))

            if aggregate.condition:
                for node in aggregate.condition.walk():
                    name = intermediate.get(node) or intermediate.get(node.name)
                    if name:
                        node.replace(exp.column(name, step.name))

            aggregate.add_dependency(step)
            step = aggregate

            if defer_windows:
                window_step = Window()
                window_step.source = step.name
                window_step.name = step.name
                window_step.windows = windows
                window_step.add_dependency(step)
                step = window_step
        else:
            aggregate = None

        order: exp.Order | None = expression.args.get("order")

        if order is not None:
            if aggregate is not None and isinstance(step, Aggregate):
                for i, ordered in enumerate(order.expressions):
                    if extract_agg_operands(exp.alias_(ordered.this, f"_o_{i}", quoted=True)):
                        ordered.this.replace(exp.column(f"_o_{i}", step.name, quoted=True))

                set_ops_and_aggs(aggregate)

            sort = Sort()
            sort.name = step.name
            sort.key = order.expressions
            sort.add_dependency(step)
            step = sort

        step.projections = projections

        if isinstance(expression, exp.Select) and expression.args.get("distinct"):
            distinct = Aggregate()
            distinct.source = step.name
            distinct.name = step.name
            distinct.group = {
                e.alias_or_name: exp.column(col=e.alias_or_name, table=step.name)
                for e in projections or expression.expressions
            }
            distinct.add_dependency(step)
            step = distinct

        limit: exp.Limit | None = expression.args.get("limit")

        if limit is not None:
            step.limit = int(limit.text("expression"))

        offset: exp.Offset | None = expression.args.get("offset")

        if offset is not None:
            step.offset = int(offset.text("expression"))

        return step

    def __init__(self) -> None:
        self.name: str | None = None
        self.dependencies: set[Step] = set()
        self.dependents: set[Step] = set()
        self.projections: Sequence[exp.Expr] = []
        self.limit: float = math.inf
        self.offset: int = 0
        self.condition: exp.Expr | None = None

    def add_dependency(self, dependency: Step) -> None:
        self.dependencies.add(dependency)
        dependency.dependents.add(self)

    def __repr__(self) -> str:
        return self.to_s()

    def to_s(self, level: int = 0) -> str:
        indent = "  " * level
        nested = f"{indent}    "

        context = self._to_s(f"{nested}  ")

        if context:
            context = [f"{nested}Context:"] + context

        lines = [
            f"{indent}- {self.id}",
            *context,
            f"{nested}Projections:",
        ]

        for expression in self.projections:
            lines.append(f"{nested}  - {expression.sql()}")

        if self.condition:
            lines.append(f"{nested}Condition: {self.condition.sql()}")

        if self.limit is not math.inf:
            lines.append(f"{nested}Limit: {self.limit}")

        if self.offset:
            lines.append(f"{nested}Offset: {self.offset}")

        if self.dependencies:
            lines.append(f"{nested}Dependencies:")
            for dependency in self.dependencies:
                lines.append("  " + dependency.to_s(level + 1))

        return "\n".join(lines)

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    @property
    def id(self) -> str:
        name = self.name
        name = f" {name}" if name else ""
        return f"{self.type_name}:{name} ({id(self)})"

    def _to_s(self, _indent: str) -> list[str]:
        return []


class Scan(Step):
    @classmethod
    def from_expression(cls, expression: exp.Expr, ctes: dict[str, Step] | None = None) -> Step:
        table: exp.Expr = expression
        alias_ = expression.alias_or_name

        if isinstance(expression, exp.Subquery):
            table = expression.this
            step = Step.from_expression(table, ctes)
            step.name = alias_
            return step

        step = Scan()
        step.name = alias_
        step.source = expression
        if ctes and table.name in ctes:
            step.add_dependency(ctes[table.name])

        return step

    def __init__(self) -> None:
        super().__init__()
        self.source: exp.Expr | None = None

    def _to_s(self, indent: str) -> list[str]:
        return [f"{indent}Source: {self.source.sql() if self.source else '-static-'}"]  # type: ignore


class Join(Step):
    @classmethod
    def from_joins(cls, joins: Iterable[exp.Join], ctes: dict[str, Step] | None = None) -> Join:
        step = Join()

        for join in joins:
            source_key, join_key, condition = join_condition(join)
            step.joins[join.alias_or_name] = {
                "side": join.side,  # type: ignore
                "join_key": join_key,
                "source_key": source_key,
                "condition": condition,
            }

            step.add_dependency(Scan.from_expression(join.this, ctes))

        return step

    def __init__(self) -> None:
        super().__init__()
        self.source_name: str | None = None
        self.joins: dict[str, dict[str, list[str] | exp.Expr | list[exp.Expr]]] = {}

    def _to_s(self, indent: str) -> list[str]:
        lines = [f"{indent}Source: {self.source_name or self.name}"]
        for name, join in self.joins.items():
            lines.append(f"{indent}{name}: {join['side'] or 'INNER'}")
            join_key = ", ".join(str(key) for key in t.cast(list[str], join.get("join_key") or []))
            if join_key:
                lines.append(f"{indent}Key: {join_key}")
            if join.get("condition"):
                lines.append(f"{indent}On: {join['condition'].sql()}")  # type: ignore
        return lines


class Window(Step):
    def __init__(self) -> None:
        super().__init__()
        self.windows: dict[str, exp.Window] = {}
        self.source: str | None = None

    def _to_s(self, indent: str) -> list[str]:
        lines = [f"{indent}Windows:"]

        for name, window in self.windows.items():
            lines.append(f"{indent}  - {name}: {window.sql()}")

        return lines


class Aggregate(Step):
    def __init__(self) -> None:
        super().__init__()
        self.aggregations: list[exp.Expr] = []
        self.operands: tuple[exp.Expr, ...] = ()
        self.group: dict[str, exp.Expr] = {}
        # One entry per GROUP BY grouping set, each listing the `group` keys
        # that set aggregates by; keys a set leaves out come back as NULL.
        # Empty when the query groups by a plain key list.
        self.grouping_sets: list[list[str]] = []
        self.source: str | None = None

    def _to_s(self, indent: str) -> list[str]:
        lines = [f"{indent}Aggregations:"]

        for expression in self.aggregations:
            lines.append(f"{indent}  - {expression.sql()}")

        if self.group:
            lines.append(f"{indent}Group:")
            for expression in self.group.values():
                lines.append(f"{indent}  - {expression.sql()}")
        if self.grouping_sets:
            lines.append(f"{indent}Grouping Sets:")
            for keys in self.grouping_sets:
                rendered = ", ".join(self.group[key].sql() for key in keys)
                lines.append(f"{indent}  - ({rendered})")
        if self.condition:
            lines.append(f"{indent}Having:")
            lines.append(f"{indent}  - {self.condition.sql()}")
        if self.operands:
            lines.append(f"{indent}Operands:")
            for expression in self.operands:
                lines.append(f"{indent}  - {expression.sql()}")

        return lines


class Sort(Step):
    def __init__(self) -> None:
        super().__init__()
        self.key: list[exp.Expr] | None = None

    def _to_s(self, indent: str) -> list[str]:
        lines = [f"{indent}Key:"]

        for expression in self.key:  # type: ignore
            lines.append(f"{indent}  - {expression.sql()}")

        return lines


class SetOperation(Step):
    def __init__(self, op: type[exp.Expr], left: str, right: str, distinct: bool = False) -> None:
        super().__init__()
        self.op: type[exp.Expr] = op
        self.left: str = left
        self.right: str = right
        self.distinct: bool = distinct

    @classmethod
    def from_expression(
        cls, expression: exp.Expr, ctes: dict[str, Step] | None = None
    ) -> SetOperation:
        assert isinstance(expression, exp.SetOperation)

        left = Step.from_expression(expression.left, ctes)
        # SELECT 1 UNION SELECT 2  <-- these subqueries don't have names
        left.name = left.name or "left"
        right = Step.from_expression(expression.right, ctes)
        right.name = right.name or "right"

        if left.name == right.name:
            # Both branches reading the same table (`SELECT .. FROM t UNION ALL
            # SELECT .. FROM t`) would otherwise publish their outputs under one
            # name. The executor resolves a branch by name, so it read the same
            # side twice and returned that branch's rows for both.
            left.name = f"{left.name}_1"
            right.name = f"{right.name}_2"

        step = cls(
            op=expression.__class__,
            left=left.name,
            right=right.name,
            distinct=bool(expression.args.get("distinct")),
        )

        step.add_dependency(left)
        step.add_dependency(right)

        return step

    def _to_s(self, indent: str) -> list[str]:
        lines: list[str] = []
        if self.distinct:
            lines.append(f"{indent}Distinct: {self.distinct}")
        return lines

    @property
    def type_name(self) -> str:
        return self.op.__name__
