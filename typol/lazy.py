from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    Literal,
    LiteralString,
    Self,
    TypeVar,
    Unpack,
    assert_never,
    cast,
    get_args,
    overload,
)

import more_itertools
import polars as pl
import polars.lazyframe.group_by
from more_itertools import first, prepend
from packaging.version import Version

from typol.expr import (
    AggExpr,
    Alias,
    AliasShape,
    BoundDimension,
    ColumnInitializer,
    EndoAggExpr,
    EndoExpr,
    ExoExpr,
    Explosion,
    Expr,
    JoinOn,
    Shape,
    Suffixed,
    suffix,
)
from typol.frame import (
    DataFrame,
    JoinAgainstType,
    JoinOptions,
    JoinTogetherType,
    JoinType,
    enforce_shape,
)
from typol.series import LazySeries
from typol.types import list_of

if TYPE_CHECKING:
    from ty_extensions import Intersection


_S_co = TypeVar("_S_co", bound=Shape, covariant=True)


@dataclasses.dataclass(init=False, eq=False, frozen=True)
class LazyFrame(Generic[_S_co]):
    """Shape-bound dataframe whose operations are type checked"""

    shape: type[_S_co]
    dataframe: pl.LazyFrame

    @overload
    def __init__(
        self,
        of: type[_S_co],
        /,
        values: Iterable[Mapping[str, Any]]
        | Iterable[ColumnInitializer[_S_co, Any]]
        | Mapping[BoundDimension[_S_co, Any], Iterable]
        | Iterable[tuple]
        | pl.LazyFrame
        | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self, of: type[_S_co], /, values: Iterable[tuple], *, orient: Literal["row", "col"] = ...
    ) -> None: ...

    def __init__(
        self,
        shape: type[_S_co],
        /,
        values: Iterable[Mapping[str, Any]]
        | Iterable[tuple]
        | Iterable[ColumnInitializer[_S_co, Any]]
        | Mapping[BoundDimension[_S_co, Any], Iterable]
        | pl.LazyFrame
        | None = None,
        *,
        orient: Literal["row", "col"] | None = None,
    ) -> None:
        meta = shape.shape_meta()
        if isinstance(values, pl.LazyFrame):
            df = enforce_shape(shape, values)
        elif isinstance(values, tuple) and isinstance(values[0], ColumnInitializer):
            initializers = cast(tuple[ColumnInitializer[_S_co, Any], ...], values)
            df = pl.LazyFrame({i.dimension.name: i.value for i in initializers}, schema=meta.schema)
        elif isinstance(values, Mapping):
            df = pl.LazyFrame(
                {k.name if isinstance(k, BoundDimension) else k: vs for k, vs in values.items()},
                schema=meta.schema,
            )
        elif isinstance(values, Iterable):
            iterator = iter(values)
            first = more_itertools.first(iterator, None)
            if first is None:
                df = pl.LazyFrame(schema=meta.schema)
            elif isinstance(first, ColumnInitializer):
                initializers = cast(Iterable[ColumnInitializer[_S_co, Any]], values)
                df = pl.LazyFrame(
                    {i.dimension.name: i.value for i in initializers}, schema=meta.schema
                )
            elif isinstance(first, Mapping) and type(first) is not dict:
                values = cast(Iterable[Mapping], more_itertools.prepend(first, iterator))
                df = pl.LazyFrame((dict(r) for r in values), schema=meta.schema)
            else:
                df = pl.LazyFrame(
                    values
                    if isinstance(values, Collection)
                    else more_itertools.prepend(first, iterator),
                    schema=meta.schema,
                    orient=orient,
                )
        elif values is None:
            df = pl.LazyFrame(schema=meta.schema)
        else:
            assert_never(values)
        object.__setattr__(self, "dataframe", df)
        object.__setattr__(self, "shape", shape)

    @property
    def s(self) -> _S_co:
        """
        Provides a utility alias for accessing frame shape columns and attributes

        ```
        purchases.filter(purchases.s.price > 10) == purchases.filter(Purchase.price > 10)
        ```

        This is particularly useful when the shape has been constructed implicitly rather than is
        explicitly defined:

        ```
        info = purchases.join(customers, Purchase.customer.on(Customer.name))
        info.filter(info.s.age > 20)  # `.s` here refers to `Purchase & Customer`

        suffixed = customer.suffix()
        suffixed.filter(suffixed.s(Customer.name) == "Samwise")  # `.s` is `Suffixed[Customer]`
        ```
        """
        # As `Shape`s only have class-level operations, pretending this is an instance is equivalent
        # enough for direct usage of attributes (dimensions and `shape_meta`). To see the shape as a
        # shape-type, use `.shape`. This works around ty's limitation of resolving `Unknown` for
        # `type[S & Q]` (ty's fine with `type[S] & type[Q]`), by lowering to the instance level
        return cast(_S_co, self.shape)

    def __getitem__[T](self, s: ExoExpr[_S_co, T]) -> LazySeries[T]:
        """Construct a lazy series of values transformed by `s` for each row in the frame"""
        return LazySeries[T](self.dataframe.select(s.expr.alias("series")))

    def get_column[T](self, s: BoundDimension[_S_co, T]) -> LazySeries[T]:
        return self.__getitem__(s)

    def head(self, n: int = 5) -> LazyFrame[_S_co]:
        """Take only the first `n` rows"""
        return LazyFrame(self.shape, self.dataframe.head(n))

    def tail(self, n: int = 5) -> LazyFrame[_S_co]:
        """Take only the last `n` rows"""
        return LazyFrame(self.shape, self.dataframe.tail(n))

    def bottom_k(
        self,
        k: int,
        *,
        by: ExoExpr[_S_co, Any] | Collection[ExoExpr[_S_co, Any]],
        reverse: bool = False,
    ) -> LazyFrame[_S_co]:
        """Take only the smallest `k` rows, using `by` as the key"""
        exprs = by.expr if isinstance(by, Expr) else [e.expr for e in by]
        return LazyFrame(self.shape, self.dataframe.bottom_k(k, by=exprs, reverse=reverse))

    def top_k(
        self,
        k: int,
        *,
        by: ExoExpr[_S_co, Any] | Collection[ExoExpr[_S_co, Any]],
        reverse: bool = False,
    ) -> LazyFrame[_S_co]:
        """Take only the greatest `k` rows, using `by` as the key"""
        exprs = by.expr if isinstance(by, Expr) else [e.expr for e in by]
        return LazyFrame(self.shape, self.dataframe.top_k(k, by=exprs, reverse=reverse))

    def slice(self, offset: int, length: int | None = None) -> LazyFrame[_S_co]:
        return LazyFrame(self.shape, self.dataframe.slice(offset, length))

    def lazy(self) -> LazyFrame[_S_co]:
        return self

    def collect(
        self, engine: Literal["auto", "in-memory", "streaming", "gpu"] = "auto"
    ) -> DataFrame[_S_co]:
        pre_auto = Version(pl.__version__) < Version("1.40.0")
        if pre_auto and engine == "auto":
            # Older versions don't support auto
            engine = cast(Literal["auto"], "cpu")
        df = self.dataframe.collect(engine=engine, background=False)
        return DataFrame(self.shape, df)

    def filter(self, *condition: ExoExpr[_S_co, bool]) -> LazyFrame[_S_co]:
        """Only keep rows where the boolean conditions evaluate to `True`"""
        return LazyFrame(self.shape, self.dataframe.filter(*(c.expr for c in condition)))

    @overload
    def with_columns(self, *columns: EndoExpr[_S_co, Any]) -> LazyFrame[_S_co]: ...
    @overload
    def with_columns[A: LiteralString, AT](
        self, alias: Alias[_S_co, A, AT], /, *columns: EndoExpr[_S_co, Any] | Alias[_S_co, A, AT]
    ) -> LazyFrame[Intersection[_S_co, AliasShape[A, AT]]]: ...

    def with_columns[A: LiteralString, AT](
        self, *columns: EndoExpr[_S_co, Any] | Alias[_S_co, A, AT]
    ) -> LazyFrame[Intersection[_S_co, AliasShape[A, AT]]]:
        """
        Use the provided expressions to update existing columns in the shape:

        ```py
        customers.with_columns(
            customers.s.age + 1,  # Add one to their age
            customers.s.name.fill_null(customers.s.phone)  # Use their phone number as a backup name
        )
        ```

        If adding or dropping columns, use [`transform`][typol.lazy.LazyFrame.transform] instead
        """
        df = self.dataframe.with_columns(c.expr for c in columns)
        shape = self.shape
        for column in columns:
            if isinstance(column, Alias):
                shape &= column.construct_shape(df)
        return LazyFrame(shape, df)

    def transform[SNew: Shape](
        self, shape: type[SNew], *transforms: Expr[_S_co, SNew, Any]
    ) -> LazyFrame[SNew]:
        """
        Convert from one shape to another shape, using the provided expressions to map columns in
        the current shape to columns in the new shape:

        ```py
        # Transform also acts like a select, picking all `Person` columns from a `Customer`
        customers.transform(Person)
        # You can map any Customer column to any Person column in the meantime
        customers.transform(Person, customers.s.name.str.strip_chars(), customers.s.age + 1)
        # Don't use transform when staying within the same shape, just use with_columns
        customers.with_columns(customers.s.name.str.strip_chars(), customers.s.age + 1)
        ```

        - Any columns with the same name in both the current and new shapes without an expression
          mapping to them will be preserved
        - Any columns in the new shape not in the original shape, and not mapped to, will throw a
          runtime error
        - Any columns in the current shape not in the new shape will be dropped

        See [with_columns][typol.lazy.LazyFrame.with_columns] when not changing between shapes
        """
        return LazyFrame(shape, self.dataframe.with_columns(t.expr for t in transforms))

    def explode(self, *explosions: Explosion[_S_co, _S_co, Any]) -> LazyFrame[_S_co]:
        """
        Take a series of list columns and create a new row for each value in the list:

        ```
        accounts.explode(
            Account.link_name.implode().over(Account.type).list.explode_to(Account.link_name)
        )
        ```

        The above will create a new row for every linked name from any account for the same type
        """
        return LazyFrame(
            self.shape,
            self.dataframe.with_columns(
                e.expr.to_out(e.to.name).cast(list_of(e.to.ty)).expr for e in explosions
            ).explode(*(e.to.name for e in explosions)),
        )

    def explode_transform[Q: Shape](
        self, shape: type[Q], *explosions: Explosion[_S_co, Q, Any] | Expr[_S_co, Q, Any]
    ) -> LazyFrame[Q]:
        """Take a series of list columns and create a new row for each value in the list"""
        return LazyFrame(
            shape,
            self.dataframe.with_columns(
                (
                    e.expr.to_out(e.to.name).cast(list_of(e.to.ty))
                    if isinstance(e, Explosion)
                    else e
                ).expr
                for e in explosions
            ).explode(*(e.to.name for e in explosions if isinstance(e, Explosion))),
        )

    def agg(self, *agg: EndoAggExpr[_S_co, Any]) -> LazyFrame[_S_co]:
        """
        Define the aggregating expressions to group rows in the dataframe. Any columns not
        aggregated will be treated as the group by keys, since all columns must be preserved. To
        drop columns instead use `transform` before `agg`
        """
        aggregating = {e.expr.meta.output_name() for e in agg}
        return LazyFrame(
            self.shape,
            self.dataframe.group_by(
                *map(pl.col, self.dataframe.collect_schema().keys() - aggregating)
            ).agg(*(e.expr for e in agg)),
        )

    def group_by(self, *keys: EndoExpr[_S_co, Any]) -> LazyGroupBy[_S_co, _S_co]:
        """
        Determine a series of expressions to group the dataframe by, this should be followed by an
        agg to apply aggregations to the grouped frame
        """
        return LazyGroupBy(self.shape, self.dataframe.group_by(*(k.expr for k in keys)))

    def agg_transform[Q: Shape](
        self, shape: type[Q], *exprs: AggExpr[_S_co, Q, Any] | Expr[_S_co, Q, Any]
    ) -> LazyFrame[Q]:
        """
        Define the aggregating expressions to group rows in the dataframe. Any columns not
        aggregated will be treated as the group by keys, since all columns must be preserved. To
        drop columns instead use `transform` before `agg`.

        This allows transforming the aggregated columns since aggregation may change types
        """
        aggregating = {e.expr.meta.output_name(): e.expr for e in exprs if isinstance(e, AggExpr)}
        non_aggregating = {e.expr.meta.output_name(): e.expr for e in exprs if isinstance(e, Expr)}
        assert aggregating.keys().isdisjoint(non_aggregating.keys()), (
            f"Can't aggregate and group by same dimensions: {aggregating.keys() & non_aggregating.keys()}"
        )
        return LazyFrame(
            shape,
            self.dataframe.group_by(
                *(
                    e if (e := non_aggregating.get(k)) is not None else pl.col(k)
                    for k in shape.shape_meta().datatypes.keys() - aggregating.keys()
                )
            ).agg(*aggregating.values()),
        )

    def group_by_transform[Q: Shape](
        self, shape: type[Q], *keys: Expr[_S_co, Q, Any]
    ) -> LazyGroupBy[_S_co, Q]:
        """
        Determine a series of expressions to group the dataframe by, this should be followed by an
        agg to apply aggregations to the grouped frame
        """
        return LazyGroupBy(shape, self.dataframe.group_by(*(k.expr for k in keys)))

    @staticmethod
    def concat[S: Shape](shape: type[S], frames: Iterable[LazyFrame[S]]) -> LazyFrame[S]:
        frames = iter(frames)
        if head := first(frames, None):
            return head.vstack(*frames)
        return LazyFrame(shape)

    def vstack(self, *frames: Self) -> LazyFrame[_S_co]:
        # We know these share the same shape, but we don't know the order of the columns matches.
        # We select the columns to reorder them to match in the vstack
        columns = self.dataframe.collect_schema().names()
        return LazyFrame(
            self.shape,
            pl.concat(
                prepend(self.dataframe, (d.dataframe.select(columns) for d in frames)),
                how="vertical",
            ),
        )

    def unique(
        self,
        *exprs: BoundDimension[_S_co, Any],
        keep: Literal["first", "last", "none", "any"] = "any",
        maintain_order: bool = False,
    ) -> LazyFrame[_S_co]:
        names = [e.name for e in exprs] if exprs else None
        return LazyFrame(
            self.shape, self.dataframe.unique(names, keep=keep, maintain_order=maintain_order)
        )

    def sort(
        self,
        *exprs: ExoExpr[_S_co, Any],
        descending: tuple[bool, ...] | bool = False,
        nulls_last: tuple[bool, ...] | bool = False,
        maintain_order: bool = False,
    ) -> LazyFrame[_S_co]:
        return LazyFrame(
            self.shape,
            self.dataframe.sort(
                (e.expr for e in (exprs or self.shape.shape_meta().dimensions)),
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
            ),
        )

    def suffix(self, suffixed: type[Suffixed[_S_co]] | None = None) -> LazyFrame[Suffixed[_S_co]]:
        """
        Suffix the columns of the shape to distinguish them from conflicts with other shape column
        names, retyping the dataframe as `Suffixed[CurrentShape]`:

        ```
        suffixed = customer.suffix()  # type: tp.LazyFrame[Suffixed[Customer]]
        ```

        Suffixed shapes cannot have their fields accessed directly, instead must be accessed through
        projecting the dimensions via the suffixed shape:

        ```
        # Note, suffixed.s(...) converts an original `Customer` dimension into a suffixed dimension
        suffixed[suffixed.s(Customer.name)].collect().to_list()
        ```

        This is most useful in joint/intersection shapes, or self-joins:

        ```
        # Add a suffix to all the columns so they can be referred to independently
        other_customers = customers.suffix()
        # Join customers against itself to find ones where the names conflict
        customers_with_the_same_name = customers.join(
            other_customers,
            # To refer to suffixed columns, do suffixed_shape(original_column)
            customers.s.name.on(other_customers.s(Customer.name))
            how="cross"
        ).filter(customer.s.phone != other_customers.s(Customer.phone))
        ```

        [see `expr.suffix`][typol.expr.suffix] for more info on shape suffixing
        """
        suffixed = suffixed or suffix(self.shape)
        return LazyFrame[Any](suffixed, self.dataframe.rename(suffixed.mapping_to()))

    def pipe[**P, T](
        self, function: Callable[Concatenate[Self, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return function(self, *args, **kwargs)

    @staticmethod
    def scan_csv[S: Shape](
        shape: type[S],
        source: IO[str] | str | bytes | Path | IO[bytes],
        mappings: Mapping[BoundDimension[S, Any], str] | None = None,
        *,
        has_header: bool = True,
        skip_rows: int = 0,
        low_memory: bool = False,
    ) -> LazyFrame[S]:
        if not has_header:
            return LazyFrame(
                shape,
                pl.scan_csv(
                    source,
                    schema=shape.shape_meta().schema,
                    has_header=False,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                    try_parse_dates=True,
                    skip_rows=skip_rows,
                    infer_schema=False,
                ),
            )
        if mappings is None:
            # The default is to match up the columns from the file with the dimensions from the
            # shape alphanumerically case-insenstively
            headers_only = pl.scan_csv(source, infer_schema_length=0, skip_rows=skip_rows, n_rows=0)
            schema = headers_only.collect_schema()
            columns = {re.sub(r"[\W_]", "", h).lower(): h for h in schema.keys()}
            mappings = {
                d: columns[re.sub(r"[\W_]", "", d.name).lower()]
                for d in shape.shape_meta().dimensions
            }

        return LazyFrame(
            shape,
            pl.scan_csv(
                source,
                schema_overrides={mappings[d]: d.ty.pl_ty for d in shape.shape_meta().dimensions},
                has_header=has_header,
                ignore_errors=True,
                truncate_ragged_lines=True,
                try_parse_dates=True,
                skip_rows=skip_rows,
                infer_schema=False,
                infer_schema_length=0,
                low_memory=low_memory,
            )
            .select(mappings[d] for d in shape.shape_meta().dimensions)
            .rename({mappings[d]: d.name for d in shape.shape_meta().dimensions}),
        )

    def join_asof[Q: Shape](
        self,
        right: LazyFrame[Q],
        on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        strategy: Literal["backward", "forward", "nearest"] = "backward",
    ) -> LazyFrame[Intersection[_S_co, Q]]:
        """
        Join two tables into a common shape, by nearest

        Parameters
        ----------
        on : BoundDimension[S, _]
            Join on the same columns for the left and the right shapes based on the joint shape.
            The column must be available in both original shapes
        """
        joined = self.dataframe.join_asof(
            right.dataframe,
            left_on=(on.left if isinstance(on, JoinOn) else on).expr,
            right_on=(on.right if isinstance(on, JoinOn) else on).expr,
            strategy=strategy,
        )
        return LazyFrame["Intersection[_S_co, Q]"](self.shape & right.shape, joined)

    @overload
    def join[Q: Shape](
        self,
        right: LazyFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinTogetherType = "inner",
        **options: Unpack[JoinOptions],
    ) -> LazyFrame[Intersection[_S_co, Q]]: ...
    @overload
    def join[Q: Shape](
        self,
        right: LazyFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinAgainstType,
        **options: Unpack[JoinOptions],
    ) -> LazyFrame[_S_co]: ...
    @overload
    def join[Q: Shape](
        self,
        right: LazyFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinType = "inner",
        **options: Unpack[JoinOptions],
    ) -> LazyFrame[Intersection[_S_co, Q]] | LazyFrame[_S_co]: ...

    def join[Q: Shape](
        self,
        right: LazyFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinType = "inner",
        **options: Unpack[JoinOptions],
    ) -> LazyFrame[Intersection[_S_co, Q]] | LazyFrame[_S_co]:
        """
        Join two tables into a common shape, the intersection of the two provided shapes:

        ```py
        customers.join(
            purchases,
            customers.s.name.on(purchases.s.customer),
            customers.s.phone.on(purchases.s.billing_phone),
        )  # resultant frame of type Customer & Purchase
        ```

        If there are conflicting columns, Polars will not be able to distinguish the results.
        Explicitly distinguish two shapes with `df.suffix()`, particularly important for self-joins:

        ```py
        # Add a suffix to all the columns so they can be referred to independently
        other_customers = customers.suffix()
        # Join customers against itself to find ones where the names conflict
        customers_with_the_same_name = customers.join(
            other_customers,
            # To refer to suffixed columns, do suffixed_shape(original_column)
            customers.s.name.on(other_customers.s(Customer.name))
            how="cross"
        ).filter(customer.s.phone != other_customers.s(Customer.phone))
        ```

        [see `LazyFrame.suffix`][typol.lazy.LazyFrame.suffix]

        Parameters
        ----------
        on : BoundDimension[S, _]
            Join on the same columns for the left and the right shapes based on the joint shape.
            The column must be available in both original shapes
        how : Literal["inner", "left", "right", "full", "semi", "anti", "cross", "outer"]
            Type of join to apply. "anti" and "semi" joins are different in that they return the
            left shape only
        **options : JoinOptions
            Other Polars-support join options, may vary by Polars version
        """
        if Version(pl.__version__) < Version("1.24") and (jn := options.pop("nulls_equal", None)):
            # Pre-1.24, Polars calls this column "join_nulls"
            options["join_nulls"] = jn  # ty: ignore[invalid-key]
        if on:
            joined = self.dataframe.join(
                right.dataframe,
                left_on=[(e.left if isinstance(e, JoinOn) else e).expr for e in on],
                right_on=[(e.right if isinstance(e, JoinOn) else e).expr for e in on],
                how=how,
                **options,
            )
            if how in get_args(JoinAgainstType):
                # "anti" and "semi" joins just return the left shape, so do not need to be filled in
                # nor any intersection constructed
                return LazyFrame[_S_co](self.shape, joined)

            already_populated = frozenset(joined.collect_schema().keys())
            joined = joined.with_columns(
                # Polars will drop right columns with different names if they're simple matchups
                # Restore the right column names so the full self.shape & right.shape shape is
                # generated
                e.left.expr.alias(e.right.name)
                for e in on
                if isinstance(e, JoinOn)
                and isinstance(e.right, BoundDimension)
                and e.right.name not in already_populated
            )
        else:
            joined = self.dataframe.join(right.dataframe, how=how, **options)
        return LazyFrame["Intersection[_S_co, Q]"](self.shape & right.shape, joined)

    def sum(self) -> LazyFrame[_S_co]:
        """Sum all numeric columns in the frame, leaving other columns as null"""
        return LazyFrame(self.shape, self.dataframe.sum())

    def mean(self) -> LazyFrame[_S_co]:
        """Take the mean of all numeric columns in the frame, leaving other columns as null"""
        return LazyFrame(self.shape, self.dataframe.mean())

    def median(self) -> LazyFrame[_S_co]:
        """Take the median of all numeric columns in the frame, leaving other columns as null"""
        return LazyFrame(self.shape, self.dataframe.median())

    def max(self) -> LazyFrame[_S_co]:
        """Take the maximum of all columns in the frame"""
        return LazyFrame(self.shape, self.dataframe.max())

    def min(self) -> LazyFrame[_S_co]:
        """Take the minimum of all numeric columns in the frame"""
        return LazyFrame(self.shape, self.dataframe.min())

    def var(self) -> LazyFrame[_S_co]:
        """
        Take the variance value of all numeric columns in the frame, leaving all other columns as
        null
        """
        return LazyFrame(self.shape, self.dataframe.var())

    def quantile(
        self,
        quantile: float | ExoExpr[_S_co, float],
        interpolation: Literal[
            "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
        ] = "nearest",
    ) -> LazyFrame[_S_co]:
        """
        Take the `quantile`th (0.25, 0.75, etc.) value, using interpolation if there is no such
        exact value
        """
        return LazyFrame(
            self.shape,
            self.dataframe.quantile(
                quantile.expr if isinstance(quantile, Expr) else quantile,
                interpolation=interpolation,
            ),
        )

    def shift(self, n: int | ExoExpr[_S_co, int]) -> LazyFrame[_S_co]:
        """
        Progress all rows in the frame n entries forward, so now the 0th is the nth. If negative
        this would make the last one now the `length`-`n`th. Blank `null` rows will be inserted in
        the introduced gaps, and rows at the end will fall off, being removed from the resultant
        frame
        """
        return LazyFrame(self.shape, self.dataframe.shift(n.expr if isinstance(n, Expr) else n))

    def first(self) -> LazyFrame[_S_co]:
        """Take the first row of the dataframe, the same as `.head(1)`"""
        return LazyFrame(self.shape, self.dataframe.first())

    def last(self) -> LazyFrame[_S_co]:
        """Take the last row of the dataframe, the same as `.tail(1)`"""
        return LazyFrame(self.shape, self.dataframe.last())

    def gather(
        self, indices: LazySeries[int] | Sequence[int], *, null_on_oob: bool = False
    ) -> LazyFrame[_S_co]:
        """For each index in the provided indices, take the row at that index"""
        ix = indices.data if isinstance(indices, LazySeries) else indices
        return LazyFrame(self.shape, self.dataframe.gather(ix, null_on_oob=null_on_oob))

    def gather_every(self, n: int, offset: int = 0) -> LazyFrame[_S_co]:
        """Take each `n`th row from the frame, starting at `offset`"""
        return LazyFrame(self.shape, self.dataframe.gather_every(n, offset))

    def interpolate(self) -> LazyFrame[_S_co]:
        """Fill in null values between set values with linear interpolations"""
        return LazyFrame(self.shape, self.dataframe.interpolate())

    def limit(self, n: int = 5) -> LazyFrame[_S_co]:
        """Get the first `n` rows, alias for [head][typol.lazy.LazyFrame.head]"""
        return self.head(n)


@dataclasses.dataclass
class LazyGroupBy[S: Shape, Q: Shape]:
    shape: type[Q]
    group_by: pl.lazyframe.group_by.LazyGroupBy

    def agg(self, *agg: AggExpr[S, Q, Any]) -> LazyFrame[Q]:
        """Define the aggregating expressions to group rows in the dataframe"""
        return LazyFrame(self.shape, self.group_by.agg(*(e.expr for e in agg)))
