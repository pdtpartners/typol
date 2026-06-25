from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
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
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    assert_never,
    cast,
    overload,
)

import more_itertools
import polars as pl
from polars.interchange.dataframe import PolarsDataFrame

from typol.expr import (
    AggExpr,
    Alias,
    AliasShape,
    BoundDimension,
    ColumnInitializer,
    EndoAggExpr,
    EndoExpr,
    ExoAggExpr,
    ExoExpr,
    Explosion,
    Expr,
    JoinOn,
    Shape,
    Suffixed,
)
from typol.row import Row
from typol.series import Series

if TYPE_CHECKING:
    from ty_extensions import Intersection

    from typol.lazy import LazyFrame

CsvQuoteStyle: TypeAlias = Literal["necessary", "always", "non_numeric", "never"]
JoinTogetherType: TypeAlias = Literal["inner", "left", "right", "full", "cross", "outer"]
JoinAgainstType: TypeAlias = Literal["semi", "anti"]
JoinType: TypeAlias = JoinTogetherType | JoinAgainstType


class JoinOptions(TypedDict, total=False):
    nulls_equal: bool
    maintain_order: Literal["none", "left", "right", "left_right", "right_left"] | None
    validate: Literal["m:m", "m:1", "1:m", "1:1"]


_S_co = TypeVar("_S_co", bound=Shape, covariant=True)


@dataclasses.dataclass(init=False, eq=False, frozen=True)
class DataFrame(Generic[_S_co]):
    """Shape-bound dataframe whose operations are type checked"""

    shape: type[_S_co]
    dataframe: pl.DataFrame

    @overload
    def __init__(
        self,
        of: type[_S_co],
        /,
        values: Iterable[Mapping[str, Any]]
        | Iterable[tuple]
        | Iterable[ColumnInitializer[_S_co, Any]]
        | Mapping[BoundDimension[_S_co, Any], Iterable]
        | tuple[ColumnInitializer[_S_co, Any], ...]
        | pl.DataFrame
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
        | pl.DataFrame
        | None = None,
        *,
        orient: Literal["row", "col"] | None = None,
    ) -> None:
        meta = shape.shape_meta()
        if isinstance(values, pl.DataFrame):
            df = enforce_shape(shape, values)
        elif isinstance(values, Mapping):
            df = pl.DataFrame(
                {k.name if isinstance(k, BoundDimension) else k: vs for k, vs in values.items()},
                schema=meta.schema,
            )
        elif isinstance(values, Iterable):
            iterator = iter(values)
            first = more_itertools.first(iterator, None)
            if first is None:
                df = pl.DataFrame(schema=meta.schema)
            elif isinstance(first, ColumnInitializer):
                initializers = cast(Iterable[ColumnInitializer[_S_co, Any]], values)
                df = pl.DataFrame(
                    {i.dimension.name: i.value for i in initializers}, schema=meta.schema
                )
            elif isinstance(first, Mapping) and type(first) is not dict:
                values = cast(Iterable[Mapping], more_itertools.prepend(first, iterator))
                df = pl.DataFrame((dict(v) for v in values), schema=meta.schema)
            else:
                df = pl.DataFrame(
                    values
                    if isinstance(values, Collection)
                    else more_itertools.prepend(first, iterator),
                    schema=meta.schema,
                    orient=orient,
                )
        elif values is None:
            df = pl.DataFrame(schema=meta.schema)
        else:
            assert_never(values)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dataframe", df)

    @property
    def s(self) -> _S_co:
        """
        Provides a utility alias for accessing frame shape columns and attributes

        ```py
        purchases.filter(purchases.s.price > 10) == purchases.filter(Purchase.price > 10)
        ```

        This is particularly useful when the shape has been constructed implicitly rather than is
        explicitly defined:

        ```py
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

    def __getitem__[T](self, s: ExoExpr[_S_co, T] | ExoAggExpr[_S_co, T]) -> Series[T]:
        """Construct a series of the value of expr `s` for each row in the frame"""
        if not isinstance(s, BoundDimension):
            return Series[T](self.dataframe.select(s.expr).to_series())
        return self.get_column(s)

    def get_column[T](self, s: BoundDimension[_S_co, T]) -> Series[T]:
        return Series[T](self.dataframe[s.name])

    def head(self, n: int = 5) -> DataFrame[_S_co]:
        """Take only the first `n` rows"""
        return DataFrame(self.shape, self.dataframe.head(n))

    def tail(self, n: int = 5) -> DataFrame[_S_co]:
        """Take only the last `n` rows"""
        return DataFrame(self.shape, self.dataframe.tail(n))

    def bottom_k(
        self,
        k: int,
        *,
        by: ExoExpr[_S_co, Any] | Collection[ExoExpr[_S_co, Any]],
        reverse: bool = False,
    ) -> DataFrame[_S_co]:
        """Take only the smallest `k` rows, using `by` as the key"""
        exprs = by.expr if isinstance(by, Expr) else [e.expr for e in by]
        return DataFrame(self.shape, self.dataframe.bottom_k(k, by=exprs, reverse=reverse))

    def top_k(
        self,
        k: int,
        *,
        by: ExoExpr[_S_co, Any] | Collection[ExoExpr[_S_co, Any]],
        reverse: bool = False,
    ) -> DataFrame[_S_co]:
        """Take only the greatest `k` rows, using `by` as the key"""
        exprs = by.expr if isinstance(by, Expr) else [e.expr for e in by]
        return DataFrame(self.shape, self.dataframe.top_k(k, by=exprs, reverse=reverse))

    def slice(self, offset: int, length: int | None = None) -> DataFrame[_S_co]:
        return DataFrame(self.shape, self.dataframe.slice(offset, length))

    def lazy(self) -> LazyFrame[_S_co]:
        # Lazy import to avoid circular dependencies
        from typol.lazy import LazyFrame  # noqa: PLC0415,RUF100

        return LazyFrame(self.shape, self.dataframe.lazy())

    def collect(self, streaming: bool = False) -> DataFrame[_S_co]:
        return self

    def filter(self, *condition: ExoExpr[_S_co, bool]) -> DataFrame[_S_co]:
        """Only keep rows where the boolean conditions evaluate to `True`"""
        return DataFrame(self.shape, self.dataframe.filter(*(c.expr for c in condition)))

    @overload
    def with_columns(self, *columns: EndoExpr[_S_co, Any]) -> DataFrame[_S_co]: ...
    @overload
    def with_columns[A: LiteralString, AT](
        self, *columns: EndoExpr[_S_co, Any] | Alias[_S_co, A, AT]
    ) -> DataFrame[Intersection[_S_co, AliasShape[A, AT]]]: ...

    def with_columns[A: LiteralString, AT](
        self, *columns: EndoExpr[_S_co, Any] | Alias[_S_co, A, AT]
    ) -> DataFrame[Intersection[_S_co, AliasShape[A, AT]]]:
        """
        Use the provided expressions to update existing columns in the shape:

        ```py
        customers.with_columns(
            customers.s.age + 1,  # Add one to their age
            customers.s.name.fill_null(customers.s.phone)  # Use their phone number as a backup name
        )
        ```

        If adding or dropping columns, use [`transform`][typol.frame.DataFrame.transform] instead
        """
        return self.lazy().with_columns(*columns).collect()

    def transform[SNew: Shape](
        self, shape: type[SNew], *transforms: Expr[_S_co, SNew, Any]
    ) -> DataFrame[SNew]:
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

        See [with_columns][typol.frame.DataFrame.with_columns] when not changing between shapes
        """
        return self.lazy().transform(shape, *transforms).collect()

    def agg(self, *agg: EndoAggExpr[_S_co, Any]) -> DataFrame[_S_co]:
        """
        Define the aggregating expressions to group rows in the dataframe. Any columns not
        aggregated will be treated as the group by keys, since all columns must be preserved. To
        drop columns instead use `transform` before `agg`
        """
        return self.lazy().agg(*agg).collect()

    def group_by(self, *keys: EndoExpr[_S_co, Any]) -> GroupBy[_S_co, _S_co]:
        """
        Determine a series of expressions to group the dataframe by, this should be followed by an
        agg to apply aggregations to the grouped frame
        """
        return GroupBy(self.shape, self.dataframe.group_by(*(k.expr for k in keys)))

    def agg_transform[Q: Shape](
        self, shape: type[Q], *agg: AggExpr[_S_co, Q, Any] | Expr[_S_co, Q, Any]
    ) -> DataFrame[Q]:
        """
        Define the aggregating expressions to group rows in the dataframe. Any columns not
        aggregated will be treated as the group by keys, since all columns must be preserved. To
        drop columns instead use `transform` before `agg`.

        This allows transforming the aggregated columns since aggregation may change types
        """
        return self.lazy().agg_transform(shape, *agg).collect()

    def group_by_transform[Q: Shape](
        self, shape: type[Q], *keys: Expr[_S_co, Q, Any]
    ) -> GroupBy[_S_co, Q]:
        """
        Determine a series of expressions to group the dataframe by, this should be followed by an
        agg to apply aggregations to the grouped frame
        """
        return GroupBy(shape, self.dataframe.group_by(*(k.expr for k in keys)))

    def explode(self, *explosions: Explosion[_S_co, _S_co, Any]) -> DataFrame[_S_co]:
        """
        Take a series of list columns and create a new row for each value in the list:

        ```
        accounts.explode(
            Account.link_name.implode().over(Account.type).list.explode_to(Account.link_name)
        )
        ```

        The above will create a new row for every linked name from any account for the same type
        """
        return self.lazy().explode(*explosions).collect()

    def explode_transform[Q: Shape](
        self, shape: type[Q], *explosions: Explosion[_S_co, Q, Any] | Expr[_S_co, Q, Any]
    ) -> DataFrame[Q]:
        """Take a series of list columns and create a new row for each value in the list"""
        return self.lazy().explode_transform(shape, *explosions).collect()

    def vstack(self, other: DataFrame[_S_co]) -> DataFrame[_S_co]:
        # We know these share the same shape, but we don't know the order of the columns matches.
        # We select the columns to reorder them to match in the vstack
        return DataFrame(
            self.shape, self.dataframe.vstack(other.dataframe.select(self.dataframe.columns))
        )

    @staticmethod
    def concat[S: Shape](shape: type[S], frames: Iterable[DataFrame[S]]) -> DataFrame[S]:
        # Lazy import to avoid circular dependencies
        from typol.lazy import LazyFrame  # noqa: PLC0415,RUF100

        return LazyFrame.concat(shape, map(DataFrame.lazy, frames)).collect()

    def rechunk(self) -> DataFrame[_S_co]:
        return DataFrame(self.shape, self.dataframe.rechunk())

    def unique(
        self,
        *exprs: BoundDimension[_S_co, Any],
        keep: Literal["first", "last", "none", "any"] = "any",
        maintain_order: bool = False,
    ) -> DataFrame[_S_co]:
        return self.lazy().unique(*exprs, keep=keep, maintain_order=maintain_order).collect()

    def sort(
        self,
        *exprs: ExoExpr[_S_co, Any],
        descending: tuple[bool, ...] | bool = False,
        nulls_last: tuple[bool, ...] | bool = False,
        maintain_order: bool = False,
    ) -> DataFrame[_S_co]:
        return (
            self.lazy()
            .sort(
                *exprs, descending=descending, nulls_last=nulls_last, maintain_order=maintain_order
            )
            .collect()
        )

    def iter_raw(self) -> Iterator[tuple[Any, ...]]:
        """Yield each row of the frame as a tuple of values. Use `iter_rows` for well-typed access"""
        return self.dataframe.iter_rows(named=False)

    def iter_dicts(self) -> Iterator[dict[str, Any]]:
        """
        Yield each row of the frame as a dictionary of column name to value. Use `iter_rows` for
        well-typed access
        """
        return self.dataframe.iter_rows(named=True)

    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Return a list of the rows of the frame as a dictionary of column name to value. This is
        particularly useful for debugging for getting a Python object out of a Polars frame
        """
        return self.dataframe.to_dicts()

    def iter_rows(self) -> Iterator[Row[_S_co]]:
        """
        Yield a shape-typed `Row` for each row in the frame. Access to fields of these rows can
        be done in a well typed manner, using `row[S.column]`, which will have the right output type
        """
        return map(Row, self.dataframe.iter_rows(named=True))

    def is_empty(self) -> bool:
        return self.dataframe.is_empty()

    def __len__(self) -> int:
        """The number of rows in the dataframe"""
        return len(self.dataframe)

    def equals(self, other: DataFrame[_S_co]) -> bool:
        return self.dataframe.equals(other.dataframe)

    def suffix(self, suffixed: type[Suffixed[_S_co]] | None = None) -> DataFrame[Suffixed[_S_co]]:
        """
        Suffix the columns of the shape to distinguish them from conflicts with other shape column
        names, retyping the dataframe as `Suffixed[CurrentShape]`:

        ```py
        suffixed = customer.suffix()  # type: tp.DataFrame[Suffixed[Customer]]
        ```

        Suffixed shapes cannot have their fields accessed directly, instead must be accessed through
        projecting the dimensions via the suffixed shape:

        ```py
        # Note, suffixed.s(...) converts an original `Customer` dimension into a suffixed dimension
        suffixed[suffixed.s(Customer.name)].to_list()
        ```

        This is most useful in joint/intersection shapes, or self-joins:

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

        [see `expr.suffix`][typol.expr.suffix] for more info on shape suffixing
        """
        return self.lazy().suffix(suffixed).collect()

    @overload
    def glimpse(self, *, return_type: Literal["string"]) -> str: ...
    @overload
    def glimpse(self, *, return_type: Literal["frame"]) -> pl.DataFrame: ...
    @overload
    def glimpse(self, *, return_type: None = None) -> None: ...

    def glimpse(
        self, *, return_type: Literal["string", "frame"] | None = None
    ) -> str | pl.DataFrame | None:
        """Print a dense preview DataFrame"""
        return self.dataframe.glimpse(return_type=return_type)

    def pipe[**P, T](
        self, function: Callable[Concatenate[Self, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return function(self, *args, **kwargs)

    @staticmethod
    def read_csv[S: Shape](
        shape: type[S],
        source: IO[str] | str | bytes | Path | IO[bytes],
        mappings: Mapping[BoundDimension[S, Any], str] | None = None,
        *,
        has_header: bool = True,
        skip_rows: int = 0,
    ) -> DataFrame[S]:
        if not has_header:
            return DataFrame(
                shape,
                pl.read_csv(
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
            headers_only = pl.read_csv(source, infer_schema_length=0, skip_rows=skip_rows, n_rows=0)
            columns = {re.sub(r"[\W_]", "", h).lower(): h for h in headers_only.columns}
            mappings = {
                d: columns[re.sub(r"[\W_]", "", d.name).lower()]
                for d in shape.shape_meta().dimensions
            }

        return DataFrame(
            shape,
            pl.read_csv(
                source,
                columns=list(mappings.values()),
                schema_overrides={mappings[d]: d.ty.pl_ty for d in shape.shape_meta().dimensions},
                has_header=True,
                ignore_errors=True,
                truncate_ragged_lines=True,
                try_parse_dates=True,
                skip_rows=skip_rows,
                infer_schema=False,
            ).rename({mappings[d]: d.name for d in shape.shape_meta().dimensions}),
        )

    @overload
    def write_csv(
        self,
        sink: None = None,
        mappings: Mapping[BoundDimension[_S_co, Any], str]
        | Sequence[BoundDimension[_S_co, Any]]
        | None = None,
        *,
        include_header: bool = True,
        null_marker: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        line_terminator: str = "\n",
    ) -> str: ...
    @overload
    def write_csv(
        self,
        sink: IO[str] | str | Path | IO[bytes],
        mappings: Mapping[BoundDimension[_S_co, Any], str]
        | Sequence[BoundDimension[_S_co, Any]]
        | None = None,
        *,
        include_header: bool = True,
        null_marker: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        line_terminator: str = "\n",
    ) -> None: ...

    def write_csv(
        self,
        sink: IO[str] | str | Path | IO[bytes] | None = None,
        mappings: Mapping[BoundDimension[_S_co, Any], str]
        | Sequence[BoundDimension[_S_co, Any]]
        | None = None,
        *,
        include_header: bool = True,
        null_marker: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        line_terminator: str = "\n",
    ) -> str | None:
        """
        Output the dataframe to a file. By default, this uses the column names in the Shape. Use
        `mappings` to select which columns to output in order and rename them, e.g.

        ```
        {
            Account.name: "Account Name",
            Account.broker: "Broker"
        }
        ```

        or just provide a sequence of relevant columns in order: `(Account.name, Account.broker)`
        """
        dataframe = self.dataframe
        if isinstance(mappings, Sequence):
            dataframe = dataframe.select(c.name for c in mappings)
        elif isinstance(mappings, Mapping):
            dataframe = (
                dataframe.lazy()
                .rename({d.name: h for d, h in mappings.items()})
                .select(mappings.values())
                .collect(background=False)
            )

        return dataframe.write_csv(
            sink,
            include_header=include_header,
            null_value=null_marker,
            quote_style=quote_style,
            float_scientific=float_scientific,
            float_precision=float_precision,
            line_terminator=line_terminator,
        )

    def write_csv_of(
        self,
        sink: IO[str] | str | Path | IO[bytes],
        *exprs: ExoExpr[_S_co, Any],
        include_header: bool = True,
        null_marker: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        line_terminator: str = "\n",
    ) -> None:
        """
        Output the given expressions to a CSV. This is useful to apply a final transformation to the
        dataframe (e.g. for formatting or tidying up), without having to define a new `Shape`.

        The columns will be named based on the source shape unless renamed, use `to_out("...")` to
        rename the columns to an arbitrary value:

        ```
        df.write_csv_of(
            output_path,
            Account.balance.round(5).to_out("balance"),
            Account.closed.dt.strftime("%d/%m/%Y").to_out("Closed On")
            Account.code,  # column will just be "code"
        )
        """
        dataframe = self.dataframe.select(e.expr for e in exprs)

        dataframe.write_csv(
            sink,
            include_header=include_header,
            null_value=null_marker,
            quote_style=quote_style,
            float_scientific=float_scientific,
            float_precision=float_precision,
            line_terminator=line_terminator,
        )

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> PolarsDataFrame:
        """Deprecated compatibility with the Dataframe Interchange Protocol"""
        return self.dataframe.__dataframe__(nan_as_null, allow_copy)  # ty: ignore[deprecated]

    def join_asof[Q: Shape](
        self,
        right: DataFrame[Q],
        on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        strategy: Literal["backward", "forward", "nearest"] = "backward",
    ) -> DataFrame[Intersection[_S_co, Q]]:
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
        return DataFrame["Intersection[_S_co, Q]"](self.shape & right.shape, joined)

    @overload
    def join[Q: Shape](
        self,
        right: DataFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinTogetherType = "inner",
        **options: Unpack[JoinOptions],
    ) -> DataFrame[Intersection[_S_co, Q]]: ...
    @overload
    def join[Q: Shape](
        self,
        right: DataFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinAgainstType,
        **options: Unpack[JoinOptions],
    ) -> DataFrame[_S_co]: ...

    def join[Q: Shape](
        self,
        right: DataFrame[Q],
        *on: ExoExpr[_S_co | Q, Any] | JoinOn[_S_co, Q, Any],
        how: JoinType | Literal["anti"] = "inner",
        **options: Unpack[JoinOptions],
    ) -> DataFrame[Intersection[_S_co, Q]] | DataFrame[_S_co]:
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

        [see `DataFrame.suffix`][typol.frame.DataFrame.suffix]

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
        return self.lazy().join(right.lazy(), *on, how=how, **options).collect()

    def sum(self) -> DataFrame[_S_co]:
        """Sum all numeric columns in the frame, leaving other columns as null"""
        return DataFrame(self.shape, self.dataframe.sum())

    def mean(self) -> DataFrame[_S_co]:
        """Take the mean of all numeric columns in the frame, leaving other columns as null"""
        return DataFrame(self.shape, self.dataframe.mean())

    def median(self) -> DataFrame[_S_co]:
        """Take the median of all numeric columns in the frame, leaving other columns as null"""
        return DataFrame(self.shape, self.dataframe.median())

    def max(self) -> DataFrame[_S_co]:
        """Take the maximum of all columns in the frame"""
        return DataFrame(self.shape, self.dataframe.max())

    def min(self) -> DataFrame[_S_co]:
        """Take the minimum of all numeric columns in the frame"""
        return DataFrame(self.shape, self.dataframe.min())

    def var(self) -> DataFrame[_S_co]:
        """
        Take the variance value of all numeric columns in the frame, leaving all other columns as
        null
        """
        return DataFrame(self.shape, self.dataframe.var())

    def quantile(
        self,
        quantile: float,
        interpolation: Literal[
            "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
        ] = "nearest",
    ) -> DataFrame[_S_co]:
        """
        Take the `quantile`th (0.25, 0.75, etc.) value, using interpolation if there is no such
        exact value
        """
        return DataFrame(self.shape, self.dataframe.quantile(quantile, interpolation=interpolation))

    def shift(self, n: int) -> DataFrame[_S_co]:
        """
        Progress all rows in the frame n entries forward, so now the 0th is the nth. If negative
        this would make the last one now the `length`-`n`th. Blank `null` rows will be inserted in
        the introduced gaps, and rows at the end will fall off, being removed from the resultant
        frame
        """
        return DataFrame(self.shape, self.dataframe.shift(n))

    def gather(
        self, indices: Series[int] | Sequence[int], *, null_on_oob: bool = False
    ) -> DataFrame[_S_co]:
        """For each index in the provided indices, take the row at that index"""
        ix = indices.data if isinstance(indices, Series) else indices
        return DataFrame(self.shape, self.dataframe.gather(ix, null_on_oob=null_on_oob))

    def gather_every(self, n: int, offset: int = 0) -> DataFrame[_S_co]:
        """Take each `n`th row from the frame, starting at `offset`"""
        return DataFrame(self.shape, self.dataframe.gather_every(n, offset))

    def interpolate(self) -> DataFrame[_S_co]:
        """Fill in null values between set values with linear interpolations"""
        return DataFrame(self.shape, self.dataframe.interpolate())

    def limit(self, n: int = 5) -> DataFrame[_S_co]:
        """Get the first `n` rows, alias for [head][typol.lazy.LazyFrame.head]"""
        return self.head(n)


@overload
def enforce_shape[S: Shape](shape: type[S], dataframe: pl.DataFrame) -> pl.DataFrame: ...
@overload
def enforce_shape[S: Shape](shape: type[S], dataframe: pl.LazyFrame) -> pl.LazyFrame: ...


def enforce_shape[S: Shape](
    shape: type[S], dataframe: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame | pl.LazyFrame:
    """
    Select the relevant columns from the Polars frame and strict cast them to ensure they are
    typed correctly. This is effectively to project-and-assert `shape`
    """
    return dataframe.select(
        pl.col(d).cast(t, strict=True) for d, t in shape.shape_meta().datatypes.items()
    )


@dataclasses.dataclass
class GroupBy[S: Shape, Q: Shape]:
    shape: type[Q]
    group_by: pl.dataframe.frame.GroupBy

    def agg(self, *agg: AggExpr[S, Q, Any]) -> DataFrame[Q]:
        """Define the aggregating expressions to group rows in the dataframe"""
        return DataFrame(self.shape, self.group_by.agg(*(e.expr for e in agg)))
