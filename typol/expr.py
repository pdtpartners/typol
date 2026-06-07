"""Typed wrappers for expr and schema (shape) operations, the core of Typol"""

from __future__ import annotations

import builtins
import dataclasses
import datetime
import enum
from abc import ABC
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Never,
    Self,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

import polars as pl
import polars.datatypes
import polars.expr.whenthen
from more_itertools import first

from typol.series import Series
from typol.types import STRING, EnumOf, StructMapping, Type, Typeable, from_typeable, normalize_enum

if TYPE_CHECKING:
    from ty_extensions import Intersection

    from typol.row import Row

type Value = (
    int
    | float
    | str
    | EnumOf[int]
    | EnumOf[float]
    | Decimal
    | datetime.date
    | datetime.datetime
    | datetime.time
    | datetime.timedelta
    | Sequence[Value]
    | Mapping[str, Value]
    | enum.Enum
)


@dataclasses.dataclass(frozen=True)
class ShapeMeta[S: Shape]:
    """
    A wrapper object that all library level definitions are on to avoid name conflicts with a
    shape's dimensions

    This provides utilities for inspecting the shape and schema
    """

    shape: type[S]

    @property
    def datatypes(self) -> Mapping[str, pl.DataType | type]:
        """A mapping from dimension name to their polars data type"""
        return {d.name: d.ty.pl_ty for d in self.dimensions}

    @property
    def schema(self) -> pl.Schema:
        """
        A polars runtime schema to direct it how to configure (and enforce the types on) the
        dataframe
        """
        return pl.Schema(self.datatypes)

    @property
    def dimensions(self) -> Iterable[BoundDimension[S, Any]]:
        """Iterate through the dimensions defined in the shape"""
        # Anything locally overwritten should be considered seen, so not pulled from base classes
        seen = set(vars(self.shape).keys())
        # Iterate through base classes, rather than use `__dict__`, to preserve definition order
        # which must match runtime behaviour
        for cls in self.shape.__bases__:
            if issubclass(cls, Shape) and cls is not Shape:
                for dimension in cls.shape_meta().dimensions:
                    if dimension.name not in seen:
                        yield dataclasses.replace(dimension, shape=self.shape)
                    seen.add(dimension.name)

        # Use vars(...).keys to preserve definition order of locally defined dimensions
        for key in vars(self.shape).keys():
            # Can't use `vars(...).values()` since the descriptor won't be bound, have to getattr
            if isinstance(value := getattr(self.shape, key), BoundDimension):
                yield value


class ShapeType(type):
    """Metaclass defining shape-level operators"""

    def shape_intersection[S: Shape, Q: Shape](
        cls: type[S], other: type[Q]
    ) -> Intersection[type[S], type[Q]]:
        return cls & other

    def __and__[S: Shape, Q: Shape](cls: type[S], other: type[Q]) -> Intersection[type[S], type[Q]]:
        if cls is other:
            return cls
        return cast(
            "Intersection[S, Q]", ShapeType(f"{cls.__name__}&{other.__name__}", (cls, other), {})
        )


class Shape(metaclass=ShapeType):
    """
    This is the core component of typed polars, that lets you define the static column names and
    types of a dataframe much like a dataclass.

    ```python
    class Account(Shape):
        name = Dimension(str)
        code = Dimension(int)
    ```

    defines a two-column dataframe with a `str` and an `int` column. Operations can be done on the
    dataframe using the fields of the shape type, e.g.:

    ```python
    df.filter(Account.code.ne(0), Account.name.str.contains("SIM").not_())
    ```
    """

    @classmethod
    def shape_meta(cls) -> ShapeMeta[Self]:
        """Access utility functions for inspecting a shape"""
        return ShapeMeta(cls)


@dataclasses.dataclass(frozen=True)
class Dimension[T]:
    """
    A shape `Dimension` declares a column in the dataclass. This should be a class-level field of
    a `Shape` subtype, and will be a `BoundDimension` when accessed as `MyShape.my_dimension`

    You must provide a type when declaring a dimension, and can optionally set a polars type
    implementation and an underlying name to use. By default, the name will be the name of the field
    """

    ty: Type[T]
    name: str = ""

    def __get__[S: Shape](
        self, shape: S | None, shape_type: type[S] | None = None
    ) -> BoundDimension[S, T]:
        """
        This is the secret sauce: when a shape's dimensions are accessed by `MyShape.my_dimension`,
        this binds the shape to the dimension at the type level, which means this shape can then
        be enforced by all the operations using any dimension, and any expression created from this
        bound dimension can continue to refer to the shape it operates on in its type too by passing
        the type parameter along
        """
        shape_ty = shape_type or shape is not None and type(shape)
        assert shape_ty
        return BoundDimension(shape_ty, self.ty, self.name)

    def __set_name__(self, _owner: type, name: str) -> None:
        if not self.name:
            object.__setattr__(self, "name", name)


def dimension[T](ty: Typeable[T], name: str = "") -> Dimension[T]:
    return Dimension(from_typeable(ty), name)


def _pl_expr(expr: AggExpr | Expr | object) -> pl.Expr:
    """Take an expression or literal and convert it into a polars expression"""
    return (
        expr.expr
        if isinstance(expr, (Expr, AggExpr))
        else pl.lit(normalize_enum(expr), dtype=from_typeable(type(expr)).pl_ty)
    )


def _pl_expr_or_none(expr: AggExpr | Expr | object | None) -> pl.Expr | None:
    return _pl_expr(expr) if expr is not None else None


def _expr_or_lit[S: Shape, T](expr: Expr[S, Never, T] | T) -> Expr[S, Never, T]:
    """Take an expression or literal and convert it into a typed polars expression"""
    if isinstance(expr, Expr):
        return cast(Expr[S, Never, T], expr)
    else:
        return Expr.lit(expr)


# An expression that is local to a shape
type EndoAggExpr[S: Shape, T] = AggExpr[S, S, T]
# An expression where the output shape is irrelevant, useful for when an expression is being
# consumed rather than stored, i.e. in a filter
type ExoAggExpr[S: Shape, T] = AggExpr[S, Never, T]
# An expression which is intermediary and cannot be stored in a shape, it must be mapped to a new
# column, e.g. using `to`, to be stored
type MesoAggExpr[S: Shape, T] = AggExpr[S, Never, T]


@dataclasses.dataclass(frozen=True)
class AggExpr[S: Shape, R: Shape, T]:
    """
    An expression created by an aggregation function (e.g. `.sum()`). This can't be used as a normal
    expression, only as an aggregated value in an `.agg(...)` argument
    """

    expr: pl.Expr

    def coalesce(self, *others: ExoAggExpr[S, T]) -> AggExpr[S, R, T]:
        return AggExpr(pl.coalesce(self.expr, *map(_pl_expr, others)))

    def null_when_eq(self, value: ExoAggExpr[S, T] | T) -> AggExpr[S, R, T]:
        return AggExpr(pl.when(self.expr.ne(_pl_expr(value))).then(self.expr))

    def null_insignificant[N: (float, int)](self: AggExpr[S, R, N]) -> AggExpr[S, R, N]:
        return AggExpr(
            pl.when(self.expr.is_not_nan(), self.expr.is_not_null(), self.expr.ne(0)).then(
                self.expr
            )
        )

    def to[Q: Shape](self, dimension: BoundDimension[Q, T]) -> AggExpr[S, Q, T]:
        # We don't cast here in case it's an `agg`; Polars has weird casting behaviour with
        # this and there's no way to tell Polars is this to be aggregated
        return AggExpr(self.expr.alias(dimension.name))

    def over[Q: Shape](
        self,
        *exprs: ExoExpr[S, Any] | ExoExpr[Q, Any],
        order_by: Iterable[ExoExpr[S, Any] | ExoExpr[Q, Any]] | None = None,
        mapping_strategy: Literal["group_to_rows", "join", "explode"] = "group_to_rows",
    ) -> Expr[Intersection[S, Q], R, T]:
        """
        Restrict an aggregating expression to just a window (i.e. bucket) of values keyed on by
        `exprs`. "group_to_rows" matches values up to the current rows, "join" matches them back to
        the source rows, and "explode" does the same as join, but if there are multiple values from
        the aggregating expression, it will duplicate the existing line into multiple per each
        aggregated value:

        ```
        # Find multiple ids attached to the same username
        accounts.filter(Account.id.count().over(Account.username).gt(1))
        ```
        """
        return IntermediateExpr(
            self.expr.over(
                # Polars needs these key names to be unique, otherwise it raises `DuplicateError`
                *(_pl_expr(e).alias(f"_key{i}") for i, e in enumerate(exprs)),
                order_by=(_pl_expr(e) for e in order_by) if order_by else None,
                mapping_strategy=mapping_strategy,
            )
        )


@dataclasses.dataclass(frozen=True)
class StrExprNamespace[S: Shape, R: Shape, T]:
    """Namespace for string functions, similar to `pl.Expr.str`"""

    expr: Expr[S, R, T]

    def contains[Q: Shape](
        self, substring: ExoExpr[Q, str] | str, literal: bool = False
    ) -> MesoExpr[Intersection[S, Q], bool]:
        """Whether each column value contains the regex (or if `literal` is set, the fixed-string)"""
        substr = _pl_expr(substring)
        return IntermediateExpr(self.expr.expr.str.contains(substr, literal=literal))

    def starts_with[Q: Shape](
        self, suffix: ExoExpr[Q, str] | str
    ) -> MesoExpr[Intersection[S, Q], bool]:
        """Whether each column value starts with the given fixed string"""
        return IntermediateExpr(self.expr.expr.str.starts_with(_pl_expr(suffix)))

    def ends_with[Q: Shape](
        self, suffix: ExoExpr[Q, str] | str
    ) -> MesoExpr[Intersection[S, Q], bool]:
        """Whether each column value ends with the given fixed string"""
        return IntermediateExpr(self.expr.expr.str.ends_with(_pl_expr(suffix)))

    def len_chars(self) -> MesoExpr[S, int]:
        """Count the number of unicode characters in the string"""
        return IntermediateExpr(self.expr.expr.str.len_chars())

    def join(self, sep: str) -> AggExpr[S, R, str]:
        """Aggregate a group of strings by interspersing `sep` between them and concatenating"""
        return AggExpr(self.expr.expr.str.join(sep))

    def replace[SA: Shape, SB: Shape](
        self,
        pattern: ExoExpr[SA, str] | str,
        value: ExoExpr[SB, str] | str,
        literal: bool = False,
        n: int = 1,
    ) -> Expr[Intersection[S, SA] | SB, R, str]:
        """
        Replace `n` matches for `pattern` (regex, or fixed-string if `literal` is `True`) with
        `value`
        """
        return IntermediateExpr(
            self.expr.expr.str.replace(_pl_expr(pattern), _pl_expr(value), literal=literal, n=n)
        )

    def replace_all[SA: Shape, SB: Shape](
        self, pattern: ExoExpr[SA, str] | str, value: ExoExpr[SB, str] | str, literal: bool = False
    ) -> Expr[Intersection[S, SA] | SB, R, str]:
        """
        Replace all matches for `pattern` (regex, or fixed-string if `literal` is `True`) with
        `value`
        """
        return IntermediateExpr(
            self.expr.expr.str.replace_all(_pl_expr(pattern), _pl_expr(value), literal=literal)
        )

    def strip_chars[SA: Shape](
        self, characters: ExoExpr[SA, str] | str | None = None
    ) -> Expr[Intersection[S, SA], R, str]:
        """Remove leading and trailing characters in the given string. By default removes whitespace"""
        return IntermediateExpr(self.expr.expr.str.strip_chars(_pl_expr_or_none(characters)))

    def to_date(self, format: str, *, strict: bool = True) -> MesoExpr[S, datetime.date]:
        return IntermediateExpr(self.expr.expr.str.to_date(format, strict=strict))

    def to_datetime(
        self,
        format: str,
        *,
        time_unit: Literal["ns", "us", "ms"] | None = None,
        strict: bool = True,
    ) -> MesoExpr[S, datetime.date]:
        return IntermediateExpr(
            self.expr.expr.str.to_datetime(format, strict=strict, time_unit=time_unit)
        )

    def to_lowercase(self) -> Expr[S, R, T]:
        return IntermediateExpr(self.expr.expr.str.to_lowercase())

    def to_uppercase(self) -> Expr[S, R, T]:
        return IntermediateExpr(self.expr.expr.str.to_uppercase())

    def to_titlecase(self) -> Expr[S, R, T]:
        return IntermediateExpr(self.expr.expr.str.to_titlecase())

    def to_decimal(self, *, scale: int) -> ExoExpr[S, Decimal]:
        return IntermediateExpr(self.expr.expr.str.to_decimal(scale=scale))

    def to_integer[SA: Shape](
        self, *, base: ExoExpr[SA, int] | int = 10, dtype: Typeable[int] = int, strict: bool = True
    ) -> ExoExpr[Intersection[S, SA], int]:
        return IntermediateExpr(
            self.expr.expr.str.to_integer(
                base=_pl_expr(base),
                dtype=cast(polars.datatypes.IntegerType, from_typeable(dtype).pl_ty),
                strict=strict,
            )
        )

    def is_not_empty(self) -> MesoExpr[S, bool]:
        return IntermediateExpr(self.expr.expr.is_not_null() & self.expr.expr.ne(""))

    def split[SA: Shape](
        self, sep: ExoExpr[SA, str] | str
    ) -> MesoExpr[Intersection[S, SA], list[str]]:
        """Break a string into a list of strings, using `sep` as the separator"""
        return IntermediateExpr(self.expr.expr.str.split(_pl_expr(sep)))


@dataclasses.dataclass(frozen=True)
class DtExprNamespace[S: Shape, R: Shape, T]:
    """Namespace for date and datetime functions, similar to `pl.Expr.dt`"""

    expr: Expr[S, R, T]

    def date(self) -> Expr[S, R, datetime.date]:
        """Just take the date component of a datetime"""
        return IntermediateExpr(self.expr.expr.dt.date())

    def offset_by[Q: Shape](
        self, offset: str | ExoExpr[Q, str]
    ) -> Expr[Intersection[S, Q], R, datetime.date]:
        """
        Add an amount of time to a date or datetime, see `pl.offset_by` for all allowed interval
        strings, but examples include `-1y` or `6mo3m2s`
        """
        return IntermediateExpr(self.expr.expr.dt.offset_by(_pl_expr(offset)))

    def strftime(self, format: str) -> MesoExpr[S, str]:
        return IntermediateExpr(self.expr.expr.dt.strftime(format))

    def year(self) -> MesoExpr[S, int]:
        """Year number from date"""
        return IntermediateExpr(self.expr.expr.dt.year())

    def month(self) -> MesoExpr[S, int]:
        """Month number from date"""
        return IntermediateExpr(self.expr.expr.dt.month())

    def day(self) -> MesoExpr[S, int]:
        """Day of month from date"""
        return IntermediateExpr(self.expr.expr.dt.day())

    def add_business_days[Q: Shape](
        self,
        offset: int | ExoExpr[Q, int],
        roll: Literal["step", "snap", "raise", "forward", "backward"] = "step",
    ) -> Expr[Intersection[S, Q], R, datetime.date]:
        """
        Add `offset` business days to the current day. If the current day is not a business day, it
        will be treated based on `roll`:

            - `"step"`: Treat the first step from the current non-business day as if it is moving
              from a business day. For zero, it heads to the business day it would move from to go
              forwards (back 1)
            - `"snap"`: In the direction of `offset`, move to a business day before adding. For
              zero, it heads to the business day it would move from to go forwards (forward 1)
            - `"raise"`: Throw an error if not starting on a business day
            - `"forward"`: Snap to the next business day
            - `"backward:` Snap to the previous business day
        """
        dt = self.expr.expr.dt
        match roll:
            case "snap" | "step":
                go_forward = _pl_expr(offset).ge(0) if roll == "snap" else _pl_expr(offset).le(0)
                dt = (
                    pl.when(go_forward)
                    .then(dt.add_business_days(0, roll="forward"))
                    .otherwise(dt.add_business_days(0, roll="backward"))
                ).dt
                roll = "raise"
        return IntermediateExpr(dt.add_business_days(_pl_expr(offset), roll=roll))

    def total_days(self: DtExprNamespace[S, R, datetime.timedelta]) -> MesoExpr[S, int]:
        """The total number of days represented by the duration"""
        return IntermediateExpr(self.expr.expr.dt.total_days())

    def total_seconds(self: DtExprNamespace[S, R, datetime.timedelta]) -> MesoExpr[S, int]:
        """The total number of seconds represented by the duration"""
        return IntermediateExpr(self.expr.expr.dt.total_seconds())

    def weekday(self: DtExprNamespace[S, R, datetime.date]) -> MesoExpr[S, int]:
        """
        Day of week between 1 (Monday) and 7 (Sunday), you'll need to `- 1` to be compatible with
        `import calendar`
        """
        return IntermediateExpr(self.expr.expr.dt.weekday())

    def month_start(self: DtExprNamespace[S, R, datetime.date]) -> Expr[S, R, datetime.date]:
        """Backward to the first of the same month"""
        return IntermediateExpr(self.expr.expr.dt.month_start())

    def month_end(self: DtExprNamespace[S, R, datetime.date]) -> Expr[S, R, datetime.date]:
        """Forward to of the last of the same month"""
        return IntermediateExpr(self.expr.expr.dt.month_end())


@dataclasses.dataclass(frozen=True)
class ListExprNamespace[S: Shape, R: Shape, T]:
    """Namespace for list functions, similar to `pl.Expr.list`"""

    expr: Expr[S, R, list[T]]

    def explode_to[Q: Shape](self, to: BoundDimension[Q, T]) -> Explosion[S, Q, T]:
        """
        Explode the entire dataframe around for this list column, creating a new row for every list
        entry in a existing row
        """
        return Explosion(self.expr, to)

    def unique(self) -> Expr[S, R, list[T]]:
        """Remove duplicate elements from the list"""
        return IntermediateExpr(self.expr.expr.list.unique())

    def sort(self, descending: bool = False) -> Expr[S, R, list[T]]:
        """Order the eleements of the list"""
        return IntermediateExpr(self.expr.expr.list.sort(descending=descending))

    def explode(self) -> MesoExpr[S, T]:
        """
        Flatten a list expression into one entry per list element. This resizes the resultant series
        to the sum of the length of the lists.

        Note: Since this resizes the result, it is unsafe to simply map back to a column. Instead,
        it's useful for flattening arrays before applying some aggregate expression
        """
        return IntermediateExpr(self.expr.expr.list.explode())

    def set_difference[Q: Shape](
        self, other: ExoExpr[Q, list[T]] | list[T]
    ) -> Expr[Intersection[S, Q], R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.set_difference(_pl_expr(other)))

    def set_intersection[Q: Shape](
        self, other: ExoExpr[Q, list[T]] | list[T]
    ) -> Expr[Intersection[S, Q], R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.set_intersection(_pl_expr(other)))

    def set_union[Q: Shape](
        self, other: ExoExpr[Q, list[T]] | list[T]
    ) -> Expr[Intersection[S, Q], R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.set_union(_pl_expr(other)))

    def contains[Q: Shape](self, other: ExoExpr[Q, T] | T) -> MesoExpr[Intersection[S, Q], bool]:
        return IntermediateExpr(self.expr.expr.list.contains(_pl_expr(other)))

    def len(self) -> MesoExpr[S, int]:
        return IntermediateExpr(self.expr.expr.list.len())

    def drop_nulls(self) -> Expr[S, R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.drop_nulls())

    def concat[Q: Shape](
        self, other: ExoExpr[Q, list[T]] | list[T]
    ) -> Expr[Intersection[S, Q], R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.concat(_pl_expr(other)))

    def head[Q: Shape](self, n: int | ExoExpr[Q, int]) -> Expr[Intersection[S, Q], R, list[T]]:
        return IntermediateExpr(self.expr.expr.list.head(_pl_expr(n)))

    def get[Q: Shape](
        self, index: ExoExpr[Q, int] | int, null_on_oob: bool = False
    ) -> Expr[S, R, T]:
        return IntermediateExpr(self.expr.expr.list.get(_pl_expr(index), null_on_oob=null_on_oob))

    def join[Q: Shape](
        self: ListExprNamespace[S, R, str], sep: ExoExpr[Q, str] | str
    ) -> Expr[Intersection[S, Q], R, str]:
        return IntermediateExpr(self.expr.expr.list.join(_pl_expr(sep)))

    def eval[U](self, expr: ExoExpr[Element[T], U]) -> MesoExpr[S, list[U]]:
        """
        Evaluate an expression against each element of the list, effectively `map` but for Polars.

        Imagine we had the ages for various family members, and we wanted to know the age in 5
        years time:

        +---------+-----------+
        | surname |   ages    |
        +---------+-----------+
        | Baggins | [111, 33] |
        | Gamgee  | [38]      |
        +---------+-----------+

        To transform these ages, we could do:

        ```python
        five_years_from_now = families.with_columns(
            Family.ages.eval(Element.element() + 5).to(Family.ages)
        )
        ```
        """
        return IntermediateExpr(self.expr.expr.list.eval(expr.expr))


@dataclasses.dataclass(frozen=True)
class StructExprNamespace[S: Shape, R: Shape, M: Shape]:
    """Namespace for struct functions"""

    expr: Expr[S, R, StructMapping[M]]

    def field[T](self, dim: BoundDimension[M, T]) -> MesoExpr[S, T]:
        return IntermediateExpr(self.expr.expr.struct.field(dim.name))

    def map_rows_to[T, Q: Shape](
        self, transform: Callable[[Row[M]], T | None], to: BoundDimension[Q, T]
    ) -> Expr[S, Q, T]:
        """Apply a Python transformation on `Row`s to the structs in a column"""
        # TODO(racharm) Use PEP810 lazy imports with py315, needed to avoid circular import issues
        from typol.row import Row  # noqa: PLC0415,RUF100

        return self.expr.map_to(lambda x: transform(Row.from_struct_mapping(x)), to)


_S_contra = TypeVar("_S_contra", bound="Shape", contravariant=True)
_R_contra = TypeVar("_R_contra", bound="Shape", contravariant=True)
_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class Explosion(Generic[_S_contra, _R_contra, _T]):
    """An expression that can "explode" a frame to a new row for each output value"""

    expr: ExoExpr[_S_contra, list[_T]]
    to: BoundDimension[_R_contra, _T]


class _ReplaceKwargs(TypedDict, total=False):
    default: pl.Expr | None


class Expr(ABC, Generic[_S_contra, _R_contra, _T]):
    """
    Base class for all expressions, defining the common operations such as comparison and
    transformation
    """

    expr: pl.Expr

    def gt[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.gt(_pl_expr(other)))

    def lt[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.lt(_pl_expr(other)))

    def ge[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.ge(_pl_expr(other)))

    def le[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.le(_pl_expr(other)))

    def eq[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.eq(_pl_expr(other)))

    def ne[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return IntermediateExpr(self.expr.ne(_pl_expr(other)))

    def __gt__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.gt(other)

    def __lt__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.lt(other)

    def __ge__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.ge(other)

    def __le__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.le(other)

    def __ne__[SA: Shape](  # ty: ignore[invalid-method-override]
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.ne(other)

    def __eq__[SA: Shape](  # ty: ignore[invalid-method-override]
        self, other: ExoExpr[SA, _T] | _T
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        return self.eq(other)

    def __hash__[SA: Shape](self) -> int:
        return hash(self.expr)

    def is_between[SA: Shape](
        self,
        start: ExoExpr[SA, _T] | _T,
        end: ExoExpr[SA, _T] | _T,
        closed: Literal["both", "left", "right", "none"] = "both",
    ) -> MesoExpr[Intersection[_S_contra, SA], bool]:
        """Check if this expression is between the given lower and upper bounds"""
        return IntermediateExpr(self.expr.is_between(_pl_expr(start), _pl_expr(end), closed))

    def round[N: (float, int)](
        self: Expr[_S_contra, _R_contra, N], decimals: int = 0
    ) -> Expr[_S_contra, _R_contra, N]:
        return IntermediateExpr(self.expr.round(decimals))

    def floor[N: (float, int)](
        self: Expr[_S_contra, _R_contra, N],
    ) -> Expr[_S_contra, _R_contra, N]:
        return IntermediateExpr(self.expr.floor())

    def ceil[N: (float, int)](self: Expr[_S_contra, _R_contra, N]) -> Expr[_S_contra, _R_contra, N]:
        return IntermediateExpr(self.expr.ceil())

    def is_null(self) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_null())

    def is_nan(
        self: Expr[_S_contra, _R_contra, float] | Expr[_S_contra, _R_contra, int],
    ) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_nan())

    def is_infinite(
        self: Expr[_S_contra, _R_contra, float] | Expr[_S_contra, _R_contra, int],
    ) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_infinite())

    def is_not_null(self) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_not_null())

    def is_not_nan(
        self: Expr[_S_contra, _R_contra, float] | Expr[_S_contra, _R_contra, int],
    ) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_not_nan())

    def is_finite(
        self: Expr[_S_contra, _R_contra, float] | Expr[_S_contra, _R_contra, int],
    ) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_finite())

    def is_significant[N: (int, float)](
        self: Expr[_S_contra, _R_contra, N],
    ) -> MesoExpr[_S_contra, bool]:
        """Is the numeric value a significant number, not `nan`, `0` or `null`"""
        return IntermediateExpr(self.expr.is_not_nan() & self.expr.is_not_null() & self.expr.ne(0))

    def not_(self: Expr[_S_contra, _R_contra, bool]) -> Expr[_S_contra, _R_contra, bool]:
        return IntermediateExpr(self.expr.not_())

    def is_in[SA: Shape](
        self, other: ExoExpr[SA, builtins.list[_T]] | Collection[_T] | Series[_T]
    ) -> MesoExpr[_S_contra, bool]:
        match other:
            case Series():
                return IntermediateExpr(self.expr.is_in(other.data))
            case Expr():
                return IntermediateExpr(self.expr.is_in(other.expr))
            case Collection():
                return IntermediateExpr(self.expr.is_in(normalize_enum(other)))

    @staticmethod
    def lit[L](expr: L | None) -> MesoExpr[Shape, L]:
        return IntermediateExpr(_pl_expr(expr))

    @overload
    def fill_null[SA: Shape](
        self, fill: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def fill_null[SA: Shape](
        self, *, strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...

    def fill_null[SA: Shape](
        self,
        fill: ExoExpr[SA, _T] | _T | None = None,
        strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"] | None = None,
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(
            self.expr.fill_null(_pl_expr(fill) if fill is not None else None, strategy)
        )

    def fill_nan[SA: Shape](
        self, fill: ExoExpr[SA, _T] | _T | None
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(self.expr.fill_nan(_pl_expr(fill)))

    def reverse(self) -> Expr[_S_contra, _R_contra, _T]:
        return IntermediateExpr(self.expr.reverse())

    def to[Q: Shape](self, dimension: BoundDimension[Q, _T]) -> Expr[_S_contra, Q, _T]:
        return IntermediateExpr(self.expr.alias(dimension.name).cast(dimension.ty.pl_ty))

    def to_out(self, label: builtins.str) -> Expr[_S_contra, Shape, _T]:
        """
        The _out variant of `to` lets you rename a column, but it must be renamed again before it
        can be stored in a shape. However, if you're  going out to a file, this controls the output
        column name, so is most useful with `transform_write_csv`
        """
        return IntermediateExpr(self.expr.alias(label))

    def agg(self) -> MesoAggExpr[_S_contra, builtins.list[_T]]:
        """Collect all values in the given group into a list"""
        return AggExpr(self.expr)

    def gather_every(self, n: int, offset: int = 0) -> MesoAggExpr[_S_contra, builtins.list[_T]]:
        """Collect all values in the given group into a list"""
        return AggExpr(self.expr.gather_every(n, offset))

    def sum(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.sum())

    def mode(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.mode())

    def mean[N: (int, float, Decimal)](
        self: Expr[_S_contra, _R_contra, N],
    ) -> AggExpr[_S_contra, _R_contra, N]:
        return AggExpr(self.expr.mean())

    def median[N: (int, float, Decimal)](
        self: Expr[_S_contra, _R_contra, N],
    ) -> AggExpr[_S_contra, _R_contra, N]:
        return AggExpr(self.expr.median())

    def unique(self, maintain_order: bool = False) -> Expr[_S_contra, _R_contra, _T]:
        return IntermediateExpr(self.expr.unique(maintain_order=maintain_order))

    def drop_nulls(self) -> Expr[_S_contra, _R_contra, _T]:
        return IntermediateExpr(self.expr.drop_nulls())

    def sort(self, *, descending: bool = False) -> Expr[_S_contra, _R_contra, _T]:
        return IntermediateExpr(self.expr.sort(descending=descending))

    def first(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.first())

    def last(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.last())

    def min(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.min())

    def max(self) -> AggExpr[_S_contra, _R_contra, _T]:
        return AggExpr(self.expr.max())

    def count(self) -> MesoAggExpr[_S_contra, int]:
        return AggExpr(self.expr.count())

    def len(self) -> MesoAggExpr[_S_contra, int]:
        return AggExpr(self.expr.len())

    def implode(self: MesoExpr[_S_contra, _T]) -> MesoExpr[_S_contra, builtins.list[_T]]:
        """
        Group all the elements into a single. This resizes the resultant series to a single element.

        Note: Since this resizes the result, it is unsafe to simply map back to a column. Instead,
        it's useful for creating arrays for intermediates used in `ExoExpr`s. You might want
        `agg` instead to create a aggregated list out of elements
        """
        return IntermediateExpr(self.expr.implode())

    @overload
    def over[Q: Shape](
        self,
        *exprs: ExoExpr[_S_contra, Any]
        | ExoExpr[Q, Any]
        | ExoExpr[Intersection[_S_contra, Q], Any],
        order_by: Iterable[
            ExoExpr[_S_contra, Any] | ExoExpr[Q, Any] | ExoExpr[Intersection[_S_contra, Q], Any]
        ]
        | None = None,
        mapping_strategy: Literal["group_to_rows"] = "group_to_rows",
    ) -> MesoExpr[Intersection[_S_contra, Q], _T]: ...
    @overload
    def over[Q: Shape](
        self,
        *exprs: ExoExpr[_S_contra, Any]
        | ExoExpr[Q, Any]
        | ExoExpr[Intersection[_S_contra, Q], Any],
        order_by: Iterable[
            ExoExpr[_S_contra, Any] | ExoExpr[Q, Any] | ExoExpr[Intersection[_S_contra, Q], Any]
        ]
        | None = None,
        mapping_strategy: Literal["join"],
    ) -> MesoExpr[Intersection[_S_contra, Q], builtins.list[_T]]: ...

    def over(
        self,
        *exprs: ExoExpr,
        order_by: Iterable | None = None,
        mapping_strategy: Literal["group_to_rows", "join"] = "group_to_rows",
    ) -> MesoExpr:
        """
        Specify the expression is for the keyed group of the table. I.e.,

        Specify the expression is for a window (i.e. bucket) of values keyed on by `exprs`.
        "group_to_rows" matches values up to the current rows, "join" *implodes* the group and
        matches this list back to each of the the source rows:

        ```
        # Find accounts with a US-region account with the same username
        accounts.filter(
            Account.region.over(Account.username, mapping_strategy="join")
            .list.contains("US")
        )
        ```
        """
        return IntermediateExpr(
            self.expr.over(
                *(_pl_expr(e) for e in exprs),
                order_by=(_pl_expr(e) for e in order_by) if order_by else None,
                mapping_strategy=mapping_strategy,
            )
        )

    @staticmethod
    def _normalize_mapping[A, B](mapping: Mapping[A, B]) -> Mapping[A, B]:
        # To avoid processing large mappings we don't need to, we can just check one element, since
        # we should assume all element
        if mapping and builtins.any(isinstance(x, enum.Enum) for x in first(mapping.items(), ())):
            return {normalize_enum(k): normalize_enum(v) for k, v in mapping.items()}
        return mapping

    def replace(
        self,
        mapping: Mapping[_T, _T],
        *,
        default: ExoExpr[_S_contra, _T] | _T | None = None,
        or_null: bool = False,
    ) -> Expr[_S_contra, _R_contra, _T]:
        """
        Translate the values in the column using the given lookup table. If the lookup fails,
        this preserves the current value, use `default` or `or_null` to change this behaviour.
        """
        kwargs = _ReplaceKwargs()
        if default is not None:
            kwargs["default"] = _pl_expr(default)
        elif or_null:
            kwargs["default"] = None
        replace = self.expr.replace_strict if "default" in kwargs else self.expr.replace
        mapping = self._normalize_mapping(mapping)

        return IntermediateExpr(replace(mapping, **kwargs))

    def replace_to[Q: Shape, U](
        self,
        mapping: Mapping[_T, U],
        to: BoundDimension[Q, U],
        *,
        default: ExoExpr[_S_contra, U] | U | None = None,
        or_null: bool = False,
    ) -> Expr[_S_contra, Q, U]:
        """
        Translate the values in the column using the given lookup table

        This is the `_to` variant, which is allowed to change type but must be mapped to a new
        column

        Unlike `replace`, all values must be mapped or a default must be set, since the column type
        can change
        """
        kwargs = _ReplaceKwargs()
        if default is not None:
            kwargs["default"] = _pl_expr(default)
        elif or_null:
            kwargs["default"] = None
        mapping = self._normalize_mapping(mapping)

        return IntermediateExpr(
            self.expr.replace_strict(mapping, return_dtype=to.ty.pl_ty, **kwargs).alias(to.name)
        )

    def replace_out[U](
        self,
        mapping: Mapping[_T, U],
        ty: Typeable[U],
        *,
        default: ExoExpr[_S_contra, U] | U | None = None,
        or_null: bool = False,
    ) -> MesoExpr[_S_contra, U]:
        """
        Translate the values in the column using the given lookup table

        This is the `_out` variant, which is allowed to change type but must be mapped to a new
        column with `.to` if it is to be stored in a dataframe

        Unlike `replace`, all values must be mapped or a default must be set, since the column type
        is changing
        """
        kwargs = _ReplaceKwargs()
        if default is not None:
            kwargs["default"] = _pl_expr(default)
        elif or_null:
            kwargs["default"] = None
        mapping = self._normalize_mapping(mapping)

        return IntermediateExpr(
            self.expr.replace_strict(mapping, return_dtype=from_typeable(ty).pl_ty, **kwargs)
        )

    def map_to[Q: Shape, U](
        self, transform: Callable[[_T], U | None], to: BoundDimension[Q, U]
    ) -> Expr[_S_contra, Q, U]:
        """
        Apply a Python transformation to the values in a column. This has to be mapped straight to a
        dimension to know the polars datatype of the output. This limitation shouldn't be too
        disruptive since the `transform` function in Python should be able to get it into its final
        form, and since after being mapped to a column it can continue to be operated on
        """
        return IntermediateExpr(self.expr.map_elements(transform, to.ty.pl_ty).alias(to.name))

    def map_out[U](
        self, transform: Callable[[_T], U | None], ty: Typeable[U]
    ) -> MesoExpr[_S_contra, U]:
        """
        Apply a Python transformation to the values in a column. This has to be mapped straight to a
        dimension to know the polars datatype of the output. This limitation shouldn't be too
        disruptive since the `transform` function in Python should be able to get it into its final
        form, and since after being mapped to a column it can continue to be operated on
        """
        return IntermediateExpr(self.expr.map_elements(transform, from_typeable(ty).pl_ty))

    def cast(self, ty: Typeable[_T], *, strict: bool = True) -> Expr[_S_contra, _R_contra, _T]:
        """
        Cast the values in the column whilst preserving the type, useful if two polars
        representation have the same Python type (e.g. `Float32` and `Float64`)

        This is the `_out` variant, which is allowed to change type but must be mapped to a new
        column if it is to be stored in a dataframe
        """
        return IntermediateExpr(self.expr.cast(from_typeable(ty).pl_ty, strict=strict))

    def cast_out[U](self, ty: Typeable[U], *, strict: bool = True) -> MesoExpr[_S_contra, U]:
        """
        Cast the values in the column to change the type, i.e. casting ints to strings.

        This is the `_out` variant, which is allowed to change type but must be mapped to a new
        column if it is to be stored in a dataframe
        """
        return IntermediateExpr(self.expr.cast(from_typeable(ty).pl_ty, strict=strict))

    def __or__[SA: Shape](
        self: Expr[_S_contra, _R_contra, bool], other: ExoExpr[SA, bool] | bool
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, bool]:
        return IntermediateExpr(self.expr | _pl_expr(other))

    def __and__[SA: Shape](
        self: Expr[_S_contra, _R_contra, bool], other: ExoExpr[SA, bool] | bool
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, bool]:
        return IntermediateExpr(self.expr & _pl_expr(other))

    @overload
    def __add__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def __add__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __add__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...
    @overload
    def __add__[SA: Shape, D: datetime.date | datetime.datetime](
        self: Expr[_S_contra, _R_contra, D],
        other: ExoExpr[SA, datetime.timedelta] | datetime.timedelta,
    ) -> MesoExpr[Intersection[_S_contra, SA], D]: ...

    def __add__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr + _pl_expr(other))

    @overload
    def __sub__[SA: Shape](
        self: Expr[_S_contra, _R_contra, datetime.date | datetime.datetime],
        other: ExoExpr[SA, datetime.date | datetime.datetime] | datetime.date | datetime.datetime,
    ) -> MesoExpr[Intersection[_S_contra, SA], datetime.timedelta]: ...
    @overload
    def __sub__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def __sub__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __sub__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...
    @overload
    def __sub__[SA: Shape, D: datetime.date | datetime.datetime](
        self: Expr[_S_contra, _R_contra, D],
        other: ExoExpr[SA, datetime.timedelta] | datetime.timedelta,
    ) -> MesoExpr[Intersection[_S_contra, SA], D]: ...

    def __sub__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr - _pl_expr(other))

    @overload
    def __mul__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def __mul__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __mul__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...

    def __mul__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr * _pl_expr(other))

    @overload
    def __pow__[SA: Shape](
        self, other: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def __pow__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __pow__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...

    def __pow__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr ** _pl_expr(other))

    @overload
    def __truediv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, N] | N
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __truediv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __truediv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...
    @overload
    def __truediv__[SA: Shape](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, int] | int
    ) -> MesoExpr[Intersection[_S_contra, SA], float]: ...  # Ints give floats when divided

    def __truediv__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr / _pl_expr(other))

    @overload
    def __floordiv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, N] | N
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __floordiv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, N], other: ExoExpr[SA, int] | int
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, N]: ...
    @overload
    def __floordiv__[SA: Shape, N: float | Decimal](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, N] | N
    ) -> MesoExpr[Intersection[_S_contra, SA], N]: ...
    @overload
    def __floordiv__[SA: Shape](
        self: Expr[_S_contra, _R_contra, int], other: ExoExpr[SA, int] | int
    ) -> MesoExpr[Intersection[_S_contra, SA], float]: ...  # Ints give floats when divided

    def __floordiv__(self, other: Expr | object) -> Expr:
        return IntermediateExpr(self.expr // _pl_expr(other))

    def __neg__[N: (float, Decimal, datetime.timedelta, int)](
        self: Expr[_S_contra, _R_contra, N],
    ) -> Expr[_S_contra, _R_contra, N]:
        return IntermediateExpr(-self.expr)

    def __invert__(self: Expr[_S_contra, _R_contra, bool]) -> Expr[_S_contra, _R_contra, bool]:
        return IntermediateExpr(~self.expr)

    def abs[N: float | Decimal | int](
        self: Expr[_S_contra, _R_contra, N],
    ) -> Expr[_S_contra, _R_contra, N]:
        return IntermediateExpr(self.expr.abs())

    def is_unique(self) -> MesoExpr[_S_contra, bool]:
        return IntermediateExpr(self.expr.is_unique())

    def coalesce[SA: Shape](
        self,
        *others: ExoExpr[_S_contra, _T]
        | ExoExpr[SA, _T]
        | ExoExpr[Intersection[_S_contra, SA], _T]
        | _T,
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(pl.coalesce(self.expr, *(_pl_expr(e) for e in others)))

    def null_when[SA: Shape](
        self,
        *conditions: ExoExpr[_S_contra, bool]
        | ExoExpr[SA, bool]
        | ExoExpr[Intersection[_S_contra, SA], bool],
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return when(*conditions).otherwise(self)

    def null_when_eq[Q: Shape](
        self, expr: ExoExpr[Q, _T] | _T
    ) -> Expr[Intersection[_S_contra, Q], _R_contra, _T]:
        """Replace any value equalling `expr` with null. E.g. `.null_when_eq("NOT SET")`"""
        return when(self.eq(expr)).otherwise(self)

    def null_insignificant(self) -> Expr[_S_contra, _R_contra, _T]:
        """Replace 0 and `nan` with null"""
        return when(self.is_significant()).then(self)

    def max_horizontal[SA: Shape](
        self, *others: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(pl.max_horizontal(self.expr, *map(_pl_expr, others)))

    def min_horizontal[SA: Shape](
        self, *others: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(pl.min_horizontal(self.expr, *map(_pl_expr, others)))

    def repeat_by[SA: Shape](
        self, by: ExoExpr[SA, int] | int
    ) -> IntermediateExpr[Intersection[_S_contra, SA], _R_contra, builtins.list[_T]]:
        """
        Create a list of the element repeated `by` times. Also useful for constructing singleton
        lists with `.repeat_by(0)`
        """
        return IntermediateExpr(self.expr.repeat_by(_pl_expr(by)))

    @property
    def struct[M: Shape](
        self: Expr[_S_contra, _R_contra, StructMapping[M]],
    ) -> StructExprNamespace[_S_contra, _R_contra, M]:
        return StructExprNamespace(self)

    @property
    def str(
        self: Expr[_S_contra, _R_contra, builtins.str],
    ) -> StrExprNamespace[_S_contra, _R_contra, builtins.str]:
        return StrExprNamespace(self.cast(STRING))

    @property
    def dt[D: (datetime.datetime, datetime.date)](
        self: Expr[_S_contra, _R_contra, D],
    ) -> DtExprNamespace[_S_contra, _R_contra, D]:
        return DtExprNamespace(self)

    @property
    def list[A](
        self: Expr[_S_contra, _R_contra, builtins.list[A]],
    ) -> ListExprNamespace[_S_contra, _R_contra, A]:
        return ListExprNamespace(self)

    @overload
    def on[S: Shape, Q: Shape](self: ExoExpr[S | Q, Any]) -> JoinOn[S, Q, _T]: ...
    @overload
    def on[Q: Shape](self, other: ExoExpr[Q, _T]) -> JoinOn[_S_contra, Q, _T]: ...

    def on(self, other: ExoExpr | None = None) -> JoinOn:
        return JoinOn(self, other if other is not None else self)


# An expression that is local to a shape
EndoExpr: TypeAlias = Expr[_S_contra, _S_contra, _T]
# An expression where the output shape is irrelevant, useful for when an expression is being
# consumed rather than stored, i.e. in a filter
ExoExpr: TypeAlias = Expr[_S_contra, Never, _T]
# An expression which is intermediary and cannot be stored in a shape, it must be mapped to a new
# column, e.g. using `to`, to be stored
MesoExpr: TypeAlias = Expr[_S_contra, Never, _T]


@dataclasses.dataclass(frozen=True)
class IntermediateExpr[S: Shape, R: Shape, T](Expr[S, R, T]):
    """
    An expression created from another expression, this just stores the polars expression generated
    from whatever operation has been applied to the last expression
    """

    expr: pl.Expr = dataclasses.field()


@dataclasses.dataclass(frozen=True)
class BoundDimension(Expr[_S_contra, _S_contra, _T]):
    """
    This binds the shape to the dimension at the type level, which means this shape can then
    be enforced by all the operations using any dimension, and any expression created from this
    bound dimension can continue to refer to the shape it operates on in its type too by passing
    the type parameter along
    """

    shape: type[_S_contra]
    ty: Type[_T]
    name: str

    @property
    def expr(self) -> pl.Expr:
        return pl.col(self.name)

    def set(self, value: _T) -> Initializer[_S_contra, _T]:
        return Initializer(self, normalize_enum(value))

    def set_or_null(self, value: _T | None) -> Initializer[_S_contra, _T]:
        return Initializer(self, normalize_enum(value))

    def set_all(self, values: Iterable[_T]) -> ColumnInitializer[_S_contra, _T]:
        return ColumnInitializer(self, list(map(normalize_enum, values)))

    def set_or_null_all(self, values: Iterable[_T | None]) -> ColumnInitializer[_S_contra, _T]:
        return ColumnInitializer(self, list(map(normalize_enum, values)))

    def map(self, transform: Callable[[_T], _T]) -> EndoExpr[_S_contra, _T]:
        """
        Apply a Python transformation to the values in a column. This is defined on the dimension
        rather than on `Expr`, to know the polars datatype of the output. To change the type, use
        `Expr.map_to`
        """
        return IntermediateExpr(self.expr.map_elements(transform, return_dtype=self.ty.pl_ty))

    def null(self) -> Expr[Any, _S_contra, _T]:
        null: MesoExpr[Any, _T] = lit(None)
        return null.to(self)


@dataclasses.dataclass(frozen=True)
class Initializer(Expr[Any, _S_contra, _T]):
    """
    Used in `Entry.of` to allow constructing rows where the dimension matches the assigned column
    value
    """

    dimension: BoundDimension[_S_contra, _T]
    value: _T | None

    @property
    def expr(self) -> pl.Expr:
        return Expr.lit(self.value).to(self.dimension).expr


@dataclasses.dataclass(frozen=True)
class ColumnInitializer(Generic[_S_contra, _T]):
    """Used in dataframe constructors to initialize a dataframe column-wise"""

    dimension: BoundDimension[_S_contra, _T]
    value: list[_T | None]


@dataclasses.dataclass(frozen=True)
class When(Generic[_S_contra]):
    """
    A condition that can be combined with a value using `then` and `otherwise` to construct a
    conditional expression. Use `tp.when` rather than `When(...)` directly

    ```
    tp.when(some_condition).then(if_true).otherwise(if_false)
    ```
    """

    conditions: Collection[ExoExpr[_S_contra, bool]]

    @overload
    def then[SA: Shape, R: Shape, T](
        self, then: Expr[SA, R, T]
    ) -> PartialConditional[Intersection[_S_contra, SA], R, T]: ...
    @overload
    def then[T: Value](self, then: T) -> PartialConditional[_S_contra, Never, T]: ...

    def then[SA: Shape, R: Shape, T](
        self, then: Expr | T
    ) -> PartialConditional[Intersection[_S_contra, SA], R, T]:
        return PartialConditional(
            pl.when(c.expr for c in self.conditions).then(_expr_or_lit(then).expr)
        )

    @overload
    def otherwise[SA: Shape, R: Shape, T](
        self, otherwise: Expr[SA, R, T]
    ) -> PartialConditional[Intersection[_S_contra, SA], R, T]: ...
    @overload
    def otherwise[T: Value](self, otherwise: T) -> PartialConditional[_S_contra, Never, T]: ...

    def otherwise[SA: Shape, R: Shape, T](
        self, otherwise: Expr | T
    ) -> PartialConditional[Intersection[_S_contra, SA], R, T]:
        return PartialConditional(
            pl.when((~all_horizontal(self.conditions)).expr).then(_expr_or_lit(otherwise).expr)
        )


@dataclasses.dataclass(frozen=True)
class PartialConditional(Expr[_S_contra, _R_contra, _T]):
    """
    The intermediate state where one outcome value has been provided but not the other, which
    is assumed by default to be null. Use `.otherwise` to provide the other value, or `.when`
    again to construct an if/elif chain
    """

    expr: polars.expr.whenthen.Then | polars.expr.whenthen.ChainedThen

    def when[SA: Shape](
        self, *conditions: ExoExpr[SA, bool]
    ) -> ChainedWhen[Intersection[_S_contra, SA], _R_contra, _T]:
        return ChainedWhen(self.expr.when(c.expr for c in conditions))

    def otherwise[SA: Shape](
        self, otherwise: ExoExpr[SA, _T] | _T
    ) -> Expr[Intersection[_S_contra, SA], _R_contra, _T]:
        return IntermediateExpr(self.expr.otherwise(_pl_expr(otherwise)))


@dataclasses.dataclass(frozen=True)
class ChainedWhen(Generic[_S_contra, _R_contra, _T]):
    """
    A chain of when statements representing an if/elif chain. Construct by starting with a
    `tp.when(conds).then(if_true)`, and adding more `.when(else_cond).then(else_true)`s after
    """

    expr: polars.expr.whenthen.ChainedWhen

    @overload
    def then[SA: Shape](
        self, then: ExoExpr[SA, _T]
    ) -> PartialConditional[Intersection[_S_contra, SA], _R_contra, _T]: ...
    @overload
    def then(self, then: _T) -> PartialConditional[_S_contra, _R_contra, _T]: ...

    def then[SA: Shape](
        self, then: Expr | _T
    ) -> PartialConditional[Intersection[_S_contra, SA], _R_contra, _T]:
        return PartialConditional(self.expr.then(_expr_or_lit(then).expr))


def when[S: Shape](*conditions: ExoExpr[S, bool]) -> When[S]:
    return When(conditions)


def any_horizontal[S: Shape, R: Shape](
    *conditions: Expr[S, R, bool] | Iterable[Expr[S, R, bool]],
) -> Expr[S, R, bool]:
    """`or` all the given conditions, i.e. when any is true"""
    return IntermediateExpr(
        pl.any_horizontal(
            *(map(_pl_expr, c) if isinstance(c, Iterable) else _pl_expr(c) for c in conditions)
        )
    )


def all_horizontal[S: Shape, R: Shape](
    *conditions: Expr[S, R, bool] | Iterable[Expr[S, R, bool]],
) -> Expr[S, R, bool]:
    """`and` all the given conditions, i.e. when all is true"""
    return IntermediateExpr(
        pl.all_horizontal(
            *(map(_pl_expr, c) if isinstance(c, Iterable) else _pl_expr(c) for c in conditions)
        )
    )


type Orderable = int | float | datetime.date | datetime.datetime | Decimal | str


def min_horizontal[S: Shape, R: Shape, T: Orderable](
    *exprs: Expr[S, R, T] | T | Iterable[Expr[S, R, T] | T],
) -> Expr[S, R, T]:
    """Min the given exprs, i.e. take the smallest. For `pl.min`, use Expr.min"""
    return IntermediateExpr(
        pl.min_horizontal(
            *(map(_pl_expr, c) if isinstance(c, Iterable) else _pl_expr(c) for c in exprs)
        )
    )


def max_horizontal[S: Shape, R: Shape, T: Orderable](
    *exprs: Expr[S, R, T] | T | Iterable[Expr[S, R, T] | T],
) -> Expr[S, R, T]:
    """Max the given exprs, i.e. take the largest. For `pl.max`, use Expr.max"""
    return IntermediateExpr(
        pl.max_horizontal(
            *(map(_pl_expr, c) if isinstance(c, Iterable) else _pl_expr(c) for c in exprs)
        )
    )


def length() -> MesoAggExpr[Any, int]:
    """
    Count the number of rows in a shape or window. This is namespaced under `Expr` to avoid
    conflicts with the len builtin
    """
    return AggExpr(pl.len())


def int_range[S: Shape, Q: Shape](
    value: ExoAggExpr[S, int] | ExoAggExpr[Intersection[S, Q], int] | int,
    end: ExoAggExpr[Q, int] | ExoAggExpr[Intersection[S, Q], int] | int | None = None,
    step: int = 1,
) -> MesoExpr[S, int]:
    """
    Construct a series from start inclusive to end exclusive. If end is unspecified, `value` is end,
    otherwise `value` is start
    """
    return IntermediateExpr(pl.int_range(_pl_expr(value), _pl_expr_or_none(end), step))


def row_index() -> MesoExpr[Any, int]:
    """
    Row number of the current line of the frame or window, starting at 0

    Use this and a transform, i.e. `.transform` or `.with_columns`, if you're looking for
    `frame.with_row_index()`:

    ```
    # Polars equivalent: frame.with_row_index("line_number")
    frame.with_column(tp.row_index().to(Report.line_number))
    ```
    """
    return int_range(length())


RangeClosed = Literal["left", "right", "both", "none"]


def date_range[S: Shape](
    start: ExoExpr[S, datetime.date] | datetime.date,
    end: ExoExpr[S, datetime.date],
    interval: str | datetime.timedelta = "1d",
    closed: RangeClosed = "both",
) -> MesoExpr[S, datetime.date]:
    """Construct a series from start inclusive to end inclusive"""
    return IntermediateExpr(pl.date_range(_pl_expr(start), _pl_expr(end), interval, closed=closed))


def datetime_range[S: Shape](
    start: ExoExpr[S, datetime.datetime] | datetime.datetime,
    end: ExoExpr[S, datetime.datetime] | datetime.date,
    interval: str | datetime.timedelta,
    closed: RangeClosed = "both",
) -> MesoExpr[S, datetime.date]:
    """Construct a series from start inclusive to end inclusive"""
    return IntermediateExpr(
        pl.datetime_range(_pl_expr(start), _pl_expr(end), interval, closed=closed)
    )


def date_ranges[S: Shape](
    start: ExoExpr[S, datetime.date] | datetime.date, end: ExoExpr[S, datetime.date]
) -> MesoExpr[S, list[datetime.date]]:
    """Construct a list for each element containing a date range from start to end inclusive"""
    return IntermediateExpr(pl.date_ranges(_pl_expr(start), _pl_expr(end)))


def lit[T](value: T | None) -> Expr[Shape, Never, T]:
    """Create a literal value that can be mapped to a column"""
    return Expr.lit(value)


def null[T](ty: Typeable[T]) -> Expr[Shape, Never, T]:
    """Create a literal value that can be mapped to a column, or be unset"""
    return lit(None)


NoShape: TypeAlias = Never


class Element[T](Shape):
    """
    Special shape containing `pl.element()` for mapping single element expressions, such as
    `list.eval`
    """

    @classmethod
    def element(cls) -> EndoExpr[Element[T], T]:
        return IntermediateExpr(pl.element())


def element[T]() -> EndoExpr[Element[T], T]:
    # Define this separately so type inference for tp.element isn't assumed to be `Any`
    return Element[T].element()


class Suffixed[S: Shape](Shape):
    """
    A suffixed shape allows modifying a shape with an additional tag after each column name.
    This is critical in joins to avoid name conflicts, if two similarly named columns would
    otherwise collide with each other. Use `df.suffix()` to conveniently add a suffix to an existing
    dataframe.

    To access dimensions of a suffixed shape, use the shape to transform the base shape, i.e.

    ```
    my_shape = tp.DataFrame(...)
    suffixed = my_shape.suffix()
    col_a = suffixed[suffixed.s(my_shape.s.a)]
    ```

    The above is a little clunky to use; suffixed shapes are only intended as brief intermediaries
    when name conflicts are possible
    """

    shape: ClassVar[type[S]]  # type: ignore
    suffix: ClassVar[str]

    @classmethod
    def mapping_to(cls) -> dict[str, str]:
        return {s.name: f"{s.name}{cls.suffix}" for s in cls.shape.shape_meta().dimensions}

    def __new__[T](cls, dim: BoundDimension[S, T]) -> BoundDimension[Self, T]:
        return BoundDimension[Any, T](dim.shape, dim.ty, dim.name + cls.suffix)

    @classmethod
    def __call__[T](cls, dim: BoundDimension[S, T]) -> BoundDimension[Self, T]:
        return cls.__new__(dim)  # ty: ignore[missing-argument]

    @classmethod
    def shape_meta(cls) -> ShapeMeta:
        return SuffixedShapeMeta(cls) if cls.__bases__ == (Suffixed,) else super().shape_meta()


@dataclasses.dataclass(frozen=True)
class SuffixedShapeMeta[S: Shape](ShapeMeta[Suffixed[S]]):
    """Similar to `ShapeMeta`, but handle renaming the dimensions with the suffix"""

    shape: type[Suffixed[S]]

    @property
    def dimensions(self) -> Iterable[BoundDimension[Suffixed, Any]]:
        """Iterate through the dimensions defined in the shape"""
        for dimension in self.shape.shape.shape_meta().dimensions:
            yield dataclasses.replace(
                dimension, name=dimension.name + self.shape.suffix, shape=self.shape
            )


def suffix[S: Shape](shape: type[S], suffix: str | None = None) -> type[Suffixed[S]]:
    """
    Create a modified shape where each column name is suffixed:

    ```
    suffixed = external_accounts.suffix(suff := suffix(ExternalAccount))
    account_details = accounts.join(
        suffixed,
        Accounts.external_account_number.on(suff(ExternalAccounts.number))
    ).transform(
        # Could not do this otherwise, as the Accounts.name and ExternalAccounts.name columns would
        # conflict and Polars wouldn't be able to tell them apart (in regular Polars you'd also have
        # to be explicit)
        (Accounts.name + "-" + suff(ExternalAccounts.name)).to(AccountMatchup.name)
    )
    ```

    Parameters
    ----------
    suffix : str | None
        The string literal to append to each column name. If `None`, this will be the name of the
        shape itself
    """
    with_shape, with_suffix = shape, suffix

    class SuffixedShape(Suffixed):
        shape = with_shape
        suffix = with_suffix or shape.__qualname__

    return SuffixedShape


_SProjection_contra = TypeVar("_SProjection_contra", bound="Shape", contravariant=True)


@dataclasses.dataclass(frozen=True)
class Projection(Generic[_SProjection_contra]):
    """
    Represent a projection of a potentially wider shape onto just this shape. This is useful for
    constructing a struct out of a wider shape
    """

    shape: type[_SProjection_contra]

    def struct(self) -> MesoExpr[_SProjection_contra, StructMapping[_SProjection_contra]]:
        return IntermediateExpr(pl.struct(self.shape.shape_meta().datatypes.keys()))


def projection[S: Shape](shape: type[S]) -> Projection[S]:
    """
    Construct a projection of a shape out of a potentially wider shaped dataframe

    ```
    accounts = tp.DataFrame(Account, ...)
    external_email = accounts.with_columns(
        tp.projection(e := EmailDetails).struct()
            .struct.map_rows_to(e.email + "@" + e.organization + ".com", e.email)
    )
    ```
    """
    return Projection(shape)


def arg_sort_by[S: Shape](
    *exprs: ExoExpr[S, Any],
    descending: bool | Sequence[bool] = False,
    nulls_last: bool = False,
    maintain_order: bool = False,
) -> MesoExpr[S, int]:
    return IntermediateExpr(
        pl.arg_sort_by(
            map(_pl_expr, exprs),
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=maintain_order,
        )
    )


def duration[S: Shape](
    weeks: int | ExoExpr[S, int] | None = None,
    days: int | ExoExpr[S, int] | None = None,
    minutes: int | ExoExpr[S, int] | None = None,
    seconds: int | ExoExpr[S, int] | None = None,
    milliseconds: int | ExoExpr[S, int] | None = None,
    microseconds: int | ExoExpr[S, int] | None = None,
    nanoseconds: int | ExoExpr[S, int] | None = None,
) -> Expr[S, Never, datetime.timedelta]:
    """
    Construct a duration, either from literals of column values

    ```
    weeks = tp.duration(weeks=AccountingPeriod.week_count)
    adjusted = Rate.date + tp.duration(days=Rate.adjustment_days, seconds=10)
    ```
    """
    return IntermediateExpr(
        pl.duration(
            weeks=_pl_expr_or_none(weeks),
            days=_pl_expr_or_none(days),
            minutes=_pl_expr_or_none(minutes),
            seconds=_pl_expr_or_none(seconds),
            milliseconds=_pl_expr_or_none(milliseconds),
            microseconds=_pl_expr_or_none(microseconds),
            nanoseconds=_pl_expr_or_none(nanoseconds),
        )
    )


def date[S: Shape](
    year: int | ExoExpr[S, int], month: int | ExoExpr[S, int], day: int | ExoExpr[S, int]
) -> Expr[S, Never, datetime.date]:
    return IntermediateExpr(pl.date(_pl_expr(year), _pl_expr(month), _pl_expr(day)))


def concat_list[S: Shape, A](
    *exprs: ExoExpr[S, list[A]] | Sequence[A | ExoExpr[S, A]],
) -> MesoExpr[S, list[A]]:
    """
    Combine various list expressions into a single list. Also useful for constructing lists,
    with `tp.concat_list([expr1, expr2])`
    """
    return IntermediateExpr(
        pl.concat_list(
            *(
                _pl_expr(expr) if isinstance(expr, Expr) else [_pl_expr(e) for e in expr]
                for expr in exprs
            )
        )
    )


def struct[S: Shape, M: Shape](*exprs: Expr[S, M, Any]) -> MesoExpr[S, StructMapping[M]]:
    """
    Construct a struct expression from the underlying expressions:

    ```
    tp.struct(Account.name.to(Login.account), tp.lit(0).to(Login.attempts))
    ```

    The above will create an expression that can be put into a column for
    `tp.dimension(tp.struct_of(Login))`, or mapped with `.struct.map_rows_to(..., to)
    """
    return IntermediateExpr(pl.struct(map(_pl_expr, exprs)))


@dataclasses.dataclass
class JoinOn(Generic[_S_contra, _R_contra, _T]):
    """Represents a requirement for `left` and `right` to be equal for two rows to join"""

    left: ExoExpr[_S_contra, _T]
    right: ExoExpr[_R_contra, _T]
