from __future__ import annotations

import dataclasses
import datetime as _datetime
from collections.abc import Collection, ItemsView, KeysView, ValuesView
from decimal import Decimal
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    cast,
    overload,
    runtime_checkable,
)

import more_itertools
import polars as pl
from polars.datatypes.classes import DataTypeClass as _PlDataTypeClass
from polars.datatypes.classes import IntegerType as _PlIntegerType
from polars.datatypes.classes import UnsignedIntegerType as _PlUnsignedIntegerType

if TYPE_CHECKING:
    from typol.expr import Expr, Intersection, NoShape, Shape


@dataclasses.dataclass(frozen=True)
class Type[T]:
    """
    A representation of a Polars type for Typol that allows it to track the associated Python type.
    This is tracked as a type parameter to allow constructing typed expressions, but also reified
    to allow any transformations that might be needed (such as with enums)
    """

    ty: type[T]
    pl_ty: pl.DataType | _PlDataTypeClass

    def __call__(self, value: T) -> Expr[Any, NoShape, T]:
        """Create a literal value cast into this type, i.e. `FLOAT_32(5)` will be a f32"""
        # Avoid rescursive imports by lazily importing expr
        from typol.expr import lit  # noqa: PLC0415,RUF100

        return lit(value).cast(self)


type Typeable[T] = Type[T] | type[T]


def from_typeable[T](t: Typeable[T]) -> Type[T]:
    """
    Normalize anything allowed to be the type of a column, either a Python or Typol type, into a
    Typol type
    """
    if isinstance(t, Type):
        return t
    if issubclass(t, Enum):
        # ty can't see that this is `T`, only that it is `Enum`
        return cast(Type[T], enum(t))
    return Type(t, pl.DataType.from_python(t))


BOOLEAN: Final = Type[bool](bool, pl.Boolean)

INT_8: Final = Type[int](int, pl.Int8)
INT_16: Final = Type[int](int, pl.Int16)
INT_32: Final = Type[int](int, pl.Int32)
INT_64: Final = Type[int](int, pl.Int64)

UINT_8: Final = Type[int](int, pl.UInt8)
UINT_16: Final = Type[int](int, pl.UInt16)
UINT_32: Final = Type[int](int, pl.UInt32)
UINT_64: Final = Type[int](int, pl.UInt64)

FLOAT_32: Final = Type[float](float, pl.Float32)
FLOAT_64: Final = Type[float](float, pl.Float64)

STRING: Final = Type[str](str, pl.String)
CATEGORICAL: Final = Type[str](str, pl.String)

DATE: Final = Type[_datetime.date](_datetime.date, pl.Date)
DATETIME: Final = Type[_datetime.datetime](_datetime.datetime, pl.Datetime())
TIME: Final = Type[_datetime.time](_datetime.time, pl.Time)
DURATION: Final = Type[_datetime.timedelta](_datetime.timedelta, pl.Duration)

TimeUnit: TypeAlias = Literal["ns", "us", "ms"]


def datetime(unit: TimeUnit = "ms", time_zone: str | None = None) -> Type[_datetime.datetime]:
    """Construct a datetime type"""
    return Type(_datetime.datetime, pl.Datetime(unit, time_zone))


def list_of[T](ty: Typeable[T]) -> Type[list[T]]:
    """Construct a list type whose elements are `ty`"""
    return Type(list[T], pl.List(from_typeable(ty).pl_ty))


class StructMapping[M: Shape](Collection[Any], Protocol):
    """
    A tag type for a column containing struct objects. This means accesses of these column values
    will be able to see them as `Mapping` objects, but the field names are tracked too, so
    `expr.struct.field` accesses can be checked against the shape
    """

    @overload
    def get(self, key: str, /) -> Any | None: ...  # noqa: ANN401
    @overload
    def get[D](self, key: str, /, default: D) -> Any | D: ...  # noqa: ANN401

    def __getitem__(self, key: str) -> Any: ...  # noqa: ANN401

    def keys(self) -> KeysView[str]: ...

    def values(self) -> ValuesView[Any]: ...

    def items(self) -> ItemsView[str, Any]: ...


def struct_of[M: Shape](ty: type[M]) -> Type[StructMapping[M]]:
    """Construct a `pl.Struct` type with fields matching shape `M`"""
    return Type(StructMapping[M], pl.Struct(ty.shape_meta().schema))


class DecimalType(Type[Decimal]):
    """A Typol `Type` with specializations for handling `pl.Decimal`"""

    pl_ty: pl.Decimal

    def __call__(self, value: Decimal | int) -> Expr[Shape, NoShape, Decimal]:
        """
        Create a literal value cast into this type, i.e. `decimal(38, 8)(5)` will be a `pl.Decimal`

        To handle some known issues with Polars, this is overriden to explicitly cast to a Python
        decimal first
        """
        return super().__call__(cast(Decimal, value))


def decimal(precision: int | None = None, scale: int = 0) -> DecimalType:
    """Construct a decimal type of given `precision` and `scale`"""
    return DecimalType(Decimal, pl.Decimal(precision, scale))


@runtime_checkable
class _EnumTypeConnection[T](Protocol):
    def __init__(self, value: T, /) -> Self: ...


type EnumOf[T] = Intersection[Enum, _EnumTypeConnection[T]]


def normalize_enum[T](value: T | EnumOf[T]) -> T:
    """Normalize input values so enums are represented by their value rather than as `object`s"""
    match value:
        case Collection() if isinstance(more_itertools.first(value, None), Enum):
            return [x.value for x in value]  # type: ignore
        case Enum(value=v):
            return v
        case _:
            return value


def enum[E: Enum](enum: type[E]) -> Type[E]:
    """Construct a Polars `Enum` around `int` or `str`-backed enum type `enum`"""
    base_type = more_itertools.one(frozenset(type(v.value) for v in enum._member_map_.values()))
    if base_type is str:
        values = [v.value for v in enum._member_map_.values()]
        pl_ty = pl.Enum(values)
    elif base_type is int:
        min_value = min(v.value for v in enum._member_map_.values())
        max_value = max(v.value for v in enum._member_map_.values())
        pl_ty = integer_for(min_value, max_value)
    else:
        pl_ty = pl.DataType.from_python(base_type)

    return Type(enum, pl_ty)


def integer_for(min_value: int, max_value: int) -> type[_PlIntegerType]:
    """Find a signed integer type that fits `-min_value` though `max_value`"""
    if min_value > 0:
        return unsigned_integer_for(max_value)
    elif max_value < 1 << 7 and min_value >= -1 << 7:
        return pl.Int8
    elif max_value < 1 << 15 and min_value >= -1 << 15:
        return pl.Int16
    elif max_value < 1 << 31 and min_value >= -1 << 31:
        return pl.Int32
    else:
        return pl.Int64


def unsigned_integer_for(max_value: int) -> type[_PlUnsignedIntegerType]:
    """Find an unsigned integer type that fits 0 though `max_value`"""
    if max_value < 1 << 8:
        return pl.UInt8
    elif max_value < 1 << 16:
        return pl.UInt16
    elif max_value < 1 << 32:
        return pl.UInt32
    else:
        return pl.UInt64
