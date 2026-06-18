from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import polars as pl

from typol.types import EnumOf, Typeable, from_typeable

if TYPE_CHECKING:
    from typol.expr import BoundDimension, Shape


@dataclasses.dataclass(frozen=True)
class Series[T]:
    """Represents the values of a column, all of which have type `T`"""

    data: pl.Series

    def to[S: Shape](self, dimension: BoundDimension[S, T]) -> BoundSeries[S, T]:
        return BoundSeries(self.data.alias(dimension.name).cast(dimension.ty.pl_ty))

    @overload
    def first[V: EnumOf[int]](self: Series[V]) -> int | None: ...
    @overload
    def first[V: EnumOf[str]](self: Series[V]) -> str | None: ...
    @overload
    def first(self) -> T | None: ...

    def first(self) -> object | None:
        return self.data.first()

    @overload
    def last[V: EnumOf[int]](self: Series[V]) -> int | None: ...
    @overload
    def last[V: EnumOf[str]](self: Series[V]) -> str | None: ...
    @overload
    def last(self) -> T | None: ...

    def last(self) -> object | None:
        return self.data.last()

    def count(self) -> int:
        return self.data.count()

    def sum[N: (int, float)](self: Series[N]) -> N:
        return cast(N, self.data.sum())

    @overload
    def max[V: EnumOf[int]](self: Series[V]) -> int | None: ...
    @overload
    def max[V: EnumOf[str]](self: Series[V]) -> str | None: ...
    @overload
    def max(self) -> T | None: ...

    def max(self) -> T | None:
        return cast(T, self.data.max())

    @overload
    def min[V: EnumOf[int]](self: Series[V]) -> int | None: ...
    @overload
    def min[V: EnumOf[str]](self: Series[V]) -> str | None: ...
    @overload
    def min(self) -> T | None: ...

    def min(self) -> T | None:
        return cast(T, self.data.min())

    def unique(self) -> Series[T]:
        return Series(self.data.unique())

    def drop_nulls(self) -> Series[T]:
        return Series(self.data.drop_nulls())

    def head(self, n: int) -> Series[T]:
        return Series(self.data.head(n))

    def cast[U](self, ty: Typeable[U]) -> Series[U]:
        return Series(self.data.cast(from_typeable(ty).pl_ty))

    @overload
    def fill_null(self, fill: T) -> Series[T]: ...
    @overload
    def fill_null(
        self, *, strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ) -> Series[T]: ...

    def fill_null(
        self,
        fill: T | None = None,
        strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"] | None = None,
    ) -> Series[T]:
        return Series(self.data.fill_null(fill, strategy))

    def fill_nan[N: (int, float)](self: Series[N], fill: N | None) -> Series[N]:
        return Series(self.data.fill_nan(fill))

    def reverse(self) -> Series[T]:
        return Series(self.data.reverse())

    @overload
    def to_list[V: EnumOf[str]](self: Iterable[EnumOf[V]]) -> list[V]: ...
    @overload
    def to_list[V: EnumOf[int]](self: Iterable[EnumOf[V]]) -> list[V]: ...
    @overload
    def to_list(self) -> list[T]: ...

    def to_list(self) -> list:
        return self.data.to_list()

    @overload
    def __iter__[V: EnumOf[str]](self: Series[V]) -> Iterator[str]: ...
    @overload
    def __iter__[V: EnumOf[int]](self: Series[V]) -> Iterator[int]: ...
    @overload
    def __iter__(self) -> Iterator[T]: ...

    def __iter__(self) -> Iterator:
        """Iterate through all the values of the series"""
        return self.data.__iter__()

    def is_empty(self) -> bool:
        return self.data.is_empty()

    def __len__(self) -> int:
        """The number of values in the series"""
        return len(self.data)

    def any(self) -> bool:
        """Return whether any of the values in the column are `True`"""
        return self.data.any()


@dataclasses.dataclass(frozen=True)
class LazySeries[T]:
    """
    Represents the values of a column, all of which have type `T`, for a lazy frame (not yet
    realized). Typol needs this since it's annoying to have to create an intermediate shape for a
    single column of a lazy frame like Polars does, so we want to be able to represent these ad-hoc
    frames succinctly
    """

    data: pl.LazyFrame

    @overload
    def first[V: EnumOf[str]](self: LazySeries[V]) -> str | None: ...
    @overload
    def first[V: EnumOf[int]](self: LazySeries[V]) -> int | None: ...
    @overload
    def first(self) -> T | None: ...

    def first(self) -> object | None:
        return _collect_val(self.data.first())

    @overload
    def last[V: EnumOf[str]](self: LazySeries[V]) -> str | None: ...
    @overload
    def last[V: EnumOf[int]](self: LazySeries[V]) -> int | None: ...
    @overload
    def last(self) -> T | None: ...

    def last(self) -> object | None:
        return _collect_val(self.data.last())

    def count(self) -> int:
        return _collect_val(self.data.count()) or 0

    def sum[N: (int, float)](self: LazySeries[N]) -> N:
        return _collect_val(self.data.sum()) or 0

    def max(self) -> T | None:
        return _collect_val(self.data.max())

    def min(self) -> T | None:
        return _collect_val(self.data.min())

    def unique(self) -> LazySeries[T]:
        return LazySeries(self.data.unique())

    def drop_nulls(self) -> LazySeries[T]:
        return LazySeries(self.data.drop_nulls())

    def head(self, n: int) -> LazySeries[T]:
        return LazySeries(self.data.head(n))

    def cast[U](self, ty: Typeable[U]) -> LazySeries[U]:
        return LazySeries(self.data.select(pl.col("series").cast(from_typeable(ty).pl_ty)))

    @overload
    def fill_null(self, fill: T) -> LazySeries[T]: ...
    @overload
    def fill_null(
        self, *, strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ) -> LazySeries[T]: ...

    def fill_null(
        self,
        fill: T | None = None,
        strategy: Literal["forward", "backward", "min", "max", "mean", "zero", "one"] | None = None,
    ) -> LazySeries[T]:
        return LazySeries(self.data.fill_null(fill, strategy))

    def fill_nan[N: (int, float)](self: LazySeries[N], fill: N | None) -> LazySeries[N]:
        return LazySeries(self.data.fill_nan(fill))

    def reverse(self) -> LazySeries[T]:
        return LazySeries(self.data.reverse())

    @overload
    def to_list[V: EnumOf[str]](self: LazySeries[V]) -> list[str]: ...
    @overload
    def to_list[V: EnumOf[int]](self: LazySeries[V]) -> list[int]: ...
    @overload
    def to_list(self) -> list[T]: ...

    def to_list(self) -> list:
        return _collect(self.data).to_list()

    def is_empty(self) -> bool:
        return _collect(self.data.limit(1)).is_empty()

    def any(self) -> bool:
        """Return whether any of the values in the column are `True`"""
        return not _collect(self.data.filter(pl.col("series")).limit(1)).is_empty()

    def collect(self) -> Series[T]:
        return Series(_collect(self.data))


@dataclasses.dataclass(frozen=True)
class BoundSeries[S: Shape, T]:
    """A series bound to a name that can be inserted into shape `S`"""

    expr: pl.Series


def _collect(lf: pl.LazyFrame) -> pl.Series:
    return lf.collect()["series"]


def _collect_val(lf: pl.LazyFrame) -> Any | None:  # noqa: ANN401
    return lf.collect()["series"].first()
