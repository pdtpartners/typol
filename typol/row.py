from __future__ import annotations

import dataclasses
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, cast, overload

from more_itertools import first

from typol.expr import BoundDimension, Initializer, Shape, StructMapping
from typol.types import EnumOf


def _filter_nones(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    if None in mapping.values():
        return {k: v for k, v in mapping.items() if v is not None}
    return mapping


@dataclasses.dataclass(frozen=True)
class Row[S: Shape](Mapping[str, Any]):
    """
    Represents a single entry across columns in a `DataFrame`. Use `Entry.of` for
    constructing `Row`s in a somewhat type-safe manner

    Unlike full `Entry`s, `Row`s can be built from a `DataFrame` with `iter_rows()` since they don't
    require a reified shape; in essence `Row`s come from `DataFrame`s, whereas `Entry`s can be put
    into `DataFrames`
    """

    mapping: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Ensure only present columns are set in the mapping"""
        if None in self.mapping.values():
            object.__setattr__(
                self, "mapping", {k: v for k, v in self.mapping.items() if v is not None}
            )

    @classmethod
    def from_struct_mapping(cls, mapping: StructMapping[S]) -> Row[S]:
        return cls(cast(Mapping, mapping))

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        object.__setattr__(self, "mapping", _filter_nones(mapping))

    def __iter__(self) -> Iterator[str]:
        """Iterate through the present columns"""
        return iter(self.keys())

    def __len__(self) -> int:
        """The number of present columns"""
        return len(self.mapping)

    @overload
    def __getitem__[V: EnumOf[str]](self, dim: BoundDimension[S, list[V]]) -> list[str]: ...
    @overload
    def __getitem__[V: EnumOf[int]](self, dim: BoundDimension[S, list[V]]) -> list[int]: ...
    @overload
    def __getitem__[V: EnumOf[str]](self, dim: BoundDimension[S, V]) -> str: ...
    @overload
    def __getitem__[V: EnumOf[int]](self, dim: BoundDimension[S, V]) -> int: ...
    @overload
    def __getitem__[T](self, dim: BoundDimension[S, T] | str) -> T: ...

    def __getitem__(self, dim: BoundDimension[S, Any] | str) -> object:
        """
        Retrieve column from row. This will raise a `KeyError` if the column is not set (i.e. null)
        for this row. To get a `None` value, use `.get(...)`
        """
        name = dim if isinstance(dim, str) else dim.name
        if (value := self.mapping[name]) is None:
            raise KeyError(f"{name!r} is null for row {self}")
        return value

    @overload
    def get[V: EnumOf[str]](self, dim: BoundDimension[S, V]) -> str | None: ...
    @overload
    def get[V: EnumOf[str], T](self, dim: BoundDimension[S, V], default: T) -> str | T: ...
    @overload
    def get[V: EnumOf[int]](self, dim: BoundDimension[S, V]) -> int | None: ...
    @overload
    def get[V: EnumOf[int], T](self, dim: BoundDimension[S, V], default: T) -> int | T: ...
    @overload
    def get[T](self, dim: BoundDimension[S, T] | str) -> T | None: ...
    @overload
    def get[T, U](self, dim: BoundDimension[S, T] | str, default: U) -> T | U: ...

    def get[U](self, dim: BoundDimension[S, Any] | str, *args: U, **kwargs: U) -> object | U:
        """
        Retrieve column from row, returning `None` or `default=` if the column is unset (i.e. null)
        for this row. To avoid handling `None`s, use `row[Shape.col]`, although note that will raise
        an error if the column is unset
        """
        name = dim if isinstance(dim, str) else dim.name
        return super().get(name, *args, **kwargs)

    def items(self) -> ItemsView[str, Any]:
        return self.mapping.items()

    def values(self) -> ValuesView[Any]:
        return self.mapping.values()

    def keys(self) -> KeysView[str]:
        return self.mapping.keys()

    def entry(self, shape: type[S]) -> Entry[S]:
        """Create an `Entry` with reified (run-time) shape information for construction of a dataframe"""
        return Entry(self.mapping, shape)


@dataclasses.dataclass(frozen=True)
class Entry[S: Shape](Row[S]):
    """
    A row with shape information, useful for initializing a dataframe with an explicit schema
    rather than just representing an existing row. Use `Entry.of` for constructing DataFrames out of
    Python values in a somewhat type-safe manner

    Named `Entry` as it's something you're entering into a dataframe than just retrieving
    """

    # Entries are rows that reify their shape, so dataframes built from them can know their schema
    shape: type[S]

    @staticmethod
    def of[T: Shape](*initializers: Initializer[T, Any], partial: bool = False) -> Entry[T]:
        """
        Construct a row out of type checked `Intializer`s for a shape:

        ```python
        Entry.of(  # creates a Entry[Account]
            Account.name.set(account_name),  # ensures `account_name: str`
            Account.code.set(account_code),  # ensures `account_code: int`
        )
        ```
        """
        if not (head := first(initializers, None)):
            raise ValueError("Columns must be provided to initialize a row")
        shape = head.dimension.shape
        if any(i.dimension.shape != shape for i in initializers):
            shapes = {i.dimension.shape for i in initializers}
            raise TypeError(f"Different shapes used to intialize a row, found {shapes}")
        row = {i.dimension.name: i.value for i in initializers}
        if not partial and row.keys() != shape.shape_meta().datatypes.keys():
            missing = shape.shape_meta().datatypes.keys() - row.keys()
            raise AttributeError(
                f"Not all dimensions provided to intialise shape, missing {missing}"
            )
        return Entry(row, head.dimension.shape)

    @classmethod
    def partial[T: Shape](cls, *initializers: Initializer[T, Any]) -> Entry[T]:
        """
        Construct a row out of type checked `Intializer`s for a shape:

        ```python
        Entry.partial(  # creates a Entry[Account]
            Account.name.set(account_name),  # ensures `account_name: str`
            Account.code.set(account_code),  # ensures `account_code: int`
        )
        ```

        This allows some columns to be left unset
        """
        return cls.of(*initializers, partial=True)
