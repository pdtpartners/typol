import math
from typing import TYPE_CHECKING, Final

import typol as tp

if TYPE_CHECKING:
    from ty_extensions import TypeOf, is_equivalent_to, static_assert


class Int(tp.Shape):
    value = tp.dimension(int)


class Float(tp.Shape):
    value = tp.dimension(tp.FLOAT_64)


class String(tp.Shape):
    value = tp.dimension(str)


def _single_col_df[S: tp.Shape, T](
    dimension: tp.BoundDimension[S, T], *values: T | None
) -> tp.DataFrame[S]:
    return tp.DataFrame(
        dimension.shape, [tp.Entry.of(dimension.set_or_null(v)) for v in values]
    )


_INTS: Final = _single_col_df(Int.value, 1, 5, 2, 4, 3)
_FLOATS: Final = _single_col_df(Float.value, 1.2, 4.8, 2.5, 4.0, math.pi)
_STRS: Final = _single_col_df(String.value, "spam", "eggs", "foo", "bar")


def test_comparsions() -> None:
    assert _INTS.filter(Int.value.gt(3)).equals(_single_col_df(Int.value, 5, 4))
    assert _INTS.filter(Int.value.lt(3)).equals(_single_col_df(Int.value, 1, 2))
    assert _INTS.filter(Int.value.ge(3)).equals(_single_col_df(Int.value, 5, 4, 3))
    assert _INTS.filter(Int.value.le(3)).equals(_single_col_df(Int.value, 1, 2, 3))
    assert _INTS.filter(Int.value.eq(3)).equals(_single_col_df(Int.value, 3))

    assert _FLOATS.filter(Float.value.gt(math.pi)).equals(
        _single_col_df(Float.value, 4.8, 4.0)
    )
    assert _FLOATS.filter(Float.value.lt(math.pi)).equals(
        _single_col_df(Float.value, 1.2, 2.5)
    )
    assert _FLOATS.filter(Float.value.ge(math.pi)).equals(
        _single_col_df(Float.value, 4.8, 4.0, math.pi)
    )
    assert _FLOATS.filter(Float.value.le(math.pi)).equals(
        _single_col_df(Float.value, 1.2, 2.5, math.pi)
    )
    assert _FLOATS.filter(Float.value.eq(math.pi)).equals(
        _single_col_df(Float.value, math.pi)
    )

    assert _STRS.filter(String.value.gt("foo")).equals(
        _single_col_df(String.value, "spam")
    )
    assert _STRS.filter(String.value.lt("foo")).equals(
        _single_col_df(String.value, "eggs", "bar")
    )
    assert _STRS.filter(String.value.ge("foo")).equals(
        _single_col_df(String.value, "spam", "foo")
    )
    assert _STRS.filter(String.value.le("foo")).equals(
        _single_col_df(String.value, "eggs", "foo", "bar")
    )
    assert _STRS.filter(String.value.eq("foo")).equals(
        _single_col_df(String.value, "foo")
    )


def test_comparsions_as_operators() -> None:
    assert _INTS.filter(Int.value > 3).equals(_single_col_df(Int.value, 5, 4))
    assert _INTS.filter(Int.value < 3).equals(_single_col_df(Int.value, 1, 2))
    assert _INTS.filter(Int.value >= 3).equals(_single_col_df(Int.value, 5, 4, 3))
    assert _INTS.filter(Int.value <= 3).equals(_single_col_df(Int.value, 1, 2, 3))
    assert _INTS.filter(Int.value == 3).equals(_single_col_df(Int.value, 3))
    assert _INTS.filter(Int.value != 3).equals(_single_col_df(Int.value, 1, 5, 2, 4))

    assert _FLOATS.filter(Float.value > math.pi).equals(
        _single_col_df(Float.value, 4.8, 4.0)
    )
    assert _FLOATS.filter(Float.value < math.pi).equals(
        _single_col_df(Float.value, 1.2, 2.5)
    )
    assert _FLOATS.filter(Float.value >= math.pi).equals(
        _single_col_df(Float.value, 4.8, 4.0, math.pi)
    )
    assert _FLOATS.filter(Float.value <= math.pi).equals(
        _single_col_df(Float.value, 1.2, 2.5, math.pi)
    )
    assert _FLOATS.filter(Float.value == math.pi).equals(
        _single_col_df(Float.value, math.pi)
    )

    assert _STRS.filter(String.value > "foo").equals(
        _single_col_df(String.value, "spam")
    )
    assert _STRS.filter(String.value < "foo").equals(
        _single_col_df(String.value, "eggs", "bar")
    )
    assert _STRS.filter(String.value >= "foo").equals(
        _single_col_df(String.value, "spam", "foo")
    )
    assert _STRS.filter(String.value <= "foo").equals(
        _single_col_df(String.value, "eggs", "foo", "bar")
    )
    assert _STRS.filter(String.value == "foo").equals(
        _single_col_df(String.value, "foo")
    )


def test_arithmetic() -> None:
    assert _INTS.with_columns(Int.value * 2 + 5).equals(
        _single_col_df(Int.value, 7, 15, 9, 13, 11)
    )
    assert _FLOATS.with_columns(Float.value**2 / 2).equals(
        _single_col_df(Float.value, 0.72, 11.52, 3.125, 8.0, math.pi**2 / 2)
    )
    assert _STRS.with_columns(String.value + "!").equals(
        _single_col_df(String.value, "spam!", "eggs!", "foo!", "bar!")
    )


def test_lit() -> None:
    assert _INTS.with_columns((tp.lit(5) + Int.value * 2).to(Int.value)).equals(
        _single_col_df(Int.value, 7, 15, 9, 13, 11)
    )
    # Use a column name to make sure it doesn't pick that up
    assert _STRS.with_columns((tp.lit("value") + String.value).to(String.value)).equals(
        _single_col_df(String.value, "valuespam", "valueeggs", "valuefoo", "valuebar")
    )


def test_when() -> None:
    over_three_divide_by_two_otherwise_add_one = _INTS.with_columns(
        tp.when(Int.value > 3).then(Int.value / 2).otherwise(Int.value + 1)
    )
    assert over_three_divide_by_two_otherwise_add_one.equals(
        _single_col_df(Int.value, 2, 2, 3, 2, 4)
    )
    if TYPE_CHECKING:
        static_assert(
            is_equivalent_to(
                TypeOf[over_three_divide_by_two_otherwise_add_one], tp.DataFrame[Int]
            )
        )
    # Use a column name to make sure it doesn't pick that up
    assert _STRS.with_columns(
        tp.when(String.value.str.len_chars().gt(3)).then(String.value)
    ).equals(_single_col_df(String.value, "spam", "eggs", None, None))
