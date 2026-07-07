import datetime
import math
from typing import TYPE_CHECKING, Final

import typol as tp
from packaging.version import Version

import polars as pl

if TYPE_CHECKING:
    from ty_extensions import TypeOf, is_assignable_to, is_equivalent_to, static_assert


class Int(tp.Shape):
    value = tp.dimension(int)


class Float(tp.Shape):
    value = tp.dimension(tp.FLOAT_64)


class String(tp.Shape):
    value = tp.dimension(str)


class Person(tp.Shape):
    name = tp.dimension(str)
    date_of_birth = tp.dimension(datetime.date)


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

    if TYPE_CHECKING:
        # Make sure `ty` doesn't introduce `& Unknown when dealing with literals in comparisons
        static_assert(is_equivalent_to(TypeOf[Int.value > 3], tp.ExoExpr[Int, bool]))
        static_assert(is_equivalent_to(TypeOf[Int.value < 3], tp.ExoExpr[Int, bool]))
        static_assert(is_equivalent_to(TypeOf[Int.value == 3], tp.ExoExpr[Int, bool]))
        static_assert(is_equivalent_to(TypeOf[Int.value != 3], tp.ExoExpr[Int, bool]))
        static_assert(is_equivalent_to(TypeOf[Int.value >= 3], tp.ExoExpr[Int, bool]))
        static_assert(is_equivalent_to(TypeOf[Int.value <= 3], tp.ExoExpr[Int, bool]))

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


def test_rarithmetic() -> None:
    assert _INTS.with_columns((5 + 2 * Int.value).to(Int.value)).equals(
        _single_col_df(Int.value, 7, 15, 9, 13, 11)
    )
    assert _FLOATS.with_columns((2 / 2**Float.value).to(Float.value)).equals(
        _single_col_df(
            Float.value, 2 / 2**1.2, 2 / 2**4.8, 2 / 2**2.5, 0.125, 2 / 2**math.pi
        )
    )
    assert _STRS.with_columns(("!" + String.value).to(String.value)).equals(
        _single_col_df(String.value, "!spam", "!eggs", "!foo", "!bar")
    )


def test_abs() -> None:
    assert _INTS.with_columns((2 - Int.value).to(Int.value).abs()).equals(
        _single_col_df(Int.value, 1, 3, 0, 2, 1)
    )
    assert _INTS.with_columns(abs(2 - Int.value).to(Int.value)).equals(
        _single_col_df(Int.value, 1, 3, 0, 2, 1)
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
        static_assert(
            is_equivalent_to(
                TypeOf[tp.when(Int.value > 3).then(Int.value).otherwise(-Int.value)],
                tp.EndoExpr[Int, int],
            )
        )
    # Use a column name to make sure it doesn't pick that up
    assert _STRS.with_columns(
        tp.when(String.value.str.len_chars().gt(3)).then(String.value)
    ).equals(_single_col_df(String.value, "spam", "eggs", None, None))


def test_str_functions() -> None:
    assert _STRS.filter(_STRS.s.value.str.contains(r"a[rm]"))[
        _STRS.s.value
    ].to_list() == ["spam", "bar"]
    assert _STRS.filter(_STRS.s.value.str.contains_any([r"am", r"oo"]))[
        _STRS.s.value
    ].to_list() == ["spam", "foo"]
    assert _STRS.filter(_STRS.s.value.str.count_matches(r"[og]") > 1)[
        _STRS.s.value
    ].to_list() == ["eggs", "foo"]
    assert _STRS.filter(_STRS.s.value.str.ends_with("oo"))[_STRS.s.value].to_list() == [
        "foo"
    ]
    assert _STRS[_STRS.s.value.str.split("a")].to_list() == [
        ["sp", "m"],
        ["eggs"],
        ["foo"],
        ["b", "r"],
    ]
    # Older Polars versions don't support splitting by regex
    if Version(tp.expr.pl.__version__) >= Version("1.40"):
        assert _STRS[_STRS.s.value.str.split("[oa]", literal=False)].to_list() == [
            ["sp", "m"],
            ["eggs"],
            ["f", "", ""],
            ["b", "r"],
        ]
    assert _STRS[_STRS.s.value.str.extract("(.[oa].)")].to_list() == [
        "pam",
        None,
        "foo",
        "bar",
    ]


def test_intersections() -> None:
    assert (Person & String).shape_meta().schema == pl.Schema(
        {
            "name": pl.String,
            "date_of_birth": pl.Date,
            "value": pl.String,
        }
    )
    assert (String & Person).shape_meta().schema == pl.Schema(
        {
            "value": pl.String,
            "name": pl.String,
            "date_of_birth": pl.Date,
        }
    )
    # Conflicting columns
    assert (String & Int).shape_meta().schema == pl.Schema(
        {
            "value": pl.String,
        }
    )
    assert (Int & String).shape_meta().schema == pl.Schema(
        {
            "value": pl.Int64,
        }
    )

    class TwoStrings(String):
        value2 = tp.dimension(str)

    # Intersecting with some subshape
    assert (TwoStrings & String).shape_meta().schema == pl.Schema(
        {
            "value": pl.String,
            "value2": pl.String,
        }
    )
    assert (String & TwoStrings).shape_meta().schema == pl.Schema(
        {
            "value": pl.String,
            "value2": pl.String,
        }
    )


def test_date() -> None:
    people = tp.DataFrame(
        Person,
        [
            tp.Entry.of(
                Person.name.set("William"),
                Person.date_of_birth.set(datetime.date(1759, 8, 24)),
            ),
            tp.Entry.of(
                Person.name.set("Douglas"),
                Person.date_of_birth.set(datetime.date(1952, 3, 11)),
            ),
        ],
    )
    first_of_birth_year = tp.date(people.s.date_of_birth.dt.year(), 1, 1)
    if TYPE_CHECKING:
        static_assert(
            is_equivalent_to(
                TypeOf[first_of_birth_year], tp.ExoExpr[Person, datetime.date]
            )
        )
    first_of_birth_year = people[first_of_birth_year].to_list()
    assert first_of_birth_year == [datetime.date(1759, 1, 1), datetime.date(1952, 1, 1)]


def test_date_arithmetic() -> None:
    people = tp.DataFrame(
        Person,
        [
            tp.Entry.of(
                Person.name.set("William"),
                Person.date_of_birth.set(datetime.date(1759, 8, 24)),
            ),
            tp.Entry.of(
                Person.name.set("Douglas"),
                Person.date_of_birth.set(datetime.date(1952, 3, 11)),
            ),
        ],
    )

    assert people[people.s.date_of_birth + datetime.timedelta(days=30)].to_list() == [
        datetime.date(1759, 9, 23),
        datetime.date(1952, 4, 10),
    ]
    if TYPE_CHECKING:
        # You shouldn't be able to subtract a date from a timedelta, only the other way round
        datetime.timedelta(days=30) - people.s.date_of_birth  # ty: ignore[unsupported-operator]
        static_assert(
            is_assignable_to(
                tp.EndoExpr[Person, datetime.date],
                TypeOf[people.s.date_of_birth - datetime.timedelta(days=30)],
            )
        )
    with_turns_ten_on = people.with_columns(
        (
            tp.date(
                people.s.date_of_birth.dt.year() + 10,
                people.s.date_of_birth.dt.month(),
                people.s.date_of_birth.dt.day(),
            ).alias("turns_ten_on")
        )
    )
    assert with_turns_ten_on[
        with_turns_ten_on.s.turns_ten_on - people.s.date_of_birth
    ].to_list() == [datetime.timedelta(days=3653), datetime.timedelta(days=3652)]


def test_agg_arithmetic() -> None:
    # Older Polars versions don't allow these single-column aggregations since there's no key
    if Version(pl.__version__) >= Version("1.40"):
        assert _INTS.agg(Int.value.sum() * 2 + 5).equals(_single_col_df(Int.value, 35))
        assert _FLOATS.agg(Float.value.sum() ** 2 / Float.value.sum()).equals(
            _single_col_df(Float.value, 15.641592653589793)
        )
        assert _STRS.agg(String.value.str.join("") + "!").equals(
            _single_col_df(String.value, "spameggsfoobar!")
        )


def test_sort() -> None:
    assert _INTS[_INTS.s.value.sort()].to_list() == [1, 2, 3, 4, 5]
    assert _INTS[
        _INTS.s.value.sort_by((_INTS.s.value - 3).abs(), _INTS.s.value > 3)
    ].to_list() == [3, 2, 4, 1, 5]
