"""
Use Polars' DataFrame tests to check compatibility

source: https://github.com/pola-rs/polars/blob/main/py-polars/tests/unit/dataframe/test_df.py

This has been modified to:
 - Use typol, and thus remove flexible shape tests
 - Exclude tests based on Polars internals, or cross-comptability with direct numpy/pyarrow types
 - Some tests have not been converted and have been ommitted
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterator, Mapping
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError
from polars.testing import assert_frame_equal, assert_series_equal

import typol as tp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class IntInt(tp.Shape):
    a = tp.dimension(int)
    b = tp.dimension(int)


@pytest.mark.parametrize("input", [None, (), [], {}])
def test_init_empty(input: Collection | None) -> None:
    # test various flavours of empty init
    df = tp.DataFrame(IntInt, input)
    assert df.is_empty()


class IntFloatStr(tp.Shape):
    a = tp.dimension(int)
    b = tp.dimension(float)
    c = tp.dimension(str)


class OtherShape(tp.Shape):
    other_column = tp.dimension(int)


def test_column_selection() -> None:
    df = tp.DataFrame(
        ifs := IntFloatStr,
        (
            ifs.a.set_all([1, 2, 3]),
            ifs.b.set_all([1.0, 2.0, 3.0]),
            ifs.c.set_all(["a", "b", "c"]),
        ),
    )

    # get column by name
    b = pl.Series("b", [1.0, 2.0, 3.0])
    assert_series_equal(df[IntFloatStr.b].data, b)
    assert_series_equal(df.get_column(IntFloatStr.b).data, b)

    with pytest.raises(ColumnNotFoundError, match="other_column"):
        df.get_column(OtherShape.other_column)  # ty: ignore[invalid-argument-type]


def test_mixed_sequence_selection() -> None:
    df = tp.DataFrame(IntInt, (IntInt.a.set_all([1, 2]), IntInt.b.set_all([3, 4])))
    result = df.transform(
        ifs := IntFloatStr, IntInt.b.cast_out(float).to(ifs.b), tp.lit("c").to(ifs.c)
    )
    expected = tp.DataFrame(
        ifs := IntFloatStr,
        (ifs.a.set_all([1, 2]), ifs.b.set_all([3, 4]), ifs.c.set_all(["c", "c"])),
    )
    assert_frame_equal(result.dataframe, expected.dataframe)


def test_sort() -> None:
    df = tp.DataFrame(i := IntInt, (i.a.set_all([2, 1, 3]), i.b.set_all([1, 2, 3])))
    expected = tp.DataFrame(
        i := IntInt, (i.a.set_all([1, 2, 3]), i.b.set_all([2, 1, 3]))
    )
    assert_frame_equal(df.sort(IntInt.a).dataframe, expected.dataframe)
    assert_frame_equal(df.sort(IntInt.a, IntInt.b).dataframe, expected.dataframe)


class IntStr(tp.Shape):
    A = tp.dimension(int)
    B = tp.dimension(str)


def test_sort_maintain_order() -> None:
    l1 = (
        tp.LazyFrame(
            s := IntStr, (s.A.set_all([1] * 4), s.B.set_all(["A", "B", "C", "D"]))
        )
        .sort(s.A, maintain_order=True)
        .slice(0, 3)
        .collect()[s.B]
        .to_list()
    )
    l2 = (
        tp.LazyFrame(
            s := IntStr, (s.A.set_all([1] * 4), s.B.set_all(["A", "B", "C", "D"]))
        )
        .sort(s.A)
        .collect()
        .slice(0, 3)[s.B]
        .to_list()
    )
    assert l1 == l2 == ["A", "B", "C"]


@pytest.mark.parametrize("nulls_last", [False, True], ids=["nulls_first", "nulls_last"])
def test_sort_maintain_order_descending_repeated_nulls(nulls_last: bool) -> None:
    got = (
        tp.LazyFrame(
            i := IntInt,
            (i.a.set_or_null_all([None, -1, 1, 1, None]), i.b.set_all([1, 2, 3, 4, 5])),
        )
        .sort(i.a, descending=True, maintain_order=True, nulls_last=nulls_last)
        .collect()
    )
    if nulls_last:
        expect = tp.DataFrame(
            i := IntInt,
            (i.a.set_or_null_all([1, 1, -1, None, None]), i.b.set_all([3, 4, 2, 1, 5])),
        )
    else:
        expect = tp.DataFrame(
            i := IntInt,
            (i.a.set_or_null_all([None, None, 1, 1, -1]), i.b.set_all([1, 5, 3, 4, 2])),
        )
    assert_frame_equal(got.dataframe, expect.dataframe)


def test_assignment() -> None:
    df = tp.DataFrame(i := IntInt, (i.a.set_all([1, 2, 3]), i.b.set_all([2, 3, 4])))
    df = df.with_columns(IntInt.a.to(IntInt.a))
    # make sure that assignment does not change column order
    assert df.dataframe.columns == ["a", "b"]
    df = df.with_columns(
        tp.when(IntInt.a.gt(1)).then(9).otherwise(IntInt.a).to(IntInt.a)
    )
    assert df[IntInt.a].to_list() == [1, 9, 9]


def test_gather_every() -> None:
    df = tp.DataFrame(
        IntStr, (IntStr.A.set_all([1, 2, 3, 4]), IntStr.B.set_all(["w", "x", "y", "z"]))
    )
    expected_df = tp.DataFrame(
        IntStr, (IntStr.A.set_all([1, 3]), IntStr.B.set_all(["w", "y"]))
    )
    assert_frame_equal(expected_df.dataframe, df.gather_every(2).dataframe)

    expected_df = tp.DataFrame(
        IntStr, (IntStr.A.set_all([2, 4]), IntStr.B.set_all(["x", "z"]))
    )
    assert_frame_equal(expected_df.dataframe, df.gather_every(2, offset=1).dataframe)


class IntStrs(tp.Shape):
    g = tp.dimension(int)
    a = tp.dimension(tp.list_of(str))


def test_gather_every_agg() -> None:
    df = tp.DataFrame(
        IntStr,
        (
            IntStr.A.set_all([1, 1, 1, 2, 2, 2]),
            IntStr.B.set_all(["a", "b", "c", "d", "e", "f"]),
        ),
    )
    out = df.agg_transform(
        iss := IntStrs, IntStr.A.to(iss.g), IntStr.B.gather_every(2).to(iss.a)
    )
    expected = tp.DataFrame(
        IntStrs,
        (IntStrs.g.set_all([1, 2]), IntStrs.a.set_all([["a", "c"], ["d", "f"]])),
    )
    assert_frame_equal(out.sort(IntStrs.g).dataframe, expected.dataframe)


def test_pipe() -> None:
    df = tp.DataFrame(
        i := IntInt, (i.a.set_all([1, 2, 3]), i.b.set_or_null_all([6, None, 8]))
    )

    def _multiply(data: tp.DataFrame[IntInt], mul: int) -> tp.DataFrame[IntInt]:
        return data.with_columns(data.s.a * mul, data.s.b * mul)

    result = df.pipe(_multiply, mul=3)

    assert_frame_equal(result.dataframe, df.dataframe * 3)


class IntsStr(tp.Shape):
    nrs = tp.dimension(tp.list_of(int))
    letters = tp.dimension(str)


def test_explode() -> None:
    df = tp.DataFrame(
        iss := IntsStr,
        (iss.letters.set_all(["c", "a"]), iss.nrs.set_all([[1, 2], [1, 3]])),
    )
    out = df.explode_transform(
        IntStr, IntsStr.nrs.list.explode_to(IntStr.A), IntsStr.letters.to(IntStr.B)
    )
    assert out[IntStr.B].to_list() == ["c", "c", "a", "a"]
    assert out[IntStr.A].to_list() == [1, 2, 1, 3]


class SixCols(tp.Shape):
    a = tp.dimension(int)
    b = tp.dimension(int)
    c = tp.dimension(int)
    d = tp.dimension(int)
    e = tp.dimension(int)
    f = tp.dimension(int)


def test_file_buffer() -> None:
    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = tp.DataFrame.read_csv(SixCols, f, has_header=False)
    assert df.dataframe.shape == (2, 6)


def test_file_buffer_has_header() -> None:
    f = BytesIO()
    f.write(b"A,B,C,d,e,f\n1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = tp.DataFrame.read_csv(SixCols, f)
    assert df.dataframe.shape == (2, 6)


def test_shift() -> None:
    df = tp.DataFrame(
        i := IntStr, (i.B.set_all(["a", "b", "c"]), i.A.set_all([1, 3, 5]))
    )
    a = df.shift(1)
    b = tp.DataFrame(
        i := IntStr,
        (i.B.set_or_null_all([None, "a", "b"]), i.A.set_or_null_all([None, 1, 3])),
    )
    assert_frame_equal(a.dataframe, b.dataframe)


def test_is_null_is_not_null() -> None:
    df = tp.DataFrame(
        i := IntInt, (i.a.set_or_null_all([1, 2, None]), i.b.set_or_null_all([1, 2, 3]))
    )
    assert df[i.a.is_null()].to_list() == [False, False, True]
    assert df[i.a.is_not_null()].to_list() == [True, True, False]


class AFloat(tp.Shape):
    nrs = tp.dimension(tp.FLOAT_64)


def test_is_nan_is_not_nan() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1, 2, float("nan")]),))
    assert df[AFloat.nrs.is_nan()].to_list() == [False, False, True]
    assert df[AFloat.nrs.is_not_nan()].to_list() == [True, True, False]


def test_is_finite_is_nan() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1, 2, float("inf")]),))
    assert df[AFloat.nrs.is_infinite()].to_list() == [False, False, True]
    assert df[AFloat.nrs.is_finite()].to_list() == [True, True, False]


def test_is_finite_is_infinite_null_series() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_or_null_all([None, None, None]),))
    assert df[AFloat.nrs.is_infinite()].to_list() == [None, None, None]
    assert df[AFloat.nrs.is_finite()].to_list() == [None, None, None]


def test_is_nan_null_series() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_or_null_all([None, None, None]),))
    assert df[AFloat.nrs.is_nan()].to_list() == [None, None, None]


def test_len() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1, 2, 3]),))
    assert df[AFloat.nrs.len()].first() == 3
    assert len(tp.DataFrame(IntInt)) == 0


def test_multiple_column_sort() -> None:
    df = tp.DataFrame(
        ifs := IntFloatStr,
        (
            ifs.c.set_all(["foo", "bar", "2"]),
            ifs.a.set_all([2, 2, 3]),
            ifs.b.set_all([1.0, 2.0, 3.0]),
        ),
    )
    out = df.sort(IntFloatStr.a, IntFloatStr.c)
    assert list(out[IntFloatStr.b]) == [2.0, 1.0, 3.0]
    assert list(out[IntFloatStr.a]) == [2, 2, 3]

    df = tp.DataFrame(
        i := IntStr, (i.A.set_all(range(1, 4)), i.B.set_all(["a", "a", "b"]))
    )

    assert_frame_equal(
        df.sort(i.A, descending=True).dataframe,
        right=pl.DataFrame({"A": [3, 2, 1], "B": ["b", "a", "a"]}),
    )
    assert_frame_equal(
        df.sort(i.B, descending=True, maintain_order=True).dataframe,
        pl.DataFrame({"A": [3, 1, 2], "B": ["b", "a", "a"]}),
    )
    assert_frame_equal(
        df.sort(i.B, i.A, descending=(False, True)).dataframe,
        pl.DataFrame({"A": [2, 1, 3], "B": ["a", "a", "b"]}),
    )


class FloatIntBoolDate(tp.Shape):
    a = tp.dimension(float)
    b = tp.dimension(int)
    c = tp.dimension(bool)
    d = tp.dimension(date)


class FloatFloatBoolDatetime(tp.Shape):
    a = tp.dimension(float)
    b = tp.dimension(tp.FLOAT_32)
    c = tp.dimension(tp.STRING)
    d = tp.dimension(tp.datetime("ms"))


def test_cast_frame() -> None:
    df = tp.DataFrame(
        s := FloatIntBoolDate,
        (
            s.a.set_or_null_all([1.0, 2.5, 3.0]),
            s.b.set_or_null_all([4, 5, None]),
            s.c.set_or_null_all([True, False, True]),
            s.d.set_or_null_all([date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)]),
        ),
    )

    # cast via col:dtype map
    assert df.transform(FloatFloatBoolDatetime).dataframe.schema == {
        "a": pl.Float64,
        "b": pl.Float32,
        "c": pl.String,
        "d": pl.Datetime("ms"),
    }


class DatetimeTriplet(tp.Shape):
    a = tp.dimension(datetime)
    b = tp.dimension(datetime)
    c = tp.dimension(datetime)


def test_duration_arithmetic() -> None:
    df = tp.DataFrame(
        d := DatetimeTriplet,
        (
            d.a.set_all([datetime(2022, 1, 1, 0, 0, 0), datetime(2022, 1, 2, 0, 0, 0)]),
            d.b.set_or_null_all([None] * 2),
            d.c.set_or_null_all([None] * 2),
        ),
    )
    d1 = tp.duration(days=3, microseconds=987000)
    d2 = tp.duration(days=6, milliseconds=987)

    assert_frame_equal(
        df.with_columns((df.s.a + d1).to(d.b), (df.s.a + d2).to(d.c)).dataframe,
        pl.DataFrame(
            {
                "a": [datetime(2022, 1, 1, 0, 0, 0), datetime(2022, 1, 2, 0, 0, 0)],
                "b": [
                    datetime(2022, 1, 4, 0, 0, 0, 987000),
                    datetime(2022, 1, 5, 0, 0, 0, 987000),
                ],
                "c": [
                    datetime(2022, 1, 7, 0, 0, 0, 987000),
                    datetime(2022, 1, 8, 0, 0, 0, 987000),
                ],
            }
        ),
    )


def test_assign() -> None:
    # check if can assign in case of a single column
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1, 2, 3]),))
    # test if we can assign in case of single column
    df = df.with_columns(df.s.nrs * 2)
    assert list(df[df.s.nrs]) == [2, 4, 6]


class IntFloat(tp.Shape):
    ints = tp.dimension(int)
    floats = tp.dimension(float)


def test_arg_sort_by() -> None:
    df = tp.DataFrame(
        s := IntFloat,
        (s.ints.set_or_null_all([1, None, 2]), s.floats.set_all([1.0, 1.2, 1.1])),
    )
    idx = df[tp.arg_sort_by(df.s.ints, df.s.floats, descending=[False, True])]
    assert idx.to_list() == [1, 0, 2]

    idx = df[tp.arg_sort_by(df.s.ints, df.s.floats, descending=False)]
    assert idx.to_list() == [1, 0, 2]

    df = tp.DataFrame(
        i := IntInt, (i.a.set_all([0, 0, 0, 1, 1, 2]), i.b.set_all([9, 9, 8, 7, 6, 6]))
    )
    for expr, expected in (
        (tp.arg_sort_by(IntInt.a, IntInt.b), [2, 0, 1, 4, 3, 5]),
        (
            tp.arg_sort_by(IntInt.a, IntInt.b, descending=[True, True]),
            [5, 3, 4, 0, 1, 2],
        ),
        (
            tp.arg_sort_by(IntInt.a, IntInt.b, descending=[True, False]),
            [5, 4, 3, 2, 0, 1],
        ),
        (
            tp.arg_sort_by(IntInt.a, IntInt.b, descending=[False, True]),
            [0, 1, 2, 3, 4, 5],
        ),
    ):
        assert df[expr].to_list() == expected


class FloatIntStrDatetime(tp.Shape):
    a = tp.dimension(tp.FLOAT_64)
    b = tp.dimension(tp.INT_8)
    c = tp.dimension(str)
    d = tp.dimension(tp.datetime("ns"))


class FloatIntStrDatetimeFloat(FloatIntStrDatetime):
    e = tp.dimension(tp.FLOAT_32)


def test_literal_series() -> None:
    df = tp.DataFrame(
        s := FloatIntStrDatetime,
        (
            s.a.set_all([21.7, 21.8, 21]),
            s.b.set_all([1, 3, 2]),
            s.c.set_all(["reg1", "reg2", "reg3"]),
            s.d.set_all(
                [datetime(2022, 8, 16), datetime(2022, 8, 17), datetime(2022, 8, 18)]
            ),
        ),
    )
    out = (
        df.lazy()
        .transform(
            fisdf := FloatIntStrDatetimeFloat,
            tp.Series(pl.Series("nrs", [2, 1, 3], pl.Int32)).to(fisdf.e),
        )
        .collect()
    )
    expected_schema = {
        "a": pl.Float64,
        "b": pl.Int8,
        "c": pl.String,
        "d": pl.Datetime("ns"),
        "e": pl.Float32,
    }
    assert_frame_equal(
        pl.DataFrame(
            [
                (21.7, 1, "reg1", datetime(2022, 8, 16, 0), 2),
                (21.8, 3, "reg2", datetime(2022, 8, 17, 0), 1),
                (21.0, 2, "reg3", datetime(2022, 8, 18, 0), 3),
            ],
            schema=expected_schema,
            orient="row",
        ),
        out.dataframe,
        abs_tol=0.00001,
    )


def test_write_csv() -> None:
    df = tp.DataFrame(
        ifs := IntFloatStr,
        (
            ifs.a.set_all([1, 2, 3, 4, 5]),
            ifs.b.set_all([6, 7, 8, 9, 10]),
            ifs.c.set_all(["a", "b", "c", "d", "e"]),
        ),
    )
    expected = "a,b,c\n1,6.0,a\n2,7.0,b\n3,8.0,c\n4,9.0,d\n5,10.0,e\n"

    # if no file argument is supplied, write_csv() will return the string
    s = df.write_csv()
    assert s == expected

    # otherwise it will write to the file/iobuffer
    file = BytesIO()
    df.write_csv(file)
    file.seek(0)
    s = file.read().decode("utf8")
    assert s == expected


class StrThreeInts(tp.Shape):
    a = tp.dimension(str)
    b = tp.dimension(int)
    c = tp.dimension(int)
    d = tp.dimension(int)


class FourInts(tp.Shape):  # There are five ints
    a = tp.dimension(int)
    b = tp.dimension(int)
    c = tp.dimension(int)
    d = tp.dimension(int)


def test_from_generator_or_iterable() -> None:
    # generator function
    def gen[T: (int, str)](
        n: int, converter: Callable[[int], T]
    ) -> Iterator[tuple[T, int, int, int]]:
        for i in range(n):
            yield converter(i), 1 * i, 2**i, 3**i

    def gen_named(n: int) -> Iterator[Mapping[str, Any]]:
        for i in range(n):
            yield {"a": i, "b": 2 * i, "c": 2**i, "d": 3**i}

    # iterable object
    class Rows[T]:
        def __init__(self, n: int, converter: Callable[[int], T]) -> None:
            self._n = n
            self._converter = converter

        def __iter__(self) -> Iterator[tuple[T, int, int, int]]:
            yield from gen(self._n, converter=self._converter)

    # check init from column-oriented generator
    assert_frame_equal(
        tp.DataFrame(FourInts, gen(4, converter=int), orient="col").dataframe,
        tp.DataFrame(
            FourInts,
            [(0, 0, 1, 1), (1, 1, 2, 3), (2, 2, 4, 9), (3, 3, 8, 27)],
            orient="col",
        ).dataframe,
    )
    # check init from row-oriented generators (more common)
    expected = tp.DataFrame(StrThreeInts, list(gen(4, str)), orient="row")
    for generated_frame in (
        tp.DataFrame(StrThreeInts, gen(4, str)),
        tp.DataFrame(StrThreeInts, Rows(4, str)),
        tp.DataFrame(StrThreeInts, (x for x in Rows(4, str))),
    ):
        assert_frame_equal(expected.dataframe, generated_frame.dataframe)
        assert generated_frame.dataframe.schema == {
            "a": pl.String,
            "b": pl.Int64,
            "c": pl.Int64,
            "d": pl.Int64,
        }

    # empty iterator
    assert_frame_equal(
        tp.DataFrame(FourInts, gen(0, int)).dataframe, tp.DataFrame(FourInts).dataframe
    )

    # named dictionaries
    assert_frame_equal(
        tp.DataFrame(FourInts, gen_named(1)).dataframe,
        pl.DataFrame([{"a": 0, "b": 0, "c": 1, "d": 1}]),
    )


class SensorData(tp.Shape):
    sensor = tp.dimension(str)
    datetime = tp.dimension(tp.DATETIME)
    value = tp.dimension(int)


def test_join_dates() -> None:
    dts_in = tp.datetime_range(
        datetime(2021, 6, 24),
        datetime(2021, 6, 24, 10, 0, 0),
        interval=timedelta(hours=1),
        closed="left",
    )
    dts = (
        dts_in.cast_out(int)
        .map_out(lambda x: x + random.randint(1_000 * 60, 60_000 * 60), int)
        .cast_out(tp.DATETIME)
    )

    # some df with sensor id, (randomish) datetime and some value
    df = tp.DataFrame(
        s := SensorData,
        (
            s.sensor.set_all(["a"] * 5 + ["b"] * 5),
            s.datetime.set_or_null_all([None] * 10),
            s.value.set_all([2, 3, 4, 1, 2, 3, 5, 1, 2, 3]),
        ),
    )
    df = df.with_columns(dts.to(df.s.datetime))
    out = df.join(df, df.s.datetime)
    assert out.dataframe.height == df.dataframe.height


class Left(tp.Shape):
    a = tp.dimension(int)
    left_val = tp.dimension(str)


class Right(tp.Shape):
    a = tp.dimension(int)
    right_val = tp.dimension(int)


def test_asof_cross_join() -> None:
    left = tp.DataFrame(
        s := Left, (s.a.set_all([-10, 5, 10]), s.left_val.set_all(["a", "b", "c"]))
    )
    right = tp.DataFrame(
        s := Right, (s.a.set_all([1, 2, 3, 6, 7]), s.right_val.set_all([1, 2, 3, 6, 7]))
    )
    # only test dispatch of asof join
    out = left.join_asof(right, Left.a.on(Right.a))
    assert out.dataframe.shape == (3, 3)
    out = left.join_asof(right, (Left & Right).a.on())
    assert out.dataframe.shape == (3, 3)

    left.lazy().join_asof(right.lazy(), Left.a.on(Right.a)).collect()
    assert out.dataframe.shape == (3, 3)
    left.lazy().join_asof(right.lazy(), (Left & Right).a.on()).collect()
    assert out.dataframe.shape == (3, 3)

    # only test dispatch of cross join
    out = left.join(right, how="cross")
    assert out.dataframe.shape == (15, 3)

    left.lazy().join(right.lazy(), how="cross").collect()
    assert out.dataframe.shape == (15, 3)

    # only test dispatch of suffixed cross join
    right_s = right.suffix()
    out = left.join(right_s, how="cross")
    assert out.dataframe.shape == (15, 4)

    left.lazy().join(right_s.lazy(), how="cross").collect()
    assert out.dataframe.shape == (15, 4)


def test_str_concat() -> None:
    df = tp.DataFrame(
        s := IntStr,
        (s.A.set_all([1, 2, 3, 4]), s.B.set_or_null_all(["ham", "spam", "foo", None])),
    )
    out = df.with_columns((tp.lit("Dr. ") + IntStr.B).to(IntStr.B))
    assert out[IntStr.B].to_list() == ["Dr. ham", "Dr. spam", "Dr. foo", None]
    out = df.with_columns(IntStr.B + ", PhD")
    assert out[IntStr.B].to_list() == ["ham, PhD", "spam, PhD", "foo, PhD", None]


class StrInts(tp.Shape):
    x = tp.dimension(str)
    y = tp.dimension(tp.list_of(int))


def test_group_by_order_dispatch() -> None:
    df = tp.DataFrame(
        IntStr, (IntStr.B.set_all(list("bab")), IntStr.A.set_all(range(3)))
    )
    lf = df.lazy()

    result = df.agg(IntStr.A.len().to(IntStr.A))
    lazy_result = lf.agg(IntStr.A.len().to(IntStr.A)).sort(IntStr.B, descending=True)

    expected = tp.DataFrame(i := IntStr, (i.B.set_all(["b", "a"]), i.A.set_all([2, 1])))
    assert_frame_equal(result.sort(IntStr.B.reverse()).dataframe, expected.dataframe)
    assert_frame_equal(lazy_result.collect().dataframe, expected.dataframe)

    result = df.agg_transform(g := StrInts, i.B.to(g.x), i.A.agg().to(g.y))
    expected = tp.DataFrame(
        s := StrInts, (s.x.set_all(["b", "a"]), s.y.set_all([[0, 2], [1]]))
    )
    assert_frame_equal(result.sort(StrInts.x.reverse()).dataframe, expected.dataframe)


def test_schema() -> None:
    df = tp.DataFrame(
        ifs := IntFloatStr,
        (
            ifs.a.set_all([1, 2, 3]),
            ifs.b.set_all([6.0, 7.0, 8.0]),
            ifs.c.set_all(["a", "b", "c"]),
        ),
    )
    expected = {"a": pl.Int64, "b": pl.Float64, "c": pl.String}
    assert df.dataframe.schema == expected


class Empty(tp.Shape): ...


def test_empty_projection() -> None:
    empty_df = tp.DataFrame(
        i := IntInt, (i.a.set_all([1, 2]), i.b.set_all([3, 4]))
    ).transform(Empty)
    assert empty_df.to_dicts() == []
    assert empty_df.dataframe.schema == {}
    assert empty_df.dataframe.shape == (0, 0)


class StrStrs(tp.Shape):
    b = tp.dimension(str)
    c = tp.dimension(tp.list_of(str))


def test_fill_null() -> None:
    df = tp.DataFrame(
        i := IntInt, (i.a.set_all([1, 2]), i.b.set_or_null_all([3, None]))
    )
    assert_frame_equal(
        df.with_columns(i.b.fill_null(4)).dataframe,
        tp.DataFrame(i := IntInt, (i.a.set_all([1, 2]), i.b.set_all([3, 4]))).dataframe,
    )

    # string and list data
    # string goes via binary
    df = tp.DataFrame(
        s := StrStrs,
        (
            s.c.set_or_null_all(
                [["Anteater", "Ox"], ["Anteater", "Ox"], None, ["Cat"], None, None]
            ),
            s.b.set_or_null_all(["Anteater", "Ox", None, "Cat", None, None]),
        ),
    )

    assert df[StrStrs.c.fill_null(strategy="forward")].to_list() == [
        ["Anteater", "Ox"],
        ["Anteater", "Ox"],
        ["Anteater", "Ox"],
        ["Cat"],
        ["Cat"],
        ["Cat"],
    ]
    assert df[StrStrs.b.fill_null(strategy="forward")].to_list() == [
        "Anteater",
        "Ox",
        "Ox",
        "Cat",
        "Cat",
        "Cat",
    ]
    assert df[StrStrs.c.fill_null(strategy="backward")].to_list() == [
        ["Anteater", "Ox"],
        ["Anteater", "Ox"],
        ["Cat"],
        ["Cat"],
        None,
        None,
    ]
    assert df[StrStrs.b.fill_null(strategy="backward")].to_list() == [
        "Anteater",
        "Ox",
        "Cat",
        "Cat",
        None,
        None,
    ]


def test_fill_nan() -> None:
    df = tp.DataFrame(
        s := IntFloat, (s.ints.set_all([1, 2]), s.floats.set_all([3.0, float("nan")]))
    )
    assert_frame_equal(
        df.with_columns(IntFloat.floats.fill_nan(4)).dataframe,
        tp.DataFrame(
            s := IntFloat, (s.ints.set_all([1, 2]), s.floats.set_all([3.0, 4.0]))
        ).dataframe,
    )
    assert_frame_equal(
        df.with_columns(IntFloat.floats.fill_nan(None)).dataframe,
        tp.DataFrame(
            s := IntFloat,
            (s.ints.set_all([1, 2]), s.floats.set_or_null_all([3.0, None])),
        ).dataframe,
    )
    assert df[IntFloat.floats].fill_nan(5.0).to_list() == [3.0, 5.0]


class StrStr(tp.Shape):
    a = tp.dimension(str)
    b = tp.dimension(str)


def test_add_string() -> None:
    df = tp.DataFrame(
        s := StrStr, (s.a.set_all(["hi", "there"]), s.b.set_all(["hello", "world"]))
    )
    expected = tp.DataFrame(
        s := StrStr,
        (
            s.a.set_all(["hi hello", "there hello"]),
            s.b.set_all(["hello hello", "world hello"]),
        ),
    )
    assert_frame_equal(
        (df.with_columns(df.s.a + " hello", df.s.b + " hello")).dataframe,
        expected.dataframe,
    )

    expected = tp.DataFrame(
        s := StrStr,
        (
            s.a.set_all(["hello hi", "hello there"]),
            s.b.set_all(["hello hello", "hello world"]),
        ),
    )
    assert_frame_equal(
        df.with_columns(
            (tp.lit("hello ") + df.s.a).to(df.s.a),
            (tp.lit("hello ") + df.s.b).to(df.s.b),
        ).dataframe,
        expected.dataframe,
    )


class AnInt(tp.Shape):
    a = tp.dimension(int)


def test_ceil() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1.8, 1.2, 3.0]),))
    result = df[AFloat.nrs.ceil()]
    assert result.to_list() == [2.0, 2.0, 3.0]

    df = tp.DataFrame(i := AnInt, (i.a.set_all([1, 2, 3]),))
    result = df[AnInt.a.ceil()]
    assert result.to_list() == df[AnInt.a].to_list()


def test_floor() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1.8, 1.2, 3.0]),))
    result = df[AFloat.nrs.floor()]
    assert result.to_list() == [1.0, 1.0, 3.0]

    df = tp.DataFrame(i := AnInt, (i.a.set_all([1, 2, 3]),))
    result = df[AnInt.a.floor()]
    assert result.to_list() == df[AnInt.a].to_list()


def test_round() -> None:
    df = tp.DataFrame(f := AFloat, (f.nrs.set_all([1.8, 1.2, 3.0]),))
    col_a_rounded = df[AFloat.nrs.round(decimals=0)]
    assert_series_equal(
        col_a_rounded.data, pl.Series("nrs", [2, 1, 3]).cast(pl.Float64)
    )
