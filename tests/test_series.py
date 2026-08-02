import typol as tp

import polars as pl


class Int(tp.Shape):
    integer = tp.dimension(int)


class IntAndStr(Int):
    string = tp.dimension(str)


def test_series_into_frame() -> None:
    ints = tp.DataFrame(Int, (Int.integer.set_all([1, 2, 3]),))
    ints_and_strs = ints.transform(
        IntAndStr, tp.Series(pl.Series(list("abc"))).to(IntAndStr.string)
    )
    assert ints_and_strs.to_dicts() == [
        {
            "integer": 1,
            "string": "a",
        },
        {
            "integer": 2,
            "string": "b",
        },
        {
            "integer": 3,
            "string": "c",
        },
    ]
