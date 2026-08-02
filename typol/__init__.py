"""
Typol
-----

A typed wrapper around Polars, that allows static type enforcement of Polars code:

```python
import typol as tp

class Account(tp.Shape):
    name = tp.dimension(str)
    website = tp.dimension(str)
    account_age = tp.dimension(tp.UINT_8)
    phone = tp.dimension(str)


class Contact(tp.Shape):
    email = tp.dimension(str)
    known_since = tp.dimension(tp.UINT_16)
    phone = tp.dimension(str)


accounts.with_columns(
    # This is type checked so the `+` operator must be on a number, and the used and produced
    # dimensions must all be in `Account`
    accounts.s.account_age + 1
)

contacts = accounts.transform(
    Contact,
    # This operation must only use dimensions that are available in `Account`, and must end up at
    # a `Contact` dimension. All expression types are also checked to be `str`. All static checks
    (accounts.s.name + "@" + accounts.s.website).to(Contact.email),
    # Similar to the above, except with `int`s
    (tp.lit(2026) - account.s.account_age).to(Contact.known_since),
    # `phone` is in both shapes so we can leave it alone
)

emails = contacts[Contact.email].to_list()
reveal_type(emails)  # list[str], Contact.known_since would reveal to `list[int]`
```

As much of the above is statically enforced as is possible, giving much greater guarantees for
dataframe code. Where static enforcement is not possible, dynamic enforcement is used to ensure
the static types are always correct
"""

from . import expr, frame, types
from .expr import (
    BoundDimension,
    Dimension,
    Element,
    EndoExpr,
    ExoExpr,
    Explosion,
    Expr,
    MesoExpr,
    Shape,
    Suffixed,
    When,
    WithColumn,
    all_horizontal,
    any_horizontal,
    arg_sort_by,
    concat_list,
    date,
    date_range,
    date_ranges,
    datetime_range,
    dimension,
    duration,
    element,
    length,
    lit,
    max_horizontal,
    min_horizontal,
    null,
    projection,
    row_index,
    struct,
    suffix,
    when,
)
from .frame import DataFrame
from .lazy import LazyFrame
from .row import Entry, Row
from .series import Series
from .types import (
    BOOLEAN,
    CATEGORICAL,
    DATE,
    DATETIME,
    FLOAT_32,
    FLOAT_64,
    INT_8,
    INT_16,
    INT_32,
    INT_64,
    STRING,
    TIME,
    UINT_8,
    UINT_16,
    UINT_32,
    UINT_64,
    StructMapping,
    Type,
    datetime,
    decimal,
    enum,
    list_of,
    struct_of,
)

len = length
min = Expr.min
max = Expr.max
read_csv = DataFrame.read_csv
scan_csv = LazyFrame.scan_csv

__all__ = [
    "BOOLEAN",
    "CATEGORICAL",
    "DATE",
    "DATETIME",
    "FLOAT_32",
    "FLOAT_64",
    "INT_8",
    "INT_16",
    "INT_32",
    "INT_64",
    "STRING",
    "TIME",
    "UINT_8",
    "UINT_16",
    "UINT_32",
    "UINT_64",
    "BoundDimension",
    "DataFrame",
    "Dimension",
    "Element",
    "EndoExpr",
    "Entry",
    "ExoExpr",
    "Explosion",
    "Expr",
    "LazyFrame",
    "MesoExpr",
    "Row",
    "Series",
    "Shape",
    "StructMapping",
    "Suffixed",
    "Type",
    "When",
    "WithColumn",
    "all",
    "all_horizontal",
    "any",
    "any_horizontal",
    "arg_sort_by",
    "concat_list",
    "date",
    "date_range",
    "date_ranges",
    "datetime",
    "datetime_range",
    "decimal",
    "dimension",
    "duration",
    "element",
    "enum",
    "expr",
    "frame",
    "len",
    "list_of",
    "lit",
    "max",
    "max_horizontal",
    "min",
    "min_horizontal",
    "null",
    "projection",
    "read_csv",
    "row_index",
    "scan_csv",
    "struct",
    "struct_of",
    "suffix",
    "types",
    "when",
]
