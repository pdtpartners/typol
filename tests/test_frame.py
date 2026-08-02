from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal

import pytest
import typol as tp
from typol.expr import Expr
from typol.types import EnumOf

if TYPE_CHECKING:
    from ty_extensions import (
        Intersection,
        static_assert,
    )
    from ty_extensions._internal import TypeOf, is_assignable_to, is_equivalent_to

type _Frame[S: tp.Shape] = type[tp.DataFrame[S] | tp.LazyFrame[S]]

data_and_lazy = pytest.mark.parametrize(["cls"], [[tp.DataFrame], [tp.LazyFrame]])


class Person(tp.Shape):
    name = tp.dimension(str)
    age = tp.dimension(int)


_PEOPLE: Final = (
    tp.Entry.of(Person.name.set("Douglas"), Person.age.set(42)),
    tp.Entry.of(Person.name.set("William"), Person.age.set(80)),
)


@data_and_lazy
def test_construct_from_entries[F: _Frame](cls: F) -> None:
    frame = cls(Person, _PEOPLE)
    assert frame.collect().to_dicts() == [
        {"name": "Douglas", "age": 42},
        {"name": "William", "age": 80},
    ]


@data_and_lazy
def test_with_columns[F: _Frame](cls: F) -> None:
    frame = cls(Person, _PEOPLE)
    assert frame.with_columns(
        Person.age - 20, Person.name + " Adams"
    ).collect().to_dicts() == [
        {"name": "Douglas Adams", "age": 22},
        {"name": "William Adams", "age": 60},
    ]


class Account(tp.Shape):
    username = tp.dimension(str)
    email = tp.dimension(str)


@data_and_lazy
def test_transform[F: _Frame](cls: F) -> None:
    frame = cls(Person, _PEOPLE)
    accounts = frame.transform(
        a := Account,
        frame.s.name.str.to_lowercase().to(a.username),
        (frame.s.name.str.to_lowercase() + "@example.net").to(a.email),
    )
    assert accounts.collect().to_dicts() == [
        {"username": "douglas", "email": "douglas@example.net"},
        {"username": "william", "email": "william@example.net"},
    ]


_ACCOUNTS: Final = (
    tp.Entry.of(
        Account.username.set("douglas"), Account.email.set("douglas@adams.net")
    ),
    tp.Entry.of(
        Account.username.set("will"), Account.email.set("will@wilberforce.net")
    ),
)


def test_join_df() -> None:
    people = tp.DataFrame(Person, _PEOPLE)
    accounts = tp.DataFrame(Account, _ACCOUNTS)
    lowered_name = Person.name.str.to_lowercase()
    left = people.join(accounts, lowered_name.on(Account.username), how="left")
    right = people.join(accounts, lowered_name.on(Account.username), how="right")
    inner = people.join(accounts, lowered_name.on(Account.username), how="inner")
    outer = people.join(accounts, lowered_name.on(Account.username), how="outer")
    assert left.sort(Person.name).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {"name": "William", "age": 80, "username": None, "email": None},
    ]
    assert right.sort(right.s.username).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]
    assert inner.to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        }
    ]
    assert outer.sort(Person.name, Account.username, nulls_last=True).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {"name": "William", "age": 80, "username": None, "email": None},
        {
            "name": None,
            "age": None,
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]


def test_join_lf() -> None:
    people = tp.LazyFrame(Person, _PEOPLE)
    accounts = tp.LazyFrame(Account, _ACCOUNTS)
    lowered_name = Person.name.str.to_lowercase()
    left = people.join(accounts, lowered_name.on(Account.username), how="left")
    right = people.join(accounts, lowered_name.on(Account.username), how="right")
    inner = people.join(accounts, lowered_name.on(Account.username), how="inner")
    outer = people.join(accounts, lowered_name.on(Account.username), how="outer")
    assert left.sort(Person.name).collect().to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {"name": "William", "age": 80, "username": None, "email": None},
    ]

    assert right.sort(right.s.username).collect().to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]
    assert inner.collect().to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        }
    ]
    assert outer.sort(
        Person.name, Account.username, nulls_last=True
    ).collect().to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {"name": "William", "age": 80, "username": None, "email": None},
        {
            "name": None,
            "age": None,
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]


def test_transform_suffixed_self_join_transform_lf() -> None:
    """
    Test a self-join (on the same shape) sandwiched by transforms, to make sure suffixing works and
    transforms on the resultant shapes are typed well
    """
    people = tp.LazyFrame(Person, _PEOPLE)
    accounts = tp.LazyFrame(Account, _ACCOUNTS)
    generated_accounts = people.transform(
        a := Account,
        people.s.name.str.to_lowercase().to(a.username),
        (people.s.name.str.to_lowercase() + "@example.net").to(a.email),
    ).suffix()
    joined = accounts.join(
        generated_accounts,
        accounts.s.username.on(generated_accounts.s(a.username)),
        how="outer",
    ).with_columns(accounts.s.username.coalesce(generated_accounts.s(a.username)))
    combined_emails = (
        tp.concat_list((a.email, generated_accounts.s(a.email)))
        .list.drop_nulls()
        .list.join(",")
    )
    if TYPE_CHECKING:
        static_assert(
            is_assignable_to(
                TypeOf[combined_emails],
                tp.MesoExpr[Intersection[Account, tp.Suffixed[Account]], str],
            )
        )

    transformed = joined.transform(a := Account, combined_emails.to(a.email))
    assert transformed.sort(a.username).collect().to_dicts() == [
        {"username": "douglas", "email": "douglas@adams.net,douglas@example.net"},
        {"username": "will", "email": "will@wilberforce.net"},
        {"username": "william", "email": "william@example.net"},
    ]


class Suit(Enum):
    HEARTS = "hearts"
    CLUBS = "clubs"
    SPADES = "spades"
    DIAMONDS = "diamonds"


class Card(tp.Shape):
    number = tp.dimension(tp.UINT_8)
    suit = tp.dimension(Suit)


def test_str_enum() -> None:
    if TYPE_CHECKING:
        static_assert(
            is_assignable_to(tp.BoundDimension[Card, Suit], TypeOf[Card.suit])
        )
        static_assert(
            is_assignable_to(
                TypeOf[Card.suit.set], Callable[[Suit], tp.expr.Initializer[Card, Suit]]
            )
        )

    cards = tp.DataFrame(
        Card,
        (
            tp.Entry.of(Card.number.set(10), Card.suit.set(Suit.HEARTS)),
            tp.Entry.of(Card.number.set(2), Card.suit.set(Suit.CLUBS)),
        ),
    )

    if TYPE_CHECKING:
        static_assert(is_assignable_to(TypeOf[cards[Card.suit]], tp.Series[Suit]))
        static_assert(is_assignable_to(TypeOf[cards[Card.suit].to_list()], list[str]))

    assert cards[Card.suit].to_list() == ["hearts", "clubs"]

    row = next(cards.iter_rows())
    if TYPE_CHECKING:
        static_assert(is_assignable_to(TypeOf[row[Card.suit]], str))
    assert row[Card.suit] == "hearts"


class Result(Enum):
    GOLD = 1
    SILVER = 2
    BRONZE = 3


class Player(tp.Shape):
    name = tp.dimension(str)
    result = tp.dimension(Result)


def test_int_enum() -> None:
    if TYPE_CHECKING:
        static_assert(
            is_assignable_to(tp.BoundDimension[Player, Result], TypeOf[Player.result])
        )
        static_assert(
            is_assignable_to(
                TypeOf[Player.result.set],
                Callable[[Result], tp.expr.Initializer[Player, Result]],
            )
        )

    players = tp.DataFrame(
        Player,
        (
            tp.Entry.of(Player.name.set("Marry Oh"), Player.result.set(Result.BRONZE)),
            tp.Entry.of(Player.name.set("Louisie"), Player.result.set(Result.SILVER)),
        ),
    )

    if TYPE_CHECKING:
        static_assert(
            is_assignable_to(TypeOf[players[Player.result]], tp.Series[Result])
        )
        static_assert(
            is_assignable_to(TypeOf[players[Player.result]], Iterable[EnumOf[int]])
        )
        static_assert(
            is_assignable_to(TypeOf[players[Player.result].to_list()], list[int])
        )
        static_assert(
            is_assignable_to(TypeOf[players[Player.result].first()], int | None)
        )

    assert players[Player.result].to_list() == [3, 2]

    row = next(players.iter_rows())
    assert row[Player.result] == 3


class PersonAgeStats(tp.Shape):
    age_group = tp.dimension(int)
    names = tp.dimension(tp.list_of(str))


@data_and_lazy
def test_group_by_agg[F: _Frame](cls: F) -> None:
    df = cls(
        Person,
        (
            Person.age.set_all([25, 22, 35, 24, 35, 36]),
            Person.name.set_all(["Alice", "Bob", "Charles", "David", "Eve", "Fred"]),
        ),
    )
    assert df.group_by(Person.age // 10).agg(Person.name.str.join(", ")).sort(
        Person.age
    ).collect().to_dicts() == [
        {"age": 2, "name": "Alice, Bob, David"},
        {"age": 3, "name": "Charles, Eve, Fred"},
    ]
    assert df.group_by_transform(
        PersonAgeStats, (Person.age // 10).to(PersonAgeStats.age_group)
    ).agg(Person.name.agg().to(PersonAgeStats.names)).sort(
        PersonAgeStats.age_group
    ).collect().to_dicts() == [
        {"age_group": 2, "names": ["Alice", "Bob", "David"]},
        {"age_group": 3, "names": ["Charles", "Eve", "Fred"]},
    ]


def test_alias() -> None:
    df = tp.LazyFrame(
        Person,
        (
            Person.age.set_all([25, 22, 35, 24, 35, 36]),
            Person.name.set_all(["Alice", "Bob", "Charles", "David", "Eve", "Fred"]),
        ),
    )
    df2 = df.with_columns((df.s.age + 1).alias("next_year_age"))
    assert df2[df2.s.next_year_age].max() == 37
    if TYPE_CHECKING:
        static_assert(
            is_assignable_to(
                TypeOf[df2],
                tp.LazyFrame[
                    Intersection[
                        Person, tp.expr.AliasShape[Literal["next_year_age"], int]
                    ]
                ],
            )
        )
    df3 = df.with_columns(
        tp.when(df.s.age < 30)
        .then(Suit.SPADES)
        .otherwise(Suit.CLUBS)
        .alias("assigned_suit"),
        df.s.name.str.head(3).alias("nickname"),
    )
    assert df3[df3.s.assigned_suit].to_list() == [
        "spades",
        "spades",
        "clubs",
        "spades",
        "clubs",
        "clubs",
    ]
    assert df3[df3.s.nickname].to_list() == ["Ali", "Bob", "Cha", "Dav", "Eve", "Fre"]
    if TYPE_CHECKING:
        static_assert(
            is_equivalent_to(
                TypeOf[df3],
                tp.LazyFrame[
                    Intersection[
                        Person,
                        tp.expr.AliasShape[Literal["assigned_suit"], Suit],
                        tp.expr.AliasShape[Literal["nickname"], str],
                    ]
                ],
            )
        )


def test_join_df_nulls_equal() -> None:
    people = tp.DataFrame(
        Person,
        (
            tp.Entry.of(Person.name.set("Douglas"), Person.age.set(42)),
            tp.Entry.of(Person.name.set_or_null(None), Person.age.set(80)),
        ),
    )
    accounts = tp.DataFrame(
        Account,
        (
            tp.Entry.of(
                Account.username.set("douglas"), Account.email.set("douglas@adams.net")
            ),
            tp.Entry.of(
                Account.username.set_or_null(None),
                Account.email.set("will@wilberforce.net"),
            ),
        ),
    )
    lowered_name: Expr[Person, Person, str] = Person.name.str.to_lowercase()
    nulls_equal = people.join(
        accounts, lowered_name.on(Account.username), nulls_equal=True
    )
    assert nulls_equal.sort(Person.name).to_dicts() == [
        {
            "name": None,
            "age": 80,
            "username": None,
            "email": "will@wilberforce.net",
        },
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
    ]
    nulls_unequal = people.join(
        accounts, lowered_name.on(Account.username), nulls_equal=False
    )
    assert nulls_unequal.sort(Person.name).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "email": "douglas@adams.net",
        },
    ]


def test_concat_df() -> None:
    assert tp.DataFrame.concat(
        Person,
        (
            tp.DataFrame(Person, _PEOPLE),
            tp.DataFrame(Person, _PEOPLE).with_columns(Person.name + "2"),
        ),
    ).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
        },
        {
            "name": "William",
            "age": 80,
        },
        {
            "name": "Douglas2",
            "age": 42,
        },
        {
            "name": "William2",
            "age": 80,
        },
    ]


def test_antijoin_df() -> None:
    people = tp.DataFrame(Person, _PEOPLE)
    accounts = tp.DataFrame(Account, _ACCOUNTS)
    lowered_name = Person.name.str.to_lowercase()
    anti_people = people.join(
        accounts, lowered_name.on(accounts.s.username), how="anti"
    )
    anti_accounts = accounts.join(
        people, accounts.s.username.on(lowered_name), how="anti"
    )
    assert anti_people.sort(Person.name).to_dicts() == [
        {"name": "William", "age": 80},
    ]
    assert anti_accounts.sort(anti_accounts.s.username).to_dicts() == [
        {
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]


@data_and_lazy
def test_default_sort[F: _Frame](cls: F) -> None:
    """
    By default, sorting should be in order of definition, since this is well defined and controlled
    by the user in Typol
    """
    people = cls(Person, _PEOPLE)
    accounts = cls(Account, _ACCOUNTS)
    people = people.vstack(people.with_columns(people.s.age + 1))
    assert people.sort().collect().to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
        },
        {
            "name": "Douglas",
            "age": 43,
        },
        {
            "name": "William",
            "age": 80,
        },
        {
            "name": "William",
            "age": 81,
        },
    ]
    assert accounts.sort().collect().to_dicts() == [
        {
            "username": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "username": "will",
            "email": "will@wilberforce.net",
        },
    ]
    assert people.sort(descending=True).collect().to_dicts() == [
        {
            "name": "William",
            "age": 81,
        },
        {
            "name": "William",
            "age": 80,
        },
        {
            "name": "Douglas",
            "age": 43,
        },
        {
            "name": "Douglas",
            "age": 42,
        },
    ]
    assert accounts.sort(descending=True).collect().to_dicts() == [
        {
            "username": "will",
            "email": "will@wilberforce.net",
        },
        {
            "username": "douglas",
            "email": "douglas@adams.net",
        },
    ]


@data_and_lazy
def test_reverse[F: _Frame](cls: F) -> None:
    people = cls(Person, _PEOPLE)
    accounts = cls(Account, _ACCOUNTS)
    assert people.reverse().collect().to_dicts() == [
        {"name": "William", "age": 80},
        {"name": "Douglas", "age": 42},
    ]
    assert accounts.reverse().collect().to_dicts() == [
        {"username": "will", "email": "will@wilberforce.net"},
        {"username": "douglas", "email": "douglas@adams.net"},
    ]


def test_join_df_with_aliases() -> None:
    people = tp.DataFrame(Person, _PEOPLE)
    accounts = tp.DataFrame(Account, _ACCOUNTS)
    people = people.with_columns(people.s.name.str.to_lowercase().alias("lowered"))
    left = people.join(accounts, people.s.lowered.on(Account.username), how="left")
    right = people.join(accounts, people.s.lowered.on(Account.username), how="right")
    inner = people.join(accounts, people.s.lowered.on(Account.username), how="inner")
    outer = people.join(accounts, people.s.lowered.on(Account.username), how="outer")
    assert left.sort(Person.name).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "lowered": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": "William",
            "age": 80,
            "lowered": "william",
            "username": "william",
            "email": None,
        },
    ]
    assert right.sort(right.s.username).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "lowered": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "lowered": "will",
            "email": "will@wilberforce.net",
        },
    ]
    assert inner.to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "lowered": "douglas",
            "email": "douglas@adams.net",
        }
    ]
    assert outer.sort(Person.name, Account.username, nulls_last=True).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "lowered": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": "William",
            "age": 80,
            "username": None,
            "lowered": "william",
            "email": None,
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "lowered": None,
            "email": "will@wilberforce.net",
        },
    ]


def test_join_df_on_projection() -> None:
    people = tp.DataFrame(Person, _PEOPLE)
    accounts = tp.DataFrame(Account, _ACCOUNTS)
    people = people.with_columns(
        alias := people.s.name.str.to_lowercase().alias("matchable")
    )
    accounts = accounts.with_columns(
        accounts.s.username.str.to_lowercase().alias("matchable")
    )

    matchable_shape = alias.construct_shape(people.dataframe)
    matchable_projection = tp.projection(matchable_shape)

    left = people.join(accounts, matchable_projection, how="left")
    right = people.join(accounts, matchable_projection, how="right")
    inner = people.join(accounts, matchable_projection, how="inner")
    outer = people.join(accounts, matchable_projection, how="outer")
    assert left.sort(Person.name).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "matchable": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": "William",
            "age": 80,
            "matchable": "william",
            "username": None,
            "email": None,
        },
    ]
    assert right.sort(right.s.username).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "matchable": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "matchable": "will",
            "email": "will@wilberforce.net",
        },
    ]
    assert inner.to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "matchable": "douglas",
            "email": "douglas@adams.net",
        }
    ]
    assert outer.sort(Person.name, Account.username, nulls_last=True).to_dicts() == [
        {
            "name": "Douglas",
            "age": 42,
            "username": "douglas",
            "matchable": "douglas",
            "email": "douglas@adams.net",
        },
        {
            "name": "William",
            "age": 80,
            "username": None,
            "matchable": "william",
            "email": None,
        },
        {
            "name": None,
            "age": None,
            "username": "will",
            "matchable": None,
            "email": "will@wilberforce.net",
        },
    ]


@data_and_lazy
def test_with_row_index[F: _Frame](cls: F) -> None:
    frame: tp.DataFrame[Person] | tp.LazyFrame[Person] = cls(Person, _PEOPLE)
    indexed = frame.with_row_index().collect()
    assert indexed.to_dicts() == [
        {"name": "Douglas", "age": 42, "index": 0},
        {"name": "William", "age": 80, "index": 1},
    ]
    assert indexed[indexed.s.index].to_list() == [0, 1]
    if TYPE_CHECKING:
        static_assert(
            is_equivalent_to(TypeOf[indexed[indexed.s.index]], tp.Series[int])
        )


@data_and_lazy
def test_with_custom_row_index[F: _Frame](cls: F) -> None:
    frame: tp.DataFrame[Person] | tp.LazyFrame[Person] = cls(Person, _PEOPLE)
    positioned = frame.with_row_index("position", 1).collect()
    assert positioned.to_dicts() == [
        {"name": "Douglas", "age": 42, "position": 1},
        {"name": "William", "age": 80, "position": 2},
    ]
    assert positioned[positioned.s.position].to_list() == [1, 2]
    if TYPE_CHECKING:
        static_assert(
            is_equivalent_to(TypeOf[positioned[positioned.s.position]], tp.Series[int])
        )
