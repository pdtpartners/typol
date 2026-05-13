from __future__ import annotations

import dataclasses
import datetime
from abc import ABC, abstractmethod
from collections.abc import Collection
from enum import Enum
from typing import TYPE_CHECKING, Any

import typol as tp

if TYPE_CHECKING:
    pass


class CustomerRelationship(Enum):
    BIG = "big"
    MEDIUM = "medium"
    SMALL = "small"
    INDIVIDUAL = "individual"
    COMPETITOR = "competitor"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ScheduleQuery:
    start: datetime.date
    end: datetime.date
    origin: Collection[str] | None = None
    destination: Collection[str] | None = None
    relationship: Collection[CustomerRelationship] | None = None

    def to_client_rate_query(
        self, client: Collection[str] | None = None
    ) -> ClientRateQuery:
        return ClientRateQuery(**dataclasses.asdict(self), client=client)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ClientRateQuery(ScheduleQuery):
    """
    Not queryable in its own right, but useful for constructing a hot rate query before you know
    whether it's an Actual or Indicative query
    """

    client: Collection[str] | None = None


def _explode_nulls_over[S: tp.Shape, E: Enum](
    dimension: tp.BoundDimension[S, E], enum: Collection[E], *key: tp.ExoExpr[S, Any]
) -> tp.Explosion[S, S, E]:
    """
    Explode any rows with null values into all the remaining values that haven't been explcitly
    set, over the search key. In the following example, one row is defined with a `null`
    customer relationship:

    +--------+-------------+--------------+------+
    | origin | destination | relationship | rate |
    +--------+-------------+--------------+------+
    | UK     | DK          | big          |    5 |
    | UK     | DK          | medium       |   10 |
    | UK     | DK          | null         |   15 |
    +--------+-------------+--------------+------+

    This `null` row would be exploded to all remaining customer relationships:

    +--------+-------------+--------------+------+
    | origin | destination | relationship | rate |
    +--------+-------------+--------------+------+
    | UK     | DK          | big          |    5 |
    | UK     | DK          | medium       |   10 |
    | UK     | DK          | small        |   15 |
    | UK     | DK          | individual   |   15 |
    | UK     | DK          | competitor   |   15 |
    +--------+-------------+--------------+------+
    """
    # Unfortunately, some of the set difference and over logic doesn't seem to work well for enums,
    # getting caught in various string and integer representations of categorical. Instead, we
    # normalise to the value representation so we can do normal operations, and then cast back at
    # the end
    ty: tp.Type = tp.UINT_8 if dimension.ty.pl_ty.is_integer() else tp.STRING
    expr = dimension.cast_out(ty)

    base = (
        tp.when(expr.is_not_null())
        # If we have a set value for this row, just put it into a list
        .then(tp.concat_list(expr.cast_out(ty)))
    )

    remaining_values = (
        tp.lit(list(enum))
        .cast_out(tp.list_of(ty))
        .list.set_difference(expr.over(*key, mapping_strategy="join"))
    )

    # If we don't have a set value for this row, check if there are any conflicts (i.e.
    # if there are any rows with set values for this key)
    others_in_key_have_values: tp.Expr[S, tp.expr.NoShape, bool] = (
        expr.count().over(*key, mapping_strategy="join").gt(0)
    )
    null_exploded = base.when(others_in_key_have_values).then(
        # If there are, then we should expand this to all remaining values
        remaining_values
        # Otherwise this is left null, as we can freely match on any still without causing
        # conflicts
    )
    return null_exploded.cast_out(tp.list_of(dimension.ty)).list.explode_to(dimension)


DECIMAL = tp.decimal(38, 8)


class ScheduleRate(tp.Shape):
    start = tp.dimension(datetime.date)
    end = tp.dimension(datetime.date)
    origin = tp.dimension(str)
    destination = tp.dimension(str)
    relationship = tp.dimension(CustomerRelationship)
    rate = tp.dimension(DECIMAL)


class NegotiatedDiscount(tp.Shape):
    date = tp.dimension(datetime.date)
    origin = tp.dimension(str)
    destination = tp.dimension(str)
    client = tp.dimension(str)
    discount = tp.dimension(DECIMAL)


class PostageRate(ScheduleRate):
    client = tp.dimension(str)


class PostageRateStore(ABC):
    @abstractmethod
    def schedule(self, query: ScheduleQuery) -> tp.DataFrame[ScheduleRate]: ...

    @abstractmethod
    def negotiated_discounts(
        self, query: ClientRateQuery
    ) -> tp.DataFrame[NegotiatedDiscount]: ...

    def rates(self, query: ClientRateQuery) -> tp.DataFrame[PostageRate]:
        """
        Return a full rates table for a given query, combining the schedule and negotiated
        discounts
        """
        schedule = self.schedule(query).lazy()
        # Expand the schedule to be one line for each day of the week, to pair against
        # benchmarks whilst only including days that this rate is valid for
        schedule = schedule.explode(
            tp.date_ranges(
                tp.max(query.start, schedule.s.start), tp.min(query.end, schedule.s.end)
            ).list.explode_to(schedule.s.end)
        ).with_columns(schedule.s.end.to(schedule.s.start))

        # Some rates don't have specific relationship rates, expand these to apply to all those
        # without rates
        schedule = schedule.explode(
            _explode_nulls_over(
                schedule.s.relationship,
                CustomerRelationship,
                schedule.s.destination,
                schedule.s.origin,
                schedule.s.end,
            )
        )
        if query.relationship:
            schedule = schedule.filter(
                schedule.s.relationship.is_null()
                | schedule.s.relationship.is_in(query.relationship)
            )

        discounts = self.negotiated_discounts(query).lazy()

        discount_rates = discounts.join(
            schedule,
            discounts.s.date.on(schedule.s.end),
            (ScheduleRate & NegotiatedDiscount).origin.on(),
            (ScheduleRate & NegotiatedDiscount).destination.on(),
        ).transform(
            PostageRate,
            (ScheduleRate.rate * (tp.lit(1) - NegotiatedDiscount.discount)).to(
                PostageRate.rate
            ),
        )

        return (
            schedule.transform(PostageRate, PostageRate.client.null())
            .vstack(discount_rates)
            .collect()
        )
