import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

import typol as tp
from tests.examples.postage.store import (
    ClientRateQuery,
    CustomerRelationship,
    NegotiatedDiscount,
    PostageRateStore,
    ScheduleQuery,
    ScheduleRate,
)

FILES = Path(__file__).parent


class FileBasedPostageRateStore(PostageRateStore):
    def schedule(self, query: ScheduleQuery) -> tp.DataFrame[ScheduleRate]:
        filter = ScheduleRate.start.le(query.end) & ScheduleRate.end.ge(query.start)
        if query.destination is not None:
            filter &= ScheduleRate.destination.is_in(query.destination)
        if query.origin is not None:
            filter &= ScheduleRate.origin.is_in(query.origin)
        if query.relationship is not None:
            filter &= (
                ScheduleRate.relationship.is_null()
                | ScheduleRate.relationship.is_in(query.relationship)
            )

        return (
            tp.LazyFrame(ScheduleRate, pl.scan_csv(FILES / "schedule.csv"))
            .filter(filter)
            .collect()
        )

    def negotiated_discounts(
        self, query: ClientRateQuery
    ) -> tp.DataFrame[NegotiatedDiscount]:
        filter = NegotiatedDiscount.date.is_between(query.start, query.end)
        if query.destination is not None:
            filter &= NegotiatedDiscount.destination.is_in(query.destination)
        if query.origin is not None:
            filter &= NegotiatedDiscount.origin.is_in(query.origin)
        if query.client is not None:
            filter &= NegotiatedDiscount.client.is_in(query.client)

        return (
            tp.LazyFrame.scan_csv(NegotiatedDiscount, FILES / "discounts.csv")
            .filter(filter)
            .collect()
        )


def test_schedule() -> None:
    store = FileBasedPostageRateStore()
    assert store.schedule(
        ScheduleQuery(
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 2),
            destination=["FL"],
        )
    ).to_dicts() == [
        {
            "destination": "FL",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.05"),
            "relationship": "big",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "FL",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.10"),
            "relationship": None,
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "FL",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.20"),
            "relationship": "small",
            "start": datetime.date(2026, 1, 1),
        },
    ]
    assert store.schedule(
        ScheduleQuery(
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 1),
            origin=["UK"],
            destination=["CA"],
        )
    ).to_dicts() == [
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.10"),
            "relationship": "big",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.15"),
            "relationship": "medium",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.3"),
            "relationship": None,
            "start": datetime.date(2026, 1, 1),
        },
    ]
    assert store.schedule(
        ScheduleQuery(
            start=datetime.date(2025, 1, 1),
            end=datetime.date(2026, 1, 2),
            relationship=[CustomerRelationship.MEDIUM],
            destination=["CA"],
        )
    ).to_dicts() == [
        {
            "destination": "CA",
            "end": datetime.date(2025, 12, 31),
            "origin": "UK",
            "rate": Decimal("0.15000000"),
            "relationship": "medium",
            "start": datetime.date(2025, 12, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2025, 12, 31),
            "origin": "US",
            "rate": Decimal("0.03000000"),
            "relationship": "medium",
            "start": datetime.date(2025, 12, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2025, 12, 31),
            "origin": "UK",
            "rate": Decimal("0.30"),
            "relationship": None,
            "start": datetime.date(2025, 12, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2025, 12, 31),
            "origin": "US",
            "rate": Decimal("0.05"),
            "relationship": None,
            "start": datetime.date(2025, 12, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.15000000"),
            "relationship": "medium",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "US",
            "rate": Decimal("0.03000000"),
            "relationship": "medium",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "UK",
            "rate": Decimal("0.30"),
            "relationship": None,
            "start": datetime.date(2026, 1, 1),
        },
        {
            "destination": "CA",
            "end": datetime.date(2026, 1, 31),
            "origin": "US",
            "rate": Decimal("0.05"),
            "relationship": None,
            "start": datetime.date(2026, 1, 1),
        },
    ]


def test_negotiated_discounts() -> None:
    store = FileBasedPostageRateStore()
    assert store.negotiated_discounts(
        ClientRateQuery(
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 2),
            destination=["FL"],
        )
    ).to_dicts() == [
        {
            "client": "Express Logistics",
            "date": datetime.date(2026, 1, 1),
            "destination": "FL",
            "discount": Decimal("0.05000000"),
            "origin": "UK",
        }
    ]
    assert store.negotiated_discounts(
        ClientRateQuery(
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 2),
            client=["Transatlantic Export Ltd"],
        )
    ).to_dicts() == [
        {
            "client": "Transatlantic Export Ltd",
            "date": datetime.date(2026, 1, 2),
            "destination": "CA",
            "discount": Decimal("0.10000000"),
            "origin": "UK",
        }
    ]


def test_rates() -> None:
    store = FileBasedPostageRateStore()
    assert store.rates(
        ClientRateQuery(
            start=datetime.date(2026, 1, 1),
            end=datetime.date(2026, 1, 2),
            destination=["FL"],
            relationship=[CustomerRelationship.BIG],
        )
    ).to_dicts() == [
        {
            "client": None,
            "destination": "FL",
            "end": datetime.date(2026, 1, 1),
            "origin": "UK",
            "rate": Decimal("0.05000000"),
            "relationship": "big",
            "start": datetime.date(2026, 1, 1),
        },
        {
            "client": None,
            "destination": "FL",
            "end": datetime.date(2026, 1, 2),
            "origin": "UK",
            "rate": Decimal("0.05000000"),
            "relationship": "big",
            "start": datetime.date(2026, 1, 2),
        },
        {
            "client": "Express Logistics",
            "destination": "FL",
            "end": datetime.date(2026, 1, 1),
            "origin": "UK",
            "rate": Decimal("0.04750000"),
            "relationship": "big",
            "start": datetime.date(2026, 1, 1),
        },
    ]
    assert store.rates(
        ClientRateQuery(
            start=datetime.date(2026, 1, 2),
            end=datetime.date(2026, 1, 2),
            client=["Transatlantic Export Ltd"],
            relationship=[CustomerRelationship.SMALL],
            destination=["CA"],
        )
    ).to_dicts() == [
        {
            "client": None,
            "destination": "CA",
            "end": datetime.date(2026, 1, 2),
            "origin": "UK",
            "rate": Decimal("0.30000000"),
            "relationship": None,
            "start": datetime.date(2026, 1, 2),
        },
        {
            "client": None,
            "destination": "CA",
            "end": datetime.date(2026, 1, 2),
            "origin": "US",
            "rate": Decimal("0.05000000"),
            "relationship": None,
            "start": datetime.date(2026, 1, 2),
        },
        {
            "client": "Transatlantic Export Ltd",
            "destination": "CA",
            "end": datetime.date(2026, 1, 2),
            "origin": "UK",
            "rate": Decimal("0.27000000"),
            "relationship": None,
            "start": datetime.date(2026, 1, 2),
        },
    ]
