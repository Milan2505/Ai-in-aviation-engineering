from __future__ import annotations

import csv
from pathlib import Path

from ai_in_aviation_engineering import (
    MaintenanceModelArtifacts,
    PriceModelArtifacts,
    load_data,
    predict_maintenance,
    predict_price,
    train_maintenance_model,
    train_price_model,
)


def _build_price_records() -> list[dict[str, object]]:
    return [
        {
            "route": "RIX-ORY",
            "days_before_departure": 10,
            "seat_load_factor": 0.7,
            "departure_hour": 8,
            "final_price_eur": 120.0,
        },
        {
            "route": "RIX-ORY",
            "days_before_departure": 5,
            "seat_load_factor": 0.75,
            "departure_hour": 12,
            "final_price_eur": 135.0,
        },
        {
            "route": "CDG-JFK",
            "days_before_departure": 3,
            "seat_load_factor": 0.9,
            "departure_hour": 18,
            "final_price_eur": 220.0,
        },
        {
            "route": "LHR-BCN",
            "days_before_departure": 7,
            "seat_load_factor": 0.65,
            "departure_hour": 6,
            "final_price_eur": 95.0,
        },
    ]


def _build_maintenance_records() -> list[dict[str, object]]:
    return [
        {
            "hours_since_last_check": 100,
            "cycles_since_last_check": 50,
            "avg_daily_flight_hours": 4.5,
            "aog_flag": 0,
        },
        {
            "hours_since_last_check": 250,
            "cycles_since_last_check": 120,
            "avg_daily_flight_hours": 5.2,
            "aog_flag": 1,
        },
        {
            "hours_since_last_check": 300,
            "cycles_since_last_check": 180,
            "avg_daily_flight_hours": 6.1,
            "aog_flag": 1,
        },
        {
            "hours_since_last_check": 120,
            "cycles_since_last_check": 60,
            "avg_daily_flight_hours": 3.9,
            "aog_flag": 0,
        },
    ]


def test_load_data_drops_missing_rows(tmp_path: Path) -> None:
    rows = _build_price_records()
    rows[1] = {**rows[1], "final_price_eur": ""}
    csv_path = tmp_path / "prices.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_data(csv_path)
    assert len(loaded) == len(rows) - 1
    assert all("final_price_eur" in record for record in loaded)


def test_train_price_model_returns_artifacts() -> None:
    artefacts = train_price_model(_build_price_records(), persist=False)
    assert isinstance(artefacts, PriceModelArtifacts)
    assert artefacts.route_categories

    prediction = predict_price(
        artefacts,
        days_before_departure=4,
        load_factor=0.8,
        departure_hour=14,
        route="RIX-ORY",
    )
    assert isinstance(prediction, float)


def test_price_artifacts_round_trip(tmp_path: Path) -> None:
    artefacts = train_price_model(_build_price_records(), persist=False)
    artefacts.save(tmp_path)

    restored = PriceModelArtifacts.load(tmp_path)
    assert restored.route_categories == artefacts.route_categories

    unknown_route_prediction = restored.predict(
        days_before_departure=6,
        seat_load_factor=0.83,
        departure_hour=9,
        route="OSL-RIX",
    )
    assert isinstance(unknown_route_prediction, float)


def test_train_maintenance_model_and_predict(tmp_path: Path) -> None:
    artefacts = train_maintenance_model(_build_maintenance_records(), persist=False)
    assert isinstance(artefacts, MaintenanceModelArtifacts)

    result = predict_maintenance(
        artefacts,
        hours_since_last_check=200,
        cycles_since_last_check=100,
        avg_daily_flight_hours=5.0,
    )
    assert result in (0, 1)

    artefacts.save(tmp_path)
    restored = MaintenanceModelArtifacts.load(tmp_path)
    assert isinstance(
        restored.predict(
            hours_since_last_check=150,
            cycles_since_last_check=80,
            avg_daily_flight_hours=4.2,
        ),
        int,
    )
