#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


SCENARIOS: Dict[str, Dict[str, int]] = {
    "small": {"n_aircraft": 6, "horizon_days": 14, "base_hangar_capacity": 1},
    "medium": {"n_aircraft": 12, "horizon_days": 30, "base_hangar_capacity": 2},
    "large": {"n_aircraft": 25, "horizon_days": 45, "base_hangar_capacity": 3},
}

AIRCRAFT_TYPES = ["A320", "A321", "B737-800", "B737-MAX", "A330"]
BASE_AIRPORTS = ["IST", "ESB", "ADB", "AYT", "SAW"]
INTENSITY_FLIGHTS = {"light": 1, "normal": 2, "heavy": 4}


def classify_due_status(days_since, max_gap):
    if days_since > max_gap:
        return "overdue"
    if days_since >= max_gap - 7:
        return "due_soon"
    return "not_due"


def generate_capacity_profile(start_date, horizon_days, base_capacity, seed=0):
    rng = np.random.default_rng(seed)

    rows = []

    for d in range(horizon_days):
        date = start_date + timedelta(days=d)
        roll = float(rng.random())

        if roll < 0.30:
            cap = max(1, base_capacity - 1)
        elif roll < 0.40:
            cap = max(1, base_capacity - 1) if base_capacity > 1 else base_capacity
        elif roll < 0.80:
            cap = base_capacity
        elif roll < 0.95:
            cap = base_capacity + 1
        else:
            cap = base_capacity + 2

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_index": d + 1,
            "maintenance_capacity": int(cap),
        })

    return pd.DataFrame(rows)


def generate_synthetic_data(
    scenario="medium",
    seed=42,
    start_date="2026-01-01",
    aircraft_output_path=None,
    capacity_output_path=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(SCENARIOS)}")

    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)

    n_aircraft = cfg["n_aircraft"]
    horizon = cfg["horizon_days"]
    base_cap = cfg["base_hangar_capacity"]
    horizon_start = parse_date(start_date)

    print("Generating synthetic aircraft maintenance data...")
    print(f"Scenario: {scenario}")
    print(f"Aircraft number: {n_aircraft}")
    print(f"Horizon: {horizon} days")
    print(f"Start date: {start_date}")

    aircraft_attrs = []

    for i in range(n_aircraft):
        ac_id = f"TC-{1000 + i}"
        ac_type = str(rng.choice(AIRCRAFT_TYPES))
        base = str(rng.choice(BASE_AIRPORTS))
        max_gap = int(rng.choice([20, 25, 30, 35]))
        maint_dur = int(rng.choice([2, 3, 4]))
        intensity = str(rng.choice(["light", "normal", "heavy"], p=[0.25, 0.5, 0.25]))

        if i < n_aircraft // 2:
            days_since = int(rng.integers(max_gap - 5, max_gap + 3))
        else:
            days_since = int(rng.integers(0, max(1, max_gap - 8)))

        last_maint = horizon_start - timedelta(days=days_since)

        aircraft_attrs.append({
            "aircraft_id": ac_id,
            "aircraft_type": ac_type,
            "base_airport": base,
            "flight_intensity": intensity,
            "last_maintenance_date": last_maint,
            "max_days_between_maintenance": max_gap,
            "maintenance_duration_days": maint_dur,
        })

    records = []

    for ac in aircraft_attrs:
        base_flights = INTENSITY_FLIGHTS[ac["flight_intensity"]]

        for d in range(horizon):
            current_date = horizon_start + timedelta(days=d)
            days_since_last = (current_date - ac["last_maintenance_date"]).days

            scheduled_flights = max(0, int(rng.poisson(base_flights)))
            flight_hours = round(scheduled_flights * float(rng.uniform(2.0, 4.0)), 2)

            due_status = classify_due_status(
                days_since_last,
                ac["max_days_between_maintenance"]
            )

            maintenance_required = 1 if due_status in ("due_soon", "overdue") else 0

            records.append({
                "aircraft_id": ac["aircraft_id"],
                "date": current_date.strftime("%Y-%m-%d"),
                "scheduled_flights": scheduled_flights,
                "flight_hours": flight_hours,
                "last_maintenance_date": ac["last_maintenance_date"].strftime("%Y-%m-%d"),
                "days_since_last_maintenance": days_since_last,
                "max_days_between_maintenance": ac["max_days_between_maintenance"],
                "maintenance_duration_days": ac["maintenance_duration_days"],
                "maintenance_required": maintenance_required,
                "due_status": due_status,
                "aircraft_type": ac["aircraft_type"],
                "base_airport": ac["base_airport"],
                "flight_intensity": ac["flight_intensity"],
            })

    aircraft_df = pd.DataFrame(records)

    if len(aircraft_df) > 30:
        aircraft_df = pd.concat([aircraft_df, aircraft_df.iloc[[5]]], ignore_index=True)

        missing_indices = rng.integers(0, len(aircraft_df), 2)
        aircraft_df.loc[missing_indices, "flight_hours"] = np.nan

        negative_idx = int(rng.integers(0, len(aircraft_df)))
        aircraft_df.loc[negative_idx, "flight_hours"] = -3.0

        bad_idx = int(rng.integers(0, len(aircraft_df)))
        aircraft_df.loc[bad_idx, "last_maintenance_date"] = (
            horizon_start + timedelta(days=horizon + 5)
        ).strftime("%Y-%m-%d")

    capacity_df = generate_capacity_profile(
        start_date=horizon_start,
        horizon_days=horizon,
        base_capacity=base_cap,
        seed=seed
    )

    if aircraft_output_path:
        ensure_dir(os.path.dirname(aircraft_output_path))
        aircraft_df.to_csv(aircraft_output_path, index=False)

    if capacity_output_path:
        ensure_dir(os.path.dirname(capacity_output_path))
        capacity_df.to_csv(capacity_output_path, index=False)

    print("\nData generation completed.")
    print(f"Aircraft data shape: {aircraft_df.shape}")
    print(f"Capacity data shape: {capacity_df.shape}")

    return aircraft_df, capacity_df


aircraft_data, capacity_data = generate_synthetic_data(
    scenario="medium",
    seed=42,
    start_date="2026-01-01",
    aircraft_output_path=os.path.join("data", "synthetic_aircraft_data.csv"),
    capacity_output_path=os.path.join("data", "maintenance_capacity.csv")
)

print("\n--- Aircraft Data First 10 Rows ---")
print(aircraft_data.head(10))

print("\n--- Capacity Data First 10 Rows ---")
print(capacity_data.head(10))