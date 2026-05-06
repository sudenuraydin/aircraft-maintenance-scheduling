#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def setup_logger():
    logger = logging.getLogger("preprocessing_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


REQUIRED_AIRCRAFT_COLS = [
    "aircraft_id",
    "date",
    "scheduled_flights",
    "flight_hours",
    "last_maintenance_date",
    "days_since_last_maintenance",
    "max_days_between_maintenance",
    "maintenance_duration_days",
    "maintenance_required",
    "aircraft_type",
    "base_airport",
    "flight_intensity",
    "due_status",
]

REQUIRED_CAPACITY_COLS = [
    "date",
    "day_index",
    "maintenance_capacity",
]


def _classify_due_status(days_since: int, max_gap: int) -> str:
    if days_since > max_gap:
        return "overdue"
    if days_since >= max_gap - 7:
        return "due_soon"
    return "not_due"


def _check_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{name} veri setinde eksik kolonlar var: {missing}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aircraft dataset temizleniyor...")

    df = df.copy()
    _check_required_columns(df, REQUIRED_AIRCRAFT_COLS, "aircraft_data")

    initial_row_count = len(df)

    df = df.drop_duplicates(subset=["aircraft_id", "date"]).reset_index(drop=True)

    logger.info(f"{initial_row_count - len(df)} adet duplicate aircraft satırı silindi.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["last_maintenance_date"] = pd.to_datetime(
        df["last_maintenance_date"],
        errors="coerce"
    )

    total_missing = int(df.isna().sum().sum())

    if total_missing > 0:
        logger.info(f"{total_missing} adet eksik hücre bulundu. Dolduruluyor...")

    df["scheduled_flights"] = df["scheduled_flights"].fillna(0).astype(int)
    df["flight_hours"] = df["flight_hours"].fillna(0.0)

    negative_hours = df["flight_hours"] < 0

    if negative_hours.any():
        logger.warning(
            f"{int(negative_hours.sum())} adet negatif flight_hours değeri 0 yapıldı."
        )
        df.loc[negative_hours, "flight_hours"] = 0.0

    illogical_rows = df["last_maintenance_date"] > df["date"]

    if illogical_rows.any():
        logger.warning(
            f"{int(illogical_rows.sum())} adet mantıksız tarih satırı silindi."
        )
        df = df.loc[~illogical_rows].reset_index(drop=True)

    bad_gap = df["max_days_between_maintenance"] <= 0
    bad_duration = df["maintenance_duration_days"] <= 0

    if bad_gap.any() or bad_duration.any():
        logger.warning(
            f"{int((bad_gap | bad_duration).sum())} adet geçersiz bakım satırı silindi."
        )
        df = df.loc[~(bad_gap | bad_duration)].reset_index(drop=True)

    df["days_since_last_maintenance"] = (
        df["date"] - df["last_maintenance_date"]
    ).dt.days

    df["due_status"] = df.apply(
        lambda row: _classify_due_status(
            int(row["days_since_last_maintenance"]),
            int(row["max_days_between_maintenance"])
        ),
        axis=1
    )

    df["maintenance_required"] = df["due_status"].isin(
        ["due_soon", "overdue"]
    ).astype(int)

    logger.info(f"Aircraft dataset temizleme tamamlandı. Final shape: {df.shape}")

    return df


def clean_capacity_data(capacity_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Capacity dataset temizleniyor...")

    df = capacity_df.copy()
    _check_required_columns(df, REQUIRED_CAPACITY_COLS, "capacity_data")

    initial_row_count = len(df)

    df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    if len(df) < initial_row_count:
        logger.info(
            f"{initial_row_count - len(df)} adet duplicate capacity satırı silindi."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["maintenance_capacity"].isna().any():
        median_capacity = int(round(df["maintenance_capacity"].median(skipna=True)))
        number_of_missing = int(df["maintenance_capacity"].isna().sum())

        df["maintenance_capacity"] = df["maintenance_capacity"].fillna(median_capacity)

        logger.info(
            f"{number_of_missing} adet eksik capacity değeri median={median_capacity} ile dolduruldu."
        )

    df["maintenance_capacity"] = df["maintenance_capacity"].clip(lower=0).astype(int)

    df["day_index"] = np.arange(1, len(df) + 1)

    logger.info(f"Capacity dataset temizleme tamamlandı. Final shape: {df.shape}")

    return df


def build_model_inputs(
    df_clean: pd.DataFrame,
    capacity_df_clean: pd.DataFrame
) -> Dict:

    logger.info("Optimizasyon modeli için input yapıları oluşturuluyor...")

    df = df_clean.sort_values(["aircraft_id", "date"]).reset_index(drop=True)
    cap = capacity_df_clean.sort_values("date").reset_index(drop=True)

    days = cap["day_index"].astype(int).tolist()

    date_index = {
        int(day): pd.Timestamp(date)
        for day, date in zip(cap["day_index"], cap["date"])
    }

    idx_by_date = {
        pd.Timestamp(date): int(day)
        for day, date in zip(cap["day_index"], cap["date"])
    }

    horizon_start = pd.Timestamp(cap["date"].iloc[0])
    horizon_end = pd.Timestamp(cap["date"].iloc[-1])
    number_of_days = len(days)

    aircraft = sorted(df["aircraft_id"].unique().tolist())

    flights: Dict[Tuple[str, int], int] = {}
    flight_hours: Dict[Tuple[str, int], float] = {}

    for _, row in df.iterrows():
        current_date = pd.Timestamp(row["date"])

        if current_date not in idx_by_date:
            continue

        day = idx_by_date[current_date]
        aircraft_id = row["aircraft_id"]

        flights[(aircraft_id, day)] = int(row["scheduled_flights"])
        flight_hours[(aircraft_id, day)] = float(row["flight_hours"])

    first_rows = df.groupby("aircraft_id", as_index=False).first()

    maint_duration = {
        row.aircraft_id: int(row.maintenance_duration_days)
        for row in first_rows.itertuples()
    }

    aircraft_type = {
        row.aircraft_id: str(row.aircraft_type)
        for row in first_rows.itertuples()
    }

    base_airport = {
        row.aircraft_id: str(row.base_airport)
        for row in first_rows.itertuples()
    }

    flight_intensity = {
        row.aircraft_id: str(row.flight_intensity)
        for row in first_rows.itertuples()
    }

    due_status = {
        row.aircraft_id: str(row.due_status)
        for row in first_rows.itertuples()
    }

    due_day: Dict[str, int] = {}

    for row in first_rows.itertuples():
        deadline = pd.Timestamp(row.last_maintenance_date) + pd.Timedelta(
            days=int(row.max_days_between_maintenance)
        )

        if deadline < horizon_start:
            due_day[row.aircraft_id] = 1
        elif deadline > horizon_end:
            due_day[row.aircraft_id] = number_of_days
        else:
            due_day[row.aircraft_id] = (deadline - horizon_start).days + 1

    requires_status = (
        df.groupby("aircraft_id")["maintenance_required"].max().astype(int)
    )

    requires_maint = sorted(
        requires_status[requires_status == 1].index.tolist()
    )

    capacity = {
        int(row.day_index): int(row.maintenance_capacity)
        for row in cap.itertuples()
    }

    logger.info(
        f"Model inputları hazırlandı | aircraft={len(aircraft)} | "
        f"days={number_of_days} | maintenance_required={len(requires_maint)}"
    )

    return {
        "aircraft": aircraft,
        "days": days,
        "date_index": date_index,
        "flights": flights,
        "flight_hours": flight_hours,
        "maint_duration": maint_duration,
        "due_day": due_day,
        "requires_maint": requires_maint,
        "capacity": capacity,
        "aircraft_type": aircraft_type,
        "base_airport": base_airport,
        "flight_intensity": flight_intensity,
        "due_status": due_status,
    }


def find_file_by_keywords(folder_path, keywords):
    files = os.listdir(folder_path)

    csv_files = [
        file for file in files
        if file.lower().endswith(".csv")
    ]

    for file in csv_files:
        lower_file = file.lower()

        if all(keyword.lower() in lower_file for keyword in keywords):
            return os.path.join(folder_path, file)

    return None


if __name__ == "__main__":

    print("\n--- PREPROCESSING BAŞLADI ---\n")

    data_folder = "/Users/sude/.spyder-py3/data"

    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Data klasörü bulunamadı: {data_folder}"
        )

    print("Data klasörü bulundu:")
    print(data_folder)

    print("\nData klasörü içeriği:")
    print(os.listdir(data_folder))

    aircraft_path = find_file_by_keywords(data_folder, ["aircraft"])
    capacity_path = find_file_by_keywords(data_folder, ["capacity"])

    if aircraft_path is None:
        raise FileNotFoundError(
            "Aircraft verisi bulunamadı. "
            "Data klasöründe adında 'aircraft' geçen bir .csv dosyası olmalı."
        )

    if capacity_path is None:
        raise FileNotFoundError(
            "Capacity verisi bulunamadı. "
            "Data klasöründe adında 'capacity' geçen bir .csv dosyası olmalı."
        )

    print("\nBulunan dosyalar:")
    print("Aircraft data:", aircraft_path)
    print("Capacity data:", capacity_path)

    aircraft_df = pd.read_csv(aircraft_path)
    capacity_df = pd.read_csv(capacity_path)

    print("\nHam veri boyutları:")
    print("Aircraft data shape:", aircraft_df.shape)
    print("Capacity data shape:", capacity_df.shape)

    aircraft_clean = clean_dataset(aircraft_df)
    capacity_clean = clean_capacity_data(capacity_df)

    model_inputs = build_model_inputs(aircraft_clean, capacity_clean)

    aircraft_clean_path = os.path.join(data_folder, "aircraft_clean.csv")
    capacity_clean_path = os.path.join(data_folder, "capacity_clean.csv")

    aircraft_clean.to_csv(aircraft_clean_path, index=False)
    capacity_clean.to_csv(capacity_clean_path, index=False)

    print("\n--- TEMİZLENMİŞ DOSYALAR KAYDEDİLDİ ---")
    print(aircraft_clean_path)
    print(capacity_clean_path)

    print("\n--- MODEL INPUT ÖZETİ ---")
    print("Aircraft sayısı:", len(model_inputs["aircraft"]))
    print("Gün sayısı:", len(model_inputs["days"]))
    print("Bakım gerektiren uçak sayısı:", len(model_inputs["requires_maint"]))

    print("\n--- PREPROCESSING TAMAMLANDI ---\n")