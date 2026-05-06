#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_logger():
    logger = logging.getLogger("visualization_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_gantt(schedule_df, output_path):
    aircraft_list = schedule_df["aircraft_id"].tolist()

    max_end_day = schedule_df["end_day"].dropna().max()

    if pd.isna(max_end_day):
        print("Hiç planlanmış bakım yok. Gantt grafiği oluşturulmadı.")
        return

    n_days = int(max_end_day)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(aircraft_list))))

    y_pos = {
        aircraft: i
        for i, aircraft in enumerate(aircraft_list)
    }

    for aircraft, y in y_pos.items():
        ax.barh(
            y,
            n_days,
            left=0.5,
            height=0.6,
            color="#f0f0f0",
            edgecolor="white"
        )

    scheduled = schedule_df[schedule_df["scheduled_flag"] == 1]

    for _, row in scheduled.iterrows():
        if row["delay_days"] > 0:
            bar_color = "#E74C3C"
        else:
            bar_color = "#2C7BE5"

        ax.barh(
            y_pos[row["aircraft_id"]],
            row["duration"],
            left=row["start_day"] - 0.5,
            height=0.6,
            color=bar_color,
            edgecolor="black",
            linewidth=0.6
        )

        ax.text(
            row["start_day"] - 0.5 + row["duration"] / 2,
            y_pos[row["aircraft_id"]],
            f"D{int(row['start_day'])}-D{int(row['end_day'])}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold"
        )

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xticks(range(1, n_days + 1))
    ax.set_xlabel("Planning Day")
    ax.set_ylabel("Aircraft")
    ax.set_title("Optimized Maintenance Schedule - Gantt Chart")
    ax.set_xlim(0.5, n_days + 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150)
    plt.show()

    logger.info(f"Gantt chart kaydedildi: {output_path}")


def plot_capacity_usage(schedule_df, capacity_df, output_path):
    capacity_df = capacity_df.copy()
    capacity_df["day_index"] = capacity_df["day_index"].astype(int)

    days = capacity_df["day_index"].tolist()
    capacity = capacity_df["maintenance_capacity"].astype(int).tolist()

    usage = []

    for day in days:
        count = 0

        for _, row in schedule_df.iterrows():
            if row["scheduled_flag"] == 1:
                start_day = int(row["start_day"])
                end_day = int(row["end_day"])

                if start_day <= day <= end_day:
                    count += 1

        usage.append(count)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        days,
        usage,
        color="#2C7BE5",
        alpha=0.85,
        label="Aircraft in maintenance"
    )

    ax.step(
        days,
        capacity,
        where="mid",
        color="#E74C3C",
        linewidth=2.5,
        label="Daily maintenance capacity"
    )

    ax.scatter(
        days,
        capacity,
        color="#E74C3C",
        s=30
    )

    ax.set_xlabel("Planning Day")
    ax.set_ylabel("Number of Aircraft")
    ax.set_title("Daily Hangar Utilization vs Maintenance Capacity")
    ax.set_xticks(days)
    ax.set_ylim(0, max(max(usage), max(capacity)) + 1.5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150)
    plt.show()

    logger.info(f"Capacity usage grafiği kaydedildi: {output_path}")


def plot_due_vs_start_day(schedule_df, output_path):
    df = schedule_df[schedule_df["scheduled_flag"] == 1].copy()

    if df.empty:
        print("Planlanmış bakım yok. Due vs Start grafiği oluşturulmadı.")
        return

    df = df.sort_values("due_day").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(df))))

    y = np.arange(len(df))

    ax.hlines(
        y,
        df["due_day"],
        df["start_day"],
        color="gray",
        alpha=0.5,
        linewidth=2
    )

    ax.scatter(
        df["due_day"],
        y,
        color="#E74C3C",
        s=70,
        label="Due day"
    )

    ax.scatter(
        df["start_day"],
        y,
        color="#2C7BE5",
        s=70,
        label="Scheduled start day"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["aircraft_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Planning Day")
    ax.set_title("Maintenance Due Day vs Scheduled Start Day")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150)
    plt.show()

    logger.info(f"Due vs start day grafiği kaydedildi: {output_path}")


def plot_cancelled_flights(schedule_df, output_path):
    df = schedule_df[schedule_df["scheduled_flag"] == 1].copy()

    if df.empty:
        print("Planlanmış bakım yok. Cancelled flights grafiği oluşturulmadı.")
        return

    df = df.sort_values(
        "cancelled_flights_during_maintenance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))

    ax.barh(
        df["aircraft_id"],
        df["cancelled_flights_during_maintenance"],
        color="#2C7BE5",
        edgecolor="black",
        linewidth=0.4
    )

    for i, value in enumerate(df["cancelled_flights_during_maintenance"]):
        ax.text(
            value,
            i,
            f" {int(value)}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel("Cancelled Flights During Maintenance")
    ax.set_title("Cancelled Flights by Aircraft")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150)
    plt.show()

    logger.info(f"Cancelled flights grafiği kaydedildi: {output_path}")


def print_summary(summary_df):
    print("\n--- OPTIMIZATION SUMMARY ---\n")

    for _, row in summary_df.iterrows():
        print(f"{row['metric']}: {row['value']}")


if __name__ == "__main__":

    print("\n--- VISUALIZATION BAŞLADI ---\n")

    data_folder = "/Users/sude/.spyder-py3/data"

    schedule_path = os.path.join(data_folder, "schedule_results.csv")
    summary_path = os.path.join(data_folder, "optimization_summary.csv")
    capacity_path = os.path.join(data_folder, "capacity_clean.csv")

    print("Data klasörü:")
    print(data_folder)

    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Data klasörü bulunamadı: {data_folder}"
        )

    print("\nData klasörü içeriği:")
    print(os.listdir(data_folder))

    if not os.path.exists(schedule_path):
        raise FileNotFoundError(
            "schedule_results.csv bulunamadı. Önce optimization_model.py çalıştır."
        )

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "optimization_summary.csv bulunamadı. Önce optimization_model.py çalıştır."
        )

    if not os.path.exists(capacity_path):
        raise FileNotFoundError(
            "capacity_clean.csv bulunamadı. Önce preprocessing.py çalıştır."
        )

    schedule_df = pd.read_csv(schedule_path)
    summary_df = pd.read_csv(summary_path)
    capacity_df = pd.read_csv(capacity_path)

    print("\nDosyalar başarıyla okundu.")
    print("Schedule shape:", schedule_df.shape)
    print("Summary shape:", summary_df.shape)
    print("Capacity shape:", capacity_df.shape)

    print_summary(summary_df)

    gantt_output = os.path.join(data_folder, "gantt_schedule.png")
    capacity_output = os.path.join(data_folder, "capacity_usage.png")
    due_output = os.path.join(data_folder, "due_vs_start_day.png")
    cancelled_output = os.path.join(data_folder, "cancelled_flights.png")

    plot_gantt(schedule_df, gantt_output)
    plot_capacity_usage(schedule_df, capacity_df, capacity_output)
    plot_due_vs_start_day(schedule_df, due_output)
    plot_cancelled_flights(schedule_df, cancelled_output)

    print("\n--- GRAFİKLER KAYDEDİLDİ ---")
    print(gantt_output)
    print(capacity_output)
    print(due_output)
    print(cancelled_output)

    print("\n--- VISUALIZATION TAMAMLANDI ---\n")