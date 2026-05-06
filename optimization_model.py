#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from typing import Dict

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


def setup_logger():
    logger = logging.getLogger("optimization_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


W_CANCELLATION = 1.0
W_DELAY = 5.0
W_UNSCHEDULED = 1000.0
W_CAPACITY_SLACK = 100.0


def build_model_inputs_from_clean_data(df_clean: pd.DataFrame,
                                       capacity_df_clean: pd.DataFrame) -> Dict:
    df = df_clean.copy()
    cap = capacity_df_clean.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["last_maintenance_date"] = pd.to_datetime(df["last_maintenance_date"])
    cap["date"] = pd.to_datetime(cap["date"])

    df = df.sort_values(["aircraft_id", "date"]).reset_index(drop=True)
    cap = cap.sort_values("date").reset_index(drop=True)

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

    flights = {}
    flight_hours = {}

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

    due_day = {}

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


def build_model(model_inputs: Dict) -> pyo.ConcreteModel:
    A = model_inputs["aircraft"]
    T = model_inputs["days"]
    F = model_inputs["flights"]
    D = model_inputs["maint_duration"]
    DUE = model_inputs["due_day"]
    REQ = set(model_inputs["requires_maint"])
    CAP = model_inputs["capacity"]

    Tmax = max(T)

    m = pyo.ConcreteModel(name="AircraftMaintenanceScheduling")

    m.A = pyo.Set(initialize=A, ordered=True)
    m.T = pyo.Set(initialize=T, ordered=True)

    m.flights = pyo.Param(
        m.A,
        m.T,
        initialize=F,
        default=0,
        within=pyo.NonNegativeIntegers
    )

    m.maint_duration = pyo.Param(
        m.A,
        initialize=D,
        within=pyo.PositiveIntegers
    )

    m.due_day = pyo.Param(
        m.A,
        initialize=DUE,
        within=pyo.PositiveIntegers
    )

    m.capacity = pyo.Param(
        m.T,
        initialize=CAP,
        within=pyo.NonNegativeIntegers
    )

    m.x = pyo.Var(m.A, m.T, within=pyo.Binary)
    m.in_maint = pyo.Var(m.A, m.T, within=pyo.Binary)
    m.cancelled = pyo.Var(m.A, m.T, within=pyo.NonNegativeReals)
    m.delay = pyo.Var(m.A, within=pyo.NonNegativeReals)
    m.cap_slack = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.unsched = pyo.Var(m.A, within=pyo.Binary)

    def feasible_start_rule(m, a, t):
        if t + m.maint_duration[a] - 1 > Tmax:
            return m.x[a, t] == 0
        return pyo.Constraint.Skip

    m.feasible_start = pyo.Constraint(m.A, m.T, rule=feasible_start_rule)

    def at_most_once_rule(m, a):
        return sum(m.x[a, t] for t in m.T) <= 1

    m.at_most_once = pyo.Constraint(m.A, rule=at_most_once_rule)

    def coverage_rule(m, a):
        if a in REQ:
            return sum(m.x[a, t] for t in m.T) + m.unsched[a] >= 1
        return pyo.Constraint.Skip

    m.coverage = pyo.Constraint(m.A, rule=coverage_rule)

    def in_maint_rule(m, a, t):
        d = m.maint_duration[a]
        starts = [
            s for s in m.T
            if (s <= t) and (s >= t - d + 1)
        ]
        return m.in_maint[a, t] == sum(m.x[a, s] for s in starts)

    m.in_maint_link = pyo.Constraint(m.A, m.T, rule=in_maint_rule)

    def capacity_rule(m, t):
        return sum(m.in_maint[a, t] for a in m.A) <= m.capacity[t] + m.cap_slack[t]

    m.capacity_con = pyo.Constraint(m.T, rule=capacity_rule)

    def cancellation_rule(m, a, t):
        return m.cancelled[a, t] >= m.flights[a, t] * m.in_maint[a, t]

    m.cancellation_con = pyo.Constraint(m.A, m.T, rule=cancellation_rule)

    def delay_rule(m, a):
        late_terms = [
            (t - m.due_day[a]) * m.x[a, t]
            for t in m.T
            if t > m.due_day[a]
        ]

        if not late_terms:
            return m.delay[a] >= 0

        return m.delay[a] >= sum(late_terms)

    m.delay_con = pyo.Constraint(m.A, rule=delay_rule)

    def objective_rule(m):
        cancel_cost = W_CANCELLATION * sum(
            m.cancelled[a, t]
            for a in m.A
            for t in m.T
        )

        delay_cost = W_DELAY * sum(
            m.delay[a]
            for a in m.A
        )

        unsched_cost = W_UNSCHEDULED * sum(
            m.unsched[a]
            for a in m.A
        )

        slack_cost = W_CAPACITY_SLACK * sum(
            m.cap_slack[t]
            for t in m.T
        )

        return cancel_cost + delay_cost + unsched_cost + slack_cost

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m


def choose_available_solver():
    solver_candidates = ["glpk", "cbc", "highs"]

    for solver_name in solver_candidates:
        solver = SolverFactory(solver_name)

        if solver.available(exception_flag=False):
            print(f"Kullanılacak solver: {solver_name}")
            return solver_name

    raise RuntimeError(
        "Hiçbir solver bulunamadı. GLPK kurulu olmayabilir.\n"
        "Mac için Terminal'de şunu deneyebilirsin:\n"
        "brew install glpk\n\n"
        "Conda kullanıyorsan:\n"
        "conda install -c conda-forge glpk"
    )


def solve_model(model: pyo.ConcreteModel,
                model_inputs: Dict,
                solver_name: str = None,
                tee: bool = False,
                time_limit: int = None) -> Dict:
    if solver_name is None:
        solver_name = choose_available_solver()

    solver = SolverFactory(solver_name)

    if not solver.available(exception_flag=False):
        raise RuntimeError(f"{solver_name} solver mevcut değil.")

    if time_limit is not None and solver_name == "glpk":
        solver.options["tmlim"] = time_limit

    logger.info(f"Model {solver_name} ile çözülüyor...")

    results = solver.solve(model, tee=tee)

    status = results.solver.status
    term = results.solver.termination_condition

    print("\nSolver status:", status)
    print("Termination condition:", term)

    if status != SolverStatus.ok or term not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible
    ):
        logger.warning(f"Solver status={status}, termination={term}")

    schedule_df = extract_schedule(model, model_inputs)
    summary = extract_summary(model, schedule_df, status, term)

    logger.info(
        f"Çözüm tamamlandı | obj={summary['objective_value']:.2f} | "
        f"cancelled={summary['total_cancelled_flights']:.0f} | "
        f"delay_days={summary['total_delay_days']:.0f} | "
        f"unscheduled={summary['unscheduled_required']} | "
        f"scheduled={summary['number_of_scheduled_maintenances']}"
    )

    return {
        "schedule": schedule_df,
        "summary": summary,
        "model": model
    }


def extract_schedule(model: pyo.ConcreteModel,
                     model_inputs: Dict) -> pd.DataFrame:
    A = list(model.A)
    T = list(model.T)

    REQ = set(model_inputs["requires_maint"])
    date_index = model_inputs["date_index"]
    aircraft_type = model_inputs["aircraft_type"]
    base_airport = model_inputs["base_airport"]
    flight_intensity = model_inputs["flight_intensity"]
    due_status = model_inputs["due_status"]
    due_day = model_inputs["due_day"]
    flights = model_inputs["flights"]
    duration = model_inputs["maint_duration"]

    rows = []

    for a in A:
        start_day = None

        for t in T:
            if pyo.value(model.x[a, t]) is not None and pyo.value(model.x[a, t]) > 0.5:
                start_day = t
                break

        scheduled_flag = 1 if start_day is not None else 0
        unscheduled_flag = 1 if (a in REQ and start_day is None) else 0

        if start_day is not None:
            d = duration[a]
            end_day = start_day + d - 1
            delay_days = max(0, start_day - due_day[a])

            cancelled = int(round(sum(
                flights.get((a, t), 0)
                for t in range(start_day, end_day + 1)
            )))

            start_date = date_index[start_day].strftime("%Y-%m-%d")
            end_date = date_index[end_day].strftime("%Y-%m-%d")

        else:
            d = duration[a]
            end_day = None
            delay_days = 0
            cancelled = 0
            start_date = None
            end_date = None

        rows.append({
            "aircraft_id": a,
            "aircraft_type": aircraft_type[a],
            "base_airport": base_airport[a],
            "flight_intensity": flight_intensity[a],
            "due_status": due_status[a],
            "due_day": int(due_day[a]),
            "start_day": int(start_day) if start_day else None,
            "duration": int(d),
            "end_day": int(end_day) if end_day else None,
            "start_date": start_date,
            "end_date": end_date,
            "delay_days": int(delay_days),
            "cancelled_flights_during_maintenance": int(cancelled),
            "scheduled_flag": int(scheduled_flag),
            "unscheduled_flag": int(unscheduled_flag),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["scheduled_flag", "start_day", "unscheduled_flag", "aircraft_id"],
        ascending=[False, True, False, True],
        na_position="last"
    ).reset_index(drop=True)

    return df


def extract_summary(model: pyo.ConcreteModel,
                    schedule_df: pd.DataFrame,
                    status,
                    term) -> Dict:
    return {
        "objective_value": float(pyo.value(model.obj)),

        "total_cancelled_flights": float(sum(
            pyo.value(model.cancelled[a, t])
            for a in model.A
            for t in model.T
        )),

        "total_delay_days": float(sum(
            pyo.value(model.delay[a])
            for a in model.A
        )),

        "unscheduled_required": int(sum(
            round(pyo.value(model.unsched[a]))
            for a in model.A
        )),

        "capacity_overflow": float(sum(
            pyo.value(model.cap_slack[t])
            for t in model.T
        )),

        "number_of_scheduled_maintenances": int(
            schedule_df["scheduled_flag"].sum()
        ),

        "solver_status": str(status),
        "termination_condition": str(term),
    }


if __name__ == "__main__":

    print("\n--- OPTIMIZATION MODEL BAŞLADI ---\n")

    data_folder = "/Users/sude/.spyder-py3/data"

    aircraft_clean_path = os.path.join(data_folder, "aircraft_clean.csv")
    capacity_clean_path = os.path.join(data_folder, "capacity_clean.csv")

    print("Data klasörü:")
    print(data_folder)

    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Data klasörü bulunamadı: {data_folder}"
        )

    print("\nData klasörü içeriği:")
    print(os.listdir(data_folder))

    if not os.path.exists(aircraft_clean_path):
        raise FileNotFoundError(
            "aircraft_clean.csv bulunamadı. Önce preprocessing.py çalıştır.\n"
            f"Aranan dosya: {aircraft_clean_path}"
        )

    if not os.path.exists(capacity_clean_path):
        raise FileNotFoundError(
            "capacity_clean.csv bulunamadı. Önce preprocessing.py çalıştır.\n"
            f"Aranan dosya: {capacity_clean_path}"
        )

    print("\nTemizlenmiş veriler bulundu:")
    print("Aircraft clean:", aircraft_clean_path)
    print("Capacity clean:", capacity_clean_path)

    aircraft_clean = pd.read_csv(aircraft_clean_path)
    capacity_clean = pd.read_csv(capacity_clean_path)

    print("\nTemizlenmiş veri boyutları:")
    print("Aircraft clean shape:", aircraft_clean.shape)
    print("Capacity clean shape:", capacity_clean.shape)

    model_inputs = build_model_inputs_from_clean_data(
        aircraft_clean,
        capacity_clean
    )

    model = build_model(model_inputs)

    result = solve_model(
        model=model,
        model_inputs=model_inputs,
        solver_name=None,
        tee=False,
        time_limit=300
    )

    schedule_df = result["schedule"]
    summary = result["summary"]

    schedule_output_path = os.path.join(data_folder, "schedule_results.csv")
    summary_output_path = os.path.join(data_folder, "optimization_summary.csv")

    schedule_df.to_csv(schedule_output_path, index=False)

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    summary_df.to_csv(summary_output_path, index=False)

    print("\n--- OPTİMİZASYON SONUÇLARI ---")
    print(summary_df)

    print("\n--- BAKIM ÇİZELGESİ İLK 10 SATIR ---")
    print(schedule_df.head(10))

    print("\n--- DOSYALAR KAYDEDİLDİ ---")
    print(schedule_output_path)
    print(summary_output_path)

    print("\n--- OPTIMIZATION MODEL TAMAMLANDI ---\n")