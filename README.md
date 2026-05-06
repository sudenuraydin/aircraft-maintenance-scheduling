# Aircraft Maintenance Scheduling with Mixed-Integer Linear Programming

This project develops a Python-based aircraft maintenance scheduling model for a short- to medium-term planning horizon. The aim is to decide when each aircraft should enter maintenance while considering flight schedules, maintenance requirements, maintenance duration, and daily hangar capacity.

The project is designed as an undergraduate Industrial Engineering portfolio project. It demonstrates synthetic data generation, data preprocessing, mixed-integer linear programming, scheduling logic, and result visualization.

---

## Project Overview

Aircraft must undergo maintenance within a maximum allowed time interval. However, if an aircraft is sent to maintenance while it has scheduled flights, those flights may need to be cancelled or reassigned. Therefore, maintenance planning requires a trade-off between operational continuity and maintenance compliance.

This project addresses the following question:

Which aircraft should start maintenance on which day in order to minimize flight cancellations, maintenance delay, unscheduled required maintenance, and capacity overflow?

---

## Problem Description

Given a fleet of aircraft, daily scheduled flights for each aircraft, flight hours, last maintenance dates, maximum allowed days between maintenance events, maintenance duration for each aircraft, and daily variable maintenance capacity, the model decides the maintenance start day for each aircraft.

The objective is to minimize a weighted penalty consisting of cancelled flights during maintenance, delay beyond the due maintenance day, required maintenance that cannot be scheduled, and capacity overflow.

---

## Repository Structure

```text
aircraft-maintenance-scheduling/
│
├── data_generation.py
├── preprocessing.py
├── optimization_model.py
├── visualization.py
├── data/
│   ├── synthetic_aircraft_data.csv
│   ├── maintenance_capacity.csv
│   ├── aircraft_clean.csv
│   ├── capacity_clean.csv
│   ├── schedule_results.csv
│   ├── optimization_summary.csv
│   ├── gantt_schedule.png
│   ├── capacity_usage.png
│   ├── due_vs_start_day.png
│   └── cancelled_flights.png
└── README.md
```

---

## Project Workflow

The project is executed in four main steps:

1. Generate synthetic aircraft and capacity data.
2. Clean and preprocess the generated data.
3. Build and solve the optimization model.
4. Visualize the resulting maintenance schedule.

The files should be run in the following order:

```text
1. data_generation.py
2. preprocessing.py
3. optimization_model.py
4. visualization.py
```

---

## Synthetic Data Generation

The file `data_generation.py` creates two datasets:

1. `synthetic_aircraft_data.csv`
2. `maintenance_capacity.csv`

The aircraft dataset is generated at the aircraft-day level. Each row represents one aircraft on one planning day.

Main columns in the aircraft dataset:

| Column | Description |
|---|---|
| `aircraft_id` | Unique aircraft identifier |
| `date` | Planning date |
| `scheduled_flights` | Number of scheduled flights on that day |
| `flight_hours` | Total flight hours on that day |
| `last_maintenance_date` | Previous maintenance date |
| `days_since_last_maintenance` | Days since last maintenance |
| `max_days_between_maintenance` | Maximum allowed maintenance interval |
| `maintenance_duration_days` | Required maintenance duration |
| `maintenance_required` | 1 if maintenance is required, 0 otherwise |
| `due_status` | Maintenance status: `not_due`, `due_soon`, or `overdue` |
| `aircraft_type` | Aircraft type |
| `base_airport` | Base airport |
| `flight_intensity` | Flight intensity: `light`, `normal`, or `heavy` |

The capacity dataset contains the daily maintenance capacity.

Main columns in the capacity dataset:

| Column | Description |
|---|---|
| `date` | Planning date |
| `day_index` | Planning day index |
| `maintenance_capacity` | Available maintenance capacity on that day |

The generated data includes different scenarios:

| Scenario | Aircraft Count | Planning Horizon | Base Capacity |
|---|---:|---:|---:|
| `small` | 6 | 14 days | 1 |
| `medium` | 12 | 30 days | 2 |
| `large` | 25 | 45 days | 3 |

The generated data also includes a small number of artificial data issues such as duplicate rows, missing flight hours, negative flight hours, and inconsistent maintenance dates. These are included to demonstrate the preprocessing step.

---

## Data Preprocessing

The file `preprocessing.py` cleans the generated datasets and prepares them for the optimization model.

The preprocessing step performs duplicate row removal, date format conversion, missing value handling, correction of negative flight hours, removal of rows with inconsistent maintenance dates, validation of maintenance duration and maintenance interval values, recalculation of `days_since_last_maintenance`, recalculation of `due_status`, recalculation of `maintenance_required`, and creation of model input dictionaries.

The cleaned files are saved as:

```text
aircraft_clean.csv
capacity_clean.csv
```

The preprocessing step also creates the data structures required by the Pyomo model, including aircraft set, planning day set, scheduled flight dictionary, maintenance duration dictionary, due day dictionary, required maintenance list, and daily capacity dictionary.

---

## Optimization Model

The file `optimization_model.py` builds and solves a Mixed-Integer Linear Programming model using Pyomo.

The model determines which aircraft starts maintenance on which day.

### Sets

| Set | Description |
|---|---|
| `A` | Set of aircraft |
| `T` | Set of planning days |
| `REQ` | Set of aircraft requiring maintenance |

### Parameters

| Parameter | Description |
|---|---|
| `flights[a,t]` | Scheduled flights of aircraft `a` on day `t` |
| `maint_duration[a]` | Maintenance duration of aircraft `a` |
| `due_day[a]` | Due day of aircraft `a` |
| `capacity[t]` | Maintenance capacity on day `t` |

### Decision Variables

| Variable | Domain | Description |
|---|---|---|
| `x[a,t]` | Binary | 1 if aircraft `a` starts maintenance on day `t` |
| `in_maint[a,t]` | Binary | 1 if aircraft `a` is in maintenance on day `t` |
| `unsched[a]` | Binary | 1 if required maintenance for aircraft `a` is not scheduled |
| `cancelled[a,t]` | Nonnegative | Cancelled flights of aircraft `a` on day `t` |
| `delay[a]` | Nonnegative | Delay days beyond the due day |
| `cap_slack[t]` | Nonnegative | Capacity overflow on day `t` |

---

## Constraints

The model includes the following constraints:

Maintenance must finish within the planning horizon. An aircraft cannot start maintenance so late that the maintenance duration exceeds the planning horizon.

Each aircraft can start maintenance at most once.

If an aircraft requires maintenance, it must either be scheduled or receive an unscheduled maintenance penalty.

If an aircraft starts maintenance on a given day, it remains in maintenance for its full maintenance duration.

The number of aircraft in maintenance on a given day should not exceed the available maintenance capacity. A slack variable is included to keep the model feasible when capacity is not sufficient.

If an aircraft is in maintenance on a day when it has scheduled flights, those flights are counted as cancelled or reassigned.

If an aircraft starts maintenance after its due day, the delay is penalized.

---

## Objective Function

The objective function minimizes the weighted sum of four penalty components:

```text
minimize:
cancellation penalty
+ delay penalty
+ unscheduled maintenance penalty
+ capacity overflow penalty
```

In mathematical form:

```text
minimize:
W_CANCELLATION * sum(cancelled[a,t])
+ W_DELAY * sum(delay[a])
+ W_UNSCHEDULED * sum(unsched[a])
+ W_CAPACITY_SLACK * sum(cap_slack[t])
```

The default penalty weights are:

| Penalty | Value |
|---|---:|
| `W_CANCELLATION` | 1 |
| `W_DELAY` | 5 |
| `W_UNSCHEDULED` | 1000 |
| `W_CAPACITY_SLACK` | 100 |

These values can be changed in `optimization_model.py`.

---

## Solver

The model is solved using Pyomo.

The code first tries to find an available solver from the following list:

```text
glpk
cbc
highs
```

If none of these solvers are available, the program raises an error.

For this project, GLPK is the main solver option.

To install GLPK with conda:

```bash
conda install -c conda-forge glpk
```

To check whether GLPK is installed:

```bash
glpsol --version
```

---

## Output Files

After running the optimization model, the following files are produced:

| File | Description |
|---|---|
| `schedule_results.csv` | Optimized maintenance schedule |
| `optimization_summary.csv` | Summary of optimization results |

The schedule output includes:

| Column | Description |
|---|---|
| `aircraft_id` | Aircraft identifier |
| `aircraft_type` | Aircraft type |
| `base_airport` | Base airport |
| `flight_intensity` | Flight intensity |
| `due_status` | Maintenance urgency status |
| `due_day` | Maintenance due day |
| `start_day` | Scheduled maintenance start day |
| `duration` | Maintenance duration |
| `end_day` | Scheduled maintenance end day |
| `start_date` | Scheduled maintenance start date |
| `end_date` | Scheduled maintenance end date |
| `delay_days` | Delay beyond due day |
| `cancelled_flights_during_maintenance` | Cancelled flights during maintenance |
| `scheduled_flag` | 1 if aircraft is scheduled |
| `unscheduled_flag` | 1 if required maintenance is not scheduled |

---

## Visualization

The file `visualization.py` creates four plots:

| Plot | Description |
|---|---|
| `gantt_schedule.png` | Maintenance schedule by aircraft |
| `capacity_usage.png` | Daily maintenance utilization versus capacity |
| `due_vs_start_day.png` | Comparison of due day and scheduled start day |
| `cancelled_flights.png` | Cancelled flights by aircraft |

The Gantt chart shows the maintenance window of each aircraft. Red bars indicate maintenance operations that start after the due day.

The capacity usage chart compares the number of aircraft in maintenance with the available daily maintenance capacity.

The due versus start day chart shows whether maintenance is scheduled before or after the due day.

The cancelled flights chart shows how many flights are cancelled during each aircraft's maintenance period.

---

## How to Run

The project can be run in Spyder by opening and running the files in this order:

```text
1. data_generation.py
2. preprocessing.py
3. optimization_model.py
4. visualization.py
```

Before running the files, make sure that the required packages are installed:

```bash
conda install -c conda-forge pandas numpy matplotlib pyomo glpk
```

The project uses the following main Python libraries:

| Library | Purpose |
|---|---|
| `pandas` | Data handling and preprocessing |
| `numpy` | Random data generation and numerical operations |
| `pyomo` | Optimization modeling |
| `matplotlib` | Visualization |

---

## Assumptions

The model is based on the following assumptions:

- The dataset is synthetic.
- Maintenance operations are treated as heavy maintenance.
- Each aircraft can start maintenance at most once during the planning horizon.
- Maintenance is non-preemptive, meaning it continues without interruption once started.
- All aircraft are assumed to be compatible with the available maintenance capacity.
- Base airport information is included for realism but not used as a hard constraint.
- Flights scheduled during maintenance are treated as cancelled or reassigned.
- Daily maintenance capacity may vary by day.
- Capacity overflow is allowed through a slack variable but receives a penalty.
- Crew, technician skill, spare aircraft, and route assignment constraints are not included.

---

## Future Improvements

Possible extensions include adding maintenance base compatibility constraints, technician and workforce availability constraints, spare aircraft assignment, real airline schedule data, multiple maintenance types, more than one maintenance event per aircraft, sensitivity analysis on penalty weights, comparison of GLPK with other solvers, and a dashboard for scenario testing.

---

## Skills Demonstrated

This project demonstrates synthetic data generation, data preprocessing, mixed-integer linear programming, aircraft maintenance scheduling, Pyomo modeling, open-source solver usage, result interpretation, data visualization, and modular Python coding.
