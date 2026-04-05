---
title: DataCleanEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
- openenv
---

# DataCleanEnv 🧹

> **A real-world OpenEnv environment for training AI agents to clean tabular data.**

An RL environment where an AI agent detects and fixes common data quality issues — missing values, duplicate rows, wrong data types, and invalid values — in tabular datasets.

---

## Why DataCleanEnv?

Data cleaning is one of the most time-consuming tasks in any data pipeline. Data scientists spend ~80% of their time cleaning data. This environment trains agents to automate that process, making it immediately useful for the RL/agent community.

---

## Environment Overview

The agent receives a "dirty" table and must identify and fix all data quality issues using a set of typed actions. Issues range from simple missing values (easy) to complex combinations of duplicates, type errors, and invalid values (hard).

---

## Action Space

| Action Type | Parameters | Description |
|-------------|-----------|-------------|
| `fill_missing` | `row_index`, `column`, `new_value` | Fill a missing (None) cell |
| `fix_type` | `row_index`, `column`, `new_value` | Fix a wrong data type |
| `remove_duplicate` | `row_index` | Remove a duplicate row |
| `fix_value` | `row_index`, `column`, `new_value` | Fix an invalid value |
| `done` | — | Declare task complete |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task (easy/medium/hard) |
| `step` | int | Current step count |
| `current_table` | list[dict] | The current state of the table |
| `issues_remaining` | int | Number of unfixed issues |
| `last_action_result` | string | Result of last action |
| `done` | bool | Whether episode is complete |

---

## Tasks

### Task 1 — Easy
Fix 2 missing values in a small employee table. 4 rows x 4 columns.

### Task 2 — Medium
Fix missing values AND remove a duplicate row in a sales table. 5 rows x 4 columns.

### Task 3 — Hard
Fix missing values, a duplicate, wrong data types, and invalid values in a patient records table. 6 rows x 5 columns.

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct fix (exact value match) | +0.30 |
| Partial fix (wrong value but correct cell) | +0.10 |
| All issues resolved (bonus) | +0.20 |
| Invalid / unknown action | -0.05 |

---

## API Endpoints

```
POST /reset?task_id=easy     → Initial observation
POST /step                   → Step with Action JSON
GET  /state                  → Current full state
GET  /tasks                  → List all tasks
GET  /                       → Health check
```

---

## Setup and Usage

### Local

```bash
pip install -r requirements.txt
python environment.py
```

### Docker

```bash
docker build -t datacleanenv .
docker run -p 7860:7860 datacleanenv
```

### Run Baseline Inference

```bash
export HF_TOKEN=your_token_here
python inference.py
```

---

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.85 |
| medium | 0.70 |
| hard | 0.55 |

---

## Project Structure

```
├── environment.py
├── graders.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
