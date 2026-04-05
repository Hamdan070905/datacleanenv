"""
DataCleanEnv Environment Implementation.

A real-world data cleaning RL environment where an AI agent detects
and fixes common data quality issues in tabular datasets.
"""

import copy
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCleanAction, DataCleanObservation
except ImportError:
    from models import DataCleanAction, DataCleanObservation


class DataCleanEnvironment(Environment):
    """
    A real-world data cleaning RL environment.

    The agent must detect and fix data quality issues:
    - Missing values (None/null)
    - Duplicate rows
    - Wrong data types
    - Invalid values

    Three tasks of increasing difficulty: easy -> medium -> hard.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 30

    DATASETS = {
        "easy": {
            "table": [
                {"id": 1, "name": "Alice",   "age": 30,   "salary": 70000},
                {"id": 2, "name": "Bob",     "age": None, "salary": 65000},
                {"id": 3, "name": "Charlie", "age": 25,   "salary": None},
                {"id": 4, "name": "Diana",   "age": 28,   "salary": 72000},
            ],
            "issues": [
                {"type": "fill_missing", "row_index": 1, "column": "age",    "correct_value": 32},
                {"type": "fill_missing", "row_index": 2, "column": "salary", "correct_value": 68000},
            ],
        },
        "medium": {
            "table": [
                {"id": 1, "product": "Laptop",  "units": 5,    "price": 999.99},
                {"id": 2, "product": "Phone",   "units": None, "price": 499.99},
                {"id": 3, "product": "Tablet",  "units": 8,    "price": None},
                {"id": 4, "product": "Laptop",  "units": 5,    "price": 999.99},
                {"id": 5, "product": "Monitor", "units": 3,    "price": 299.99},
            ],
            "issues": [
                {"type": "fill_missing",     "row_index": 1, "column": "units",  "correct_value": 12},
                {"type": "fill_missing",     "row_index": 2, "column": "price",  "correct_value": 349.99},
                {"type": "remove_duplicate", "row_index": 3, "column": None,     "correct_value": None},
            ],
        },
        "hard": {
            "table": [
                {"id": 1, "name": "John",  "age": 45,      "blood_type": "A+",  "weight_kg": 72.5},
                {"id": 2, "name": "Mary",  "age": "forty", "blood_type": "B-",  "weight_kg": 58.0},
                {"id": 3, "name": "Steve", "age": 29,      "blood_type": None,  "weight_kg": 85.0},
                {"id": 4, "name": "Linda", "age": 33,      "blood_type": "O+",  "weight_kg": -5.0},
                {"id": 5, "name": "John",  "age": 45,      "blood_type": "A+",  "weight_kg": 72.5},
                {"id": 6, "name": "Sam",   "age": None,    "blood_type": "AB+", "weight_kg": 90.0},
            ],
            "issues": [
                {"type": "fix_type",         "row_index": 1, "column": "age",        "correct_value": 40},
                {"type": "fill_missing",     "row_index": 2, "column": "blood_type", "correct_value": "AB-"},
                {"type": "fix_value",        "row_index": 3, "column": "weight_kg",  "correct_value": 65.0},
                {"type": "remove_duplicate", "row_index": 4, "column": None,         "correct_value": None},
                {"type": "fill_missing",     "row_index": 5, "column": "age",        "correct_value": 38},
            ],
        },
    }

    def __init__(self):
        self.task_id = "easy"
        self.table = []
        self.original_issues = []
        self.fixed_issues = set()
        self.step_count = 0
        self._done = False
        self.last_action_result = "none"
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> DataCleanObservation:
        task_id = kwargs.get("task_id", "easy")
        if task_id not in self.DATASETS:
            task_id = "easy"
        self.task_id = task_id
        dataset = self.DATASETS[task_id]
        self.table = copy.deepcopy(dataset["table"])
        self.original_issues = copy.deepcopy(dataset["issues"])
        self.fixed_issues = set()
        self.step_count = 0
        self._done = False
        self.last_action_result = "reset"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._observe()

    def step(self, action: DataCleanAction, timeout_s=None, **kwargs) -> DataCleanObservation:
        if self._done:
            return self._observe()

        self.step_count += 1
        self._state.step_count = self.step_count
        reward_val = 0.0

        if action.action_type == "done":
            self._done = True
            self.last_action_result = "agent_declared_done"
        elif action.action_type == "fill_missing":
            reward_val = self._apply_fill_missing(action)
        elif action.action_type == "fix_type":
            reward_val = self._apply_fix_type(action)
        elif action.action_type == "remove_duplicate":
            reward_val = self._apply_remove_duplicate(action)
        elif action.action_type == "fix_value":
            reward_val = self._apply_fix_value(action)
        else:
            self.last_action_result = "unknown_action"
            reward_val = -0.05

        if len(self.fixed_issues) == len(self.original_issues):
            self._done = True
            reward_val += 0.2

        if self.step_count >= self.MAX_STEPS:
            self._done = True

        obs = self._observe()
        obs.reward = round(reward_val, 2)
        obs.done = self._done
        return obs

    @property
    def state(self) -> State:
        return self._state

    def close(self):
        pass

    def _observe(self) -> DataCleanObservation:
        return DataCleanObservation(
            task_id=self.task_id,
            step=self.step_count,
            current_table=list(self.table),
            issues_remaining=len(self.original_issues) - len(self.fixed_issues),
            last_action_result=self.last_action_result,
            done=self._done,
            reward=0.0,
        )

    def _issue_key(self, issue):
        return f"{issue['type']}_{issue['row_index']}_{issue['column']}"

    def _apply_fill_missing(self, action):
        for issue in self.original_issues:
            key = self._issue_key(issue)
            if (issue["type"] == "fill_missing"
                    and issue["row_index"] == action.row_index
                    and issue["column"] == action.column
                    and key not in self.fixed_issues):
                try:
                    self.table[action.row_index][action.column] = action.new_value
                except (IndexError, KeyError):
                    self.last_action_result = "invalid_index"
                    return -0.05
                self.fixed_issues.add(key)
                if action.new_value == issue["correct_value"]:
                    self.last_action_result = f"fill_missing:correct:{action.column}"
                    return 0.3
                else:
                    self.last_action_result = f"fill_missing:partial:{action.column}"
                    return 0.1
        self.last_action_result = "fill_missing:no_match"
        return -0.05

    def _apply_fix_type(self, action):
        for issue in self.original_issues:
            key = self._issue_key(issue)
            if (issue["type"] == "fix_type"
                    and issue["row_index"] == action.row_index
                    and issue["column"] == action.column
                    and key not in self.fixed_issues):
                try:
                    self.table[action.row_index][action.column] = action.new_value
                except (IndexError, KeyError):
                    return -0.05
                self.fixed_issues.add(key)
                if action.new_value == issue["correct_value"]:
                    self.last_action_result = f"fix_type:correct:{action.column}"
                    return 0.3
                else:
                    self.last_action_result = f"fix_type:partial:{action.column}"
                    return 0.1
        self.last_action_result = "fix_type:no_match"
        return -0.05

    def _apply_remove_duplicate(self, action):
        for issue in self.original_issues:
            key = self._issue_key(issue)
            if (issue["type"] == "remove_duplicate"
                    and issue["row_index"] == action.row_index
                    and key not in self.fixed_issues):
                try:
                    self.table.pop(action.row_index)
                except IndexError:
                    return -0.05
                for other in self.original_issues:
                    if other["row_index"] is not None and other["row_index"] > action.row_index:
                        other["row_index"] -= 1
                self.fixed_issues.add(key)
                self.last_action_result = f"remove_duplicate:correct:{action.row_index}"
                return 0.3
        self.last_action_result = "remove_duplicate:no_match"
        return -0.05

    def _apply_fix_value(self, action):
        for issue in self.original_issues:
            key = self._issue_key(issue)
            if (issue["type"] == "fix_value"
                    and issue["row_index"] == action.row_index
                    and issue["column"] == action.column
                    and key not in self.fixed_issues):
                try:
                    self.table[action.row_index][action.column] = action.new_value
                except (IndexError, KeyError):
                    return -0.05
                self.fixed_issues.add(key)
                if action.new_value == issue["correct_value"]:
                    self.last_action_result = f"fix_value:correct:{action.column}"
                    return 0.3
                else:
                    self.last_action_result = f"fix_value:partial:{action.column}"
                    return 0.1
        self.last_action_result = "fix_value:no_match"
        return -0.05
