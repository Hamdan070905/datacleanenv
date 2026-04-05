"""
Data models for the DataCleanEnv Environment.
"""

from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataCleanAction(Action):
    """Action for the DataCleanEnv environment."""

    action_type: str = Field(..., description="Type: fill_missing | fix_type | remove_duplicate | fix_value | done")
    row_index: Optional[int] = Field(default=None, description="Row index to modify")
    column: Optional[str] = Field(default=None, description="Column name to modify")
    new_value: Optional[Any] = Field(default=None, description="New value to set")


class DataCleanObservation(Observation):
    """Observation from the DataCleanEnv environment."""

    task_id: str = Field(default="easy", description="Current task: easy | medium | hard")
    step: int = Field(default=0, description="Current step count")
    current_table: list = Field(default_factory=list, description="Current state of the table as list of dicts")
    issues_remaining: int = Field(default=0, description="Number of issues left to fix")
    last_action_result: str = Field(default="reset", description="Result of the last action taken")
