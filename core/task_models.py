from __future__ import annotations

import sys as _sys, os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from core.state import TaskKind, TaskStatus

class Task(BaseModel):
    """
    Pydantic view of a task node in the planner's DAG.

    Mirrors state.AgentTask so it can be round-tripped without data loss:

        agent_task = AgentTask(**task.model_dump())

    ID convention
    -------------
    String IDs are stable across runs, merges, and log reconstruction.
    Recommended format: "<kind>-<target_slug>", e.g. "refactor-Foo.process".
    Integer IDs break when tasks from parallel runs are combined.

    Priority convention
    -------------------
    1 = highest priority, 10 = lowest (Unix nice semantics).
    The Planner schedules lowest-numbered tasks first among all
    tasks whose depends_on are fully satisfied.
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Stable string identifier, e.g. 'refactor-Foo.process'.",
    )
    kind: TaskKind = Field(
        ...,
        description="Category of work: refactor | document | analyze | evaluate.",
    )
    target: str = Field(
        ...,
        min_length=1,
        description="Fully-qualified name of the element to act on.",
    )
    agent: str = Field(
        ...,
        min_length=1,
        description="ID of the agent assigned to execute this task.",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="1 = highest, 10 = lowest. Planner schedules lower values first.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Lifecycle state. Planner only schedules PENDING tasks.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must reach DONE before this task is scheduled.",
    )
    notes: str = Field(
        default="",
        description="Optional context injected by the Planner or AnalyzerAgent.",
    )
    assigned_to: Optional[str] = Field(
        default=None,
        description="Agent ID currently owning this task, or None if unassigned.",
    )

    # ------------------------------------------------------------------
    # Field-level validators
    # ------------------------------------------------------------------

    @field_validator("id")
    @classmethod
    def id_no_whitespace(cls, v: str) -> str:
        """Task IDs are used as dict keys and log tokens — no spaces allowed."""
        if " " in v:
            raise ValueError("Task id must not contain spaces.")
        return v

    @field_validator("depends_on")
    @classmethod
    def no_self_dependency(cls, v: list[str], info) -> list[str]:
        """A task cannot depend on itself."""
        task_id = info.data.get("id")
        if task_id and task_id in v:
            raise ValueError(f"Task '{task_id}' cannot depend on itself.")
        return v

    # ------------------------------------------------------------------
    # List-level cross-validator (call on a collection, not a single Task)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_dag(tasks: list[Task]) -> list[Task]:
        """
        Verify that every depends_on ID references an existing task in the
        same collection, and that the dependency graph is acyclic (no cycles).

        Raises ValueError describing the first problem found.

        Usage (e.g. in a Planner agent):
            validated = Task.validate_dag(task_list)
        """
        id_set = {t.id for t in tasks}

        # 1. Dangling edges
        for task in tasks:
            missing = [dep for dep in task.depends_on if dep not in id_set]
            if missing:
                raise ValueError(
                    f"Task '{task.id}' depends_on unknown task IDs: {missing}"
                )

        # 2. Cycle detection via DFS (Kahn's algorithm)
        from collections import deque

        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.depends_on:
                adjacency[dep].append(task.id)
                in_degree[task.id] += 1

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if processed != len(tasks):
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            raise ValueError(f"Cycle detected in task DAG involving: {cycle_nodes}")

        return tasks

    @model_validator(mode="after")
    def status_and_assignment_consistent(self) -> Task:
        """
        If a task is IN_PROGRESS it must have an assigned agent.
        If a task is DONE or FAILED it must have a recorded agent.
        """
        needs_assignment = {TaskStatus.IN_PROGRESS, TaskStatus.DONE, TaskStatus.FAILED}
        if self.status in needs_assignment and self.assigned_to is None:
            raise ValueError(
                f"Task '{self.id}' has status '{self.status}' "
                f"but assigned_to is None."
            )
        return self