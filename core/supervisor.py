from __future__ import annotations

import sys as _sys, os as _os

from matplotlib import lines
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

"""
supervisor.py — Supervisor Agent

Position in pipeline (architecture diagram)
--------------------------------------------
  Planner Agent (creates DAG)
    ↓
  → [THIS FILE] Supervisor (orchestrates · assigns agent by task.kind)
    ↓               ↓
  Refactor Agent   Doc Agent
    ↓               ↓
        Evaluator
        ↓           ↓
  Update RepoState  Feedback loop → Planner (replan=True)

Research basis
--------------
- Scheduler-Agent-Supervisor pattern (Microsoft Azure Architecture, 2024):
  Supervisor maintains retry count per task in the state store; on threshold
  breach it triggers compensating transactions or escalates to the Planner.
- Layered fault tolerance (DEV Community, 2025): retry → fallback → error
  classification → checkpoint. Reduces unrecoverable failures from 23% → <2%.
- Adaptive retry with exponential backoff + jitter (SparkCo, 2025):
  prevents thundering-herd on transient model failures.
- Deterministic code-driven orchestration (OpenAI Agents SDK, 2024):
  routing by task.kind in code is faster, cheaper, and more auditable than
  LLM-driven routing.
- Multi-agent coordinator pattern (MyAntFarm.ai / arXiv 2511.15755, 2025):
  coordinator dispatches context to specialized agents, aggregates outputs,
  produces structured brief — architectural value is in decomposition quality,
  not model count.
- DAG-based dynamic task updates (EmergentMind, 2024): ongoing update of
  subtask definitions in response to execution results.

Responsibilities
----------------
1. Dependency-respecting dispatch: only release tasks whose depends_on are
   all DONE.
2. Kind → agent routing: TaskKind.REFACTOR → RefactorAgent,
   TaskKind.DOCUMENT → DocAgent.  No agent class names leak into the Planner.
3. Retry with exponential backoff + jitter: transient failures are retried
   up to MAX_RETRIES with capped delay; permanent failures are marked FAILED
   without wasted retries.
4. Failure classification: distinguish retriable (model timeout, OOM,
   unexpected exception) from non-retriable (validation error, missing target).
5. Feedback loop trigger: after all tasks complete (or exhaust retries), call
   PlannerAgent.run(replan=True) if any results fall below the confidence
   threshold defined in pipeline.py.
6. All state transitions go through RepoState.evolve() — no mutable state
   outside the immutable state object.
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.state import (
    RepoState,
    AgentTask,
    RefactorResult,
    DocumentationResult,
    TaskKind,
    TaskStatus,
    Severity,
)

if TYPE_CHECKING:
    from agents.refactor_agent import RefactorAgent
    from agents.doc_agent import DocAgent
    from core.planner_agent import PlannerAgent

logger = logging.getLogger(__name__)

AGENT_ID = "supervisor"

# ---------------------------------------------------------------------------
# Retry / backoff constants
# (research: exponential backoff + jitter, capped delay, max attempts)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3             # attempts before a task is marked FAILED
BACKOFF_BASE = 2.0          # seconds — base for exponential backoff
BACKOFF_CAP = 30.0          # seconds — maximum delay between retries
JITTER_FACTOR = 0.2         # ±20 % noise to stagger concurrent retries

# Confidence threshold below which the Evaluator feedback loop fires
REPLAN_CONFIDENCE_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Non-retriable error patterns
# (research: classify failures to prevent wasteful retries)
# ---------------------------------------------------------------------------

_NON_RETRIABLE_SUBSTRINGS = (
    "target not found",
    "invalid task",
    "missing required field",
    "validation error",
    "unsupported kind",
)

def _is_retriable(exc: Exception) -> bool:
    """
    Return True for transient failures (timeouts, OOM, unexpected exceptions).
    Return False for permanent failures (bad input, schema errors).

    Research: classifying errors prevents burning retries on failures that
    no amount of waiting will resolve (SparkCo, 2025).
    """
    msg = str(exc).lower()
    return not any(pattern in msg for pattern in _NON_RETRIABLE_SUBSTRINGS)

# ---------------------------------------------------------------------------
# Per-task execution bookkeeping (ephemeral — not persisted in RepoState)
# ---------------------------------------------------------------------------

@dataclass
class _TaskRecord:
    task: AgentTask
    attempts: int = 0
    last_error: str = ""
    next_eligible_at: float = field(default_factory=time.monotonic)

def _backoff_delay(attempt: int) -> float:
    """
    Exponential backoff with ±jitter.

    delay = min(base * 2^attempt, cap) * uniform(1-jitter, 1+jitter)

    This matches the pattern recommended by SparkCo (2025) and used in
    LangGraph's built-in retry policies.
    """
    raw = BACKOFF_BASE * math.pow(2.0, attempt)
    capped = min(raw, BACKOFF_CAP)
    jitter = random.uniform(1.0 - JITTER_FACTOR, 1.0 + JITTER_FACTOR)
    return capped * jitter

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class SupervisorAgent:
    """
    Orchestrates execution of the task DAG produced by PlannerAgent.

    The Supervisor is the only component that knows which concrete agent
    class handles which TaskKind.  The Planner and state model are agnostic
    of agent implementations.

    Parameters
    ----------
    refactor_agent  Instance of RefactorAgent (from pipeline.py).
    doc_agent       Instance of DocAgent (from pipeline.py).
    planner_agent   Instance of PlannerAgent — used only for the feedback
                    loop (replan=True path).
    """

    def __init__(
        self,
        refactor_agent: RefactorAgent,
        doc_agent: DocAgent,
        planner_agent: PlannerAgent,
    ) -> None:
        self.refactor_agent = refactor_agent
        self.doc_agent = doc_agent
        self.planner_agent = planner_agent

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, repo_state: RepoState) -> RepoState:
        """
        Execute the full task DAG in repo_state.tasks.

        Execution order
        ---------------
        Each iteration of the main loop selects all tasks that are:
          - PENDING
          - whose depends_on task_ids are all in completed_tasks

        Tasks on different targets run in the order they appear in the
        priority-sorted list.  Parallel execution is intentionally NOT
        implemented here — the pipeline runs on a single CPU and the
        local models are not thread-safe.  If you move to a distributed
        executor (Celery, Ray), replace the sequential loop with a
        futures-based dispatcher while keeping the same state mutation
        contract.

        Returns
        -------
        Final RepoState after all tasks have reached DONE or FAILED.
        If any refactor results fell below REPLAN_CONFIDENCE_THRESHOLD,
        a replanning cycle is triggered and its tasks are also executed.
        """
        state = repo_state

        # Build ephemeral retry records
        records: dict[str, _TaskRecord] = {
            t.task_id: _TaskRecord(task=t)
            for t in state.tasks
        }

        logger.info(
            "Supervisor starting. %d task(s) in DAG.", len(state.tasks)
        )

        # Main dispatch loop
        state = self._dispatch_loop(state, records)

        # Feedback loop: trigger replanning if low-confidence results exist
        state = self._maybe_replan(state)

        return state

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    def _dispatch_loop(
        self,
        state: RepoState,
        records: dict[str, _TaskRecord],
    ) -> RepoState:
        """
        Iterate until no runnable tasks remain.

        A task is runnable when:
          1. Its status is PENDING.
          2. All task_ids in depends_on are present in state.completed_tasks.
          3. Its next_eligible_at timestamp has passed (backoff window).
        """
        while True:
            runnable = self._get_runnable(state, records)
            if not runnable:
                break

            # Sort by priority (lower = higher priority) then by task_id for
            # deterministic ordering when priorities are equal.
            runnable.sort(key=lambda r: (r.task.priority, r.task.task_id))

            for record in runnable:
                state = self._execute_one(state, record, records)

        self._log_final_summary(state)
        return state

    def _get_runnable(
        self,
        state: RepoState,
        records: dict[str, _TaskRecord],
    ) -> list[_TaskRecord]:
        """
        Return all records that are eligible to run right now.
        """
        done_ids = set(state.completed_tasks)
        now = time.monotonic()

        runnable = []
        for task_id, record in records.items():
            task = record.task

            # Skip if not pending
            if task.status != TaskStatus.PENDING:
                continue

            # Skip if dependencies not satisfied
            if not all(dep in done_ids for dep in task.depends_on):
                continue

            # Skip if still in backoff window
            if record.next_eligible_at > now:
                continue

            runnable.append(record)

        return runnable

    # ------------------------------------------------------------------
    # Single task execution with retry
    # ------------------------------------------------------------------

    def _execute_one(
        self,
        state: RepoState,
        record: _TaskRecord,
        records: dict[str, _TaskRecord],
    ) -> RepoState:
        """
        Attempt to run a single task, handling retries and failure
        classification.  Returns updated RepoState.
        """
        task = record.task
        record.attempts += 1

        logger.info(
            "Dispatching task %s  kind=%s  target=%s  attempt=%d/%d",
            task.task_id,
            task.kind.value,
            task.target,
            record.attempts,
            MAX_RETRIES,
        )

        # Mark IN_PROGRESS in state
        state = self._transition(
            state,
            task,
            records,
            new_status=TaskStatus.IN_PROGRESS,
            action="task_started",
            summary=f"attempt {record.attempts}",
        )

        try:
            result_payload = self._route_and_run(task, state)
            state = self._on_success(state, task, records, result_payload)

        except Exception as exc:
            state = self._on_failure(state, task, records, record, exc)

        return state

    def _route_and_run(
        self,
        task: AgentTask,
        state: RepoState,
    ) -> dict[str, Any]:
        """
        Route task.kind → concrete agent and run it.

        Research: code-driven routing is faster, cheaper, and more
        auditable than LLM-driven routing (OpenAI Agents SDK, 2024).
        """
        # Resolve the target function/class from state for context injection
        target_fn = next(
            (f for f in state.functions if f.name == task.target),
            None,
        )
        target_code = state.raw_code  # full source as fallback

        if task.kind == TaskKind.REFACTOR:
            if target_fn is None:
                raise ValueError(
                    f"target not found in repo state: '{task.target}'"
                )
            # Build a focused prompt from notes + structural context
            prompt = self._build_refactor_prompt(task, target_fn, state)
            return self.refactor_agent.run(prompt, target_code)

        elif task.kind == TaskKind.DOCUMENT:
            prompt = self._build_doc_prompt(task, target_fn, state)
            parsed_ctx = self._build_parsed_context(state)
            return self.doc_agent.run(prompt, parsed_ctx)
        
        else:
            raise ValueError(f"unsupported kind: {task.kind}")

    # ------------------------------------------------------------------
    # Success / failure handlers
    # ------------------------------------------------------------------

    def _on_success(
        self,
        state: RepoState,
        task: AgentTask,
        records: dict[str, _TaskRecord],
        payload: dict[str, Any],
    ) -> RepoState:
        """
        Persist result into RepoState, mark task DONE.
        """
        if task.kind == TaskKind.REFACTOR:
            result = RefactorResult(
                task_id=task.task_id,
                target_name=task.target,
                success=True,
                changes=payload.get("refactored_code", ""),
                agent_id="refactor_agent",
                confidence=float(payload.get("confidence", 1.0)),
            )
            new_refactor_results = list(state.refactor_results) + [result]
            state = state.evolve(
                agent_id=AGENT_ID,
                action="task_done",
                summary=f"{task.task_id} DONE (confidence={result.confidence:.2f})",
                refactor_results=new_refactor_results,
                completed_tasks=list(state.completed_tasks) + [task.task_id],
            )

        elif task.kind == TaskKind.DOCUMENT:
            result = DocumentationResult(
                task_id=task.task_id,
                target_name=task.target,
                docstring=payload.get("documentation", ""),
                agent_id="doc_agent",
                confidence=float(payload.get("confidence", 1.0)),
            )
            new_doc_results = list(state.documentation_results) + [result]
            state = state.evolve(
                agent_id=AGENT_ID,
                action="task_done",
                summary=f"{task.task_id} DONE (confidence={result.confidence:.2f})",
                documentation_results=new_doc_results,
                completed_tasks=list(state.completed_tasks) + [task.task_id],
            )

        # Update in-memory record status
        records[task.task_id].task = AgentTask(
            **{**{field.name: getattr(task, field.name) for field in task.__dataclass_fields__.values()}, "status": TaskStatus.DONE}
        )

        logger.info("Task %s completed successfully.", task.task_id)
        return state

    def _on_failure(
        self,
        state: RepoState,
        task: AgentTask,
        records: dict[str, _TaskRecord],
        record: _TaskRecord,
        exc: Exception,
    ) -> RepoState:
        """
        Classify the error, apply backoff or mark FAILED.

        Research (SparkCo 2025; Azure SAS pattern 2024):
        - Retriable errors get exponential backoff + jitter, up to MAX_RETRIES.
        - Non-retriable errors are marked FAILED immediately — no wasted retries.
        - On final failure, dependent tasks are also cancelled (FAILED) to
          prevent them from waiting forever.
        """
        record.last_error = str(exc)
        retriable = _is_retriable(exc)

        if retriable and record.attempts < MAX_RETRIES:
            delay = _backoff_delay(record.attempts)
            record.next_eligible_at = time.monotonic() + delay
            logger.warning(
                "Task %s failed (retriable, attempt %d/%d). "
                "Retrying in %.1fs. Error: %s",
                task.task_id, record.attempts, MAX_RETRIES, delay, exc,
            )
            # Revert to PENDING so the dispatch loop picks it up again
            records[task.task_id].task = AgentTask(
                **{**{field.name: getattr(task, field.name) for field in task.__dataclass_fields__.values()}, "status": TaskStatus.PENDING}
            )
            state = self._transition(
                state, task, records,
                new_status=TaskStatus.PENDING,
                action="task_retry",
                summary=f"attempt {record.attempts} failed, retrying in {delay:.1f}s",
            )
        else:
            reason = (
                "non-retriable error"
                if not retriable
                else f"exhausted {MAX_RETRIES} retries"
            )
            logger.error(
                "Task %s FAILED (%s). Error: %s",
                task.task_id, reason, exc,
            )
            records[task.task_id].task = AgentTask(
                **{**{field.name: getattr(task, field.name) for field in task.__dataclass_fields__.values()}, "status": TaskStatus.FAILED}
            )
            state = self._transition(
                state, task, records,
                new_status=TaskStatus.FAILED,
                action="task_failed",
                summary=f"{reason}: {exc}",
            )
            # Cancel dependent tasks — they can never run
            state = self._cancel_dependents(state, task.task_id, records)

        return state

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    def _maybe_replan(self, state: RepoState) -> RepoState:
        """
        If any refactor result has confidence below the threshold, ask the
        Planner to generate targeted retry tasks, then execute them.

        Research: iterative replan-execute cycles are the standard pattern
        in plan-and-execute architectures (LLM Planner Agent survey, 2025;
        RefAgent, 2025).
        """
        low_conf = [
            r for r in state.refactor_results
            if r.confidence < REPLAN_CONFIDENCE_THRESHOLD
        ]
        if not low_conf:
            logger.info("No low-confidence results. Feedback loop not triggered.")
            return state

        logger.info(
            "Feedback loop triggered for %d low-confidence result(s): %s",
            len(low_conf),
            [r.target_name for r in low_conf],
        )

        # Ask planner to generate retry tasks
        state = self.planner_agent.run(
            state,
            #smells=list(state.smells),
            replan=True,
        )

        # Build retry records for new tasks only
        existing_ids = set()
        for t in state.tasks:
            if t.task_id not in existing_ids:
                existing_ids.add(t.task_id)

        retry_records: dict[str, _TaskRecord] = {
            t.task_id: _TaskRecord(task=t)
            for t in state.tasks
            if t.status == TaskStatus.PENDING
        }

        if retry_records:
            state = self._dispatch_loop(state, retry_records)
        else:
            logger.info("Planner produced no new retry tasks.")

        return state

    # ------------------------------------------------------------------
    # Dependency cancellation
    # ------------------------------------------------------------------

    def _cancel_dependents(
        self,
        state: RepoState,
        failed_task_id: str,
        records: dict[str, _TaskRecord],
    ) -> RepoState:
        """
        Mark as FAILED all tasks that transitively depend on a failed task.

        Research (Azure SAS pattern, 2024): when a step fails permanently,
        the Supervisor must ensure dependent steps do not hang waiting for
        a DONE signal that will never arrive.
        """
        to_cancel = [
            t.task_id for t in state.tasks
            if failed_task_id in t.depends_on
            and t.status == TaskStatus.PENDING
        ]

        for task_id in to_cancel:
            records[task_id].task = AgentTask(
                **{**{field.name: getattr(records[task_id].task, field.name) for field in records[task_id].task.__dataclass_fields__.values()}, "status": TaskStatus.FAILED}
            )
            logger.warning(
                "Cancelled task %s because dependency %s failed.",
                task_id, failed_task_id,
            )

        if to_cancel:
            state = state.evolve(
                agent_id=AGENT_ID,
                action="tasks_cancelled",
                summary=(
                    f"Cancelled {len(to_cancel)} dependent task(s) "
                    f"after {failed_task_id} failed: {to_cancel}"
                ),
            )

        return state

    # ------------------------------------------------------------------
    # State transition helper
    # ------------------------------------------------------------------

    def _transition(
        self,
        state: RepoState,
        task: AgentTask,
        records: dict[str, _TaskRecord],
        *,
        new_status: TaskStatus,
        action: str,
        summary: str,
    ) -> RepoState:
        """
        Produce a new RepoState with task's status updated + provenance logged.
        The task list itself is not stored in RepoState by status — status is
        tracked in the ephemeral _TaskRecord.  We call evolve() here purely
        to append a provenance entry.
        """
        return state.evolve(
            agent_id=AGENT_ID,
            action=f"{action}:{task.task_id}",
            summary=f"[{new_status.value}] {task.task_id} — {summary}",
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # (keep prompt construction in the Supervisor so agent classes stay
    # model-agnostic and testable without a full pipeline)
    # ------------------------------------------------------------------

    def _build_refactor_prompt(self, task: AgentTask, target_fn, state: RepoState) -> str:
        """
        Use the prompt generated by PromptingEngine stored in RepoState.metadata.

        This is critical: the stored prompt matches the exact CodeT5+ training
        format. Building a custom prompt here produces garbage output because
        the model was not trained on that format.
        """
        stored = state.metadata.get("refactor_prompt", "")
        if stored.strip():
            return stored

        # Fallback: match CodeT5+ training format
        code = state.raw_code
        return (
            "Refactor this Java method to improve code quality.\n\n"
            f"{code}\n\n"
            "Strategy: Extract cohesive groups into private helper methods.\n"
            "Return ONLY the refactored Java code:"
        )

    def _build_doc_prompt(self, task: AgentTask, target_fn, state: RepoState) -> str:
        """
        Use the prompt generated by PromptingEngine stored in RepoState.metadata.

        The stored prompt uses the SHORT template that matches the CodeSearchNet
        Java training format exactly.
        """
        stored = state.metadata.get("doc_prompt", "")
        if stored.strip():
            return stored

        # Fallback: match CodeSearchNet Java training format
        code = state.raw_code
        return f"Generate Javadoc for the following Java method:\n{code}"

    def _build_parsed_context(self, state: RepoState) -> dict:
        """
        Build the 'parsed' dict that DocAgent.run() expects, sourced from
        RepoState rather than re-running the parser.
        """
        return {
            "functions": [
                {
                    "name": f.name,
                    "params": f.params,
                    "docstring": f.docstring,
                    "complexity": f.complexity,
                    "loc": f.loc,
                }
                for f in state.functions
            ],
            "classes": [
                {
                    "name": c.name,
                    "docstring": c.docstring,
                }
                for c in state.classes
            ],
            "imports": state.imports,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_final_summary(self, state: RepoState) -> None:
        done = len(state.completed_tasks)
        total = len(state.tasks)
        failed = sum(
            1 for t in state.tasks
            if t.task_id not in state.completed_tasks
        )
        logger.info(
            "Supervisor finished. %d/%d tasks completed, %d failed/cancelled.",
            done, total, failed,
        )
        for entry in state.provenance_log[-10:]:
            logger.debug(
                "  v%d [%s] %s — %s",
                entry.version, entry.agent_id, entry.action, entry.summary,
            )