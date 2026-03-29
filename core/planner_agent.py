from __future__ import annotations

import sys as _sys, os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

"""
planner_agent.py — Context-Aware Planner Agent

Research basis
--------------
- RefAgent (Oueslati et al., 2025): planner uses dependency graph + code metrics,
  not smell labels alone, for planning decisions.
- LLMCompiler (Kim et al., 2024): tasks carry explicit dependency edges so a
  Task Fetching Unit can schedule parallel-safe work without planner intervention.
- Context-based smell prioritization (Palomba et al., 2017; Pecorelli et al., 2020):
  prioritization should incorporate co-occurrence, instability, and caller-count,
  not just a single severity string.
- LLM Planner Agent survey (EmergentMind, 2025): context windows must be token-
  budgeted; summarizer modules inject only salient signals.
- iSMELL / MANTRA (2024–2025): narrowing the retrieval query to smell type +
  target + metric raises recall from ~15 % to ~87 % in opportunity identification.

Architecture alignment (diagram)
---------------------------------
  Git Repo → Parser → Semantic Units → Doc Builder → Embedding/Index
  → Knowledge Base → Smell Detection → [THIS FILE] Planner Agent (creates DAG)
  → Supervisor (orchestrates) → RefactorAgent | DocAgent
  → Evaluator → Update RepoState ←→ Feedback loop to Planner

The Planner's responsibility ends at emitting a typed, validated AgentTask DAG
into RepoState. It does NOT reference concrete agent class names — that is the
Supervisor's concern. It DOES handle re-planning when called with existing
evaluation scores (feedback loop path).
"""

import logging
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

import sys as _sys
import os as _os
# Ensure project root is on path so state.py is importable
# whether this file lives at root or inside core/
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from core.task_models import Task
from core.hybrid_retriever import hybrid_retrieve
from core.state import RepoState, CodeSmell, AgentTask, TaskKind, TaskStatus, Severity

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum tokens we allow the retrieval query to consume.
# At ~4 chars/token, 120 tokens ≈ 480 chars.  Keeps the query focused.
_MAX_QUERY_CHARS = 480

# Severity → base priority (1 = highest).  Matches Unix-nice convention from
# task.py and is informed by Pecorelli et al. (2020) criticality rankings.
_SEVERITY_PRIORITY: dict[Severity, int] = {
    Severity.CRITICAL: 1,
    Severity.HIGH:     2,
    Severity.MEDIUM:   3,
    Severity.LOW:      4,
}

# How many callers push a target one priority level higher (more callers = more
# blast radius = higher priority).  Based on dependency-propagation research
# (Cerny et al., 2024; Palomba et al., 2017).
_CALLER_ESCALATION_THRESHOLD = 3

# Minimum cohesion score below which a doc task is also generated for a
# function even when no smell targets it directly (low cohesion → hard to
# understand without docs).
_LOW_COHESION_DOC_THRESHOLD = 0.4

# Re-planning: if a task's refactor confidence score is below this threshold
# the Planner splits it into a smaller-scoped retry task.
_REPLAN_CONFIDENCE_THRESHOLD = 0.65

AGENT_ID = "planner"

# ---------------------------------------------------------------------------
# Helper: build a focused, token-budgeted retrieval query
# ---------------------------------------------------------------------------

def _build_retrieval_query(
    repo_state: RepoState,
    smells: list[CodeSmell],
) -> str:
    """
    Build a semantic retrieval query that respects a token budget.

    Research finding (iSMELL, MANTRA 2024–2025): specifying smell type +
    target + metric in the query raises retrieval recall dramatically vs.
    a raw token dump.  We therefore weight smell signals first, then
    structural signals, then trim to the character budget.
    """
    tokens: list[str] = []

    # 1. Highest-signal: smell type + target for top-severity smells
    for smell in sorted(smells, key=lambda s: s.severity, reverse=True)[:3]:
        tokens.append(f"{smell.smell_type} {smell.location}")

    # 2. Structural signals: high-instability classes (fragility risk)
    for cls in sorted(repo_state.classes, key=lambda c: c.instability, reverse=True)[:2]:
        tokens.append(cls.name)

    # 3. High-complexity functions (long method / complex method smell magnets)
    for fn in sorted(repo_state.functions, key=lambda f: f.complexity, reverse=True)[:2]:
        tokens.append(fn.name)

    # 4. Domain anchor
    tokens.append("Java refactoring")

    query = " ".join(t for t in tokens if t)
    return query[:_MAX_QUERY_CHARS]

# ---------------------------------------------------------------------------
# Helper: compute context-aware priority
# ---------------------------------------------------------------------------

def _compute_priority(smell: CodeSmell, repo_state: RepoState) -> int:
    """
    Derive a 1–10 priority from smell severity + structural context.

    Escalation rules (research-grounded):
    - More callers → higher blast radius → escalate priority by 1.
      (Cerny et al. 2024; Palomba et al. 2017: smells in heavily-coupled
      classes have higher dependency-change impact.)
    - Low class instability (stable but smelly) → de-escalate by 1.
      Stable classes are safer to defer; unstable ones need immediate action.
    - co-located CRITICAL smell on same target → escalate by 1.
      (Palomba et al. 2025: smell co-occurrence correlates with higher
      dependency churn.)
    """
    base = _SEVERITY_PRIORITY.get(smell.severity, 3)

    # Caller-count escalation: find the function in the repo state
    fn = next(
        (f for f in repo_state.functions if f.name == smell.location),
        None,
    )
    if fn and len(fn.called_by) >= _CALLER_ESCALATION_THRESHOLD:
        base = max(1, base - 1)

    # Class instability check
    cls = next(
        (c for c in repo_state.classes if smell.location.startswith(c.name)),
        None,
    )
    if cls:
        if cls.instability < 0.3:        # very stable → lower urgency
            base = min(10, base + 1)
        elif cls.instability > 0.7:      # highly unstable → escalate
            base = max(1, base - 1)

    # Co-located critical smell escalation
    co_critical = any(
        s for s in repo_state.smells
        if s.location == smell.location
        and s.severity == Severity.CRITICAL
        and s is not smell
    )
    if co_critical:
        base = max(1, base - 1)

    return max(1, min(10, base))

# ---------------------------------------------------------------------------
# Helper: stable deterministic task ID
# ---------------------------------------------------------------------------

def _task_id(kind: str, target: str, suffix: str = "") -> str:
    """
    Produce a stable, human-readable task ID.

    Using a deterministic slug (not uuid4) means the same smell on the same
    target always produces the same task ID across re-planning cycles, which
    lets the Evaluator's feedback loop reference tasks by ID without a lookup
    table.
    """
    slug = target.replace(".", "-").replace(" ", "_").lower()
    base = f"{kind}-{slug}"
    return f"{base}-{suffix}" if suffix else base

# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Converts RepoState + detected smells into a validated AgentTask DAG.

    Responsibilities
    ----------------
    1. Build a focused, token-budgeted retrieval query.
    2. Retrieve relevant symbols from the knowledge base.
    3. Generate a typed, dependency-linked AgentTask list using
       context-aware priority (severity + structural metrics).
    4. Handle re-planning when called with evaluator feedback
       (scores below threshold → emit targeted retry tasks).
    5. Validate the DAG (no cycles, no dangling edges) via Task.validate_dag().
    6. Return a new RepoState via evolve() with provenance recorded.

    NOT responsible for
    -------------------
    - Knowing which concrete agent class executes a task (Supervisor's concern).
    - Executing any tasks.
    - Storing intermediate debug state outside RepoState.
    """

    def __init__(self, engine, retriever, symbol_index):
        self.engine = engine
        self.retriever = retriever
        self.symbol_index = symbol_index

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        repo_state: RepoState,
        *,
        replan: bool = False,
    ) -> RepoState:
        """
        Parameters
        ----------
        repo_state  Current shared state (immutable).
        smells      CodeSmell instances from the SmellDetector.
                    Must be typed CodeSmell — not raw dicts.
        replan      If True, the Planner is being called by the Evaluator
                    feedback loop to address failed or low-confidence tasks.
                    Existing DONE tasks are preserved; only FAILED/low-confidence
                    tasks are re-emitted.

        Returns
        -------
        New RepoState with tasks field replaced and provenance entry appended.
        """

        # ── 1. Retrieval ──────────────────────────────────────────────────
        print("🔥 PLANNER RUN VERSION 2")
        smells = repo_state.smells
        retrieved_symbols = self._retrieve(repo_state, smells)

        # ── 2. Task generation ────────────────────────────────────────────
        if replan:
            tasks = self._replan_tasks(repo_state, retrieved_symbols)
            action = "replan"
            summary = f"Re-planning {len(tasks)} task(s) from evaluator feedback"
        elif smells:
            tasks = self._tasks_from_smells(repo_state, smells, retrieved_symbols)
            action = "plan_from_smells"
            summary = f"Generated {len(tasks)} task(s) from {len(smells)} smell(s)"
        else:
            tasks = self._tasks_from_structure(repo_state)
            action = "plan_from_structure"
            summary = f"No smells — generated {len(tasks)} documentation task(s)"

        # ── 3. DAG validation ─────────────────────────────────────────────
        # Convert AgentTask → Task (Pydantic) for cross-validation, then
        # convert back.  This catches dangling edges and cycles at plan time,
        # not at execution time.
        pydantic_tasks = [
            Task(
                id=t.task_id,
                kind=t.kind,
                target=t.target,
                agent=t.assigned_to or "unassigned",
                priority=t.priority,
                status=t.status,
                depends_on=list(t.depends_on),
                notes=t.notes,
            )
            for t in tasks
        ]
        try:
            Task.validate_dag(pydantic_tasks)
        except ValueError as exc:
            logger.error("Planner produced invalid DAG: %s", exc)
            raise

        # ── 4. Log plan ───────────────────────────────────────────────────
        self._log_plan(tasks, replan=replan)

        # ── 5. Evolve state ───────────────────────────────────────────────
        # Preserve DONE tasks on replan; replace entirely on fresh plan.
        if replan:
            kept = [t for t in repo_state.tasks if t.status == TaskStatus.DONE]
            merged = kept + tasks
        else:
            merged = tasks

        return repo_state.evolve(
            agent_id=AGENT_ID,
            action=action,
            summary=summary,
            tasks=merged,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        repo_state: RepoState,
        smells: list[CodeSmell],
    ) -> list[str]:
        query = _build_retrieval_query(repo_state, smells)
        try:
            results = hybrid_retrieve(query, self.retriever, self.symbol_index)
            symbols = [r.get("symbol", "") for r in results if r.get("symbol")]
            logger.debug("Planner retrieved %d symbol(s) for query: %s", len(symbols), query)
            return symbols
        except Exception as exc:
            logger.warning("Planner retrieval failed (non-fatal): %s", exc)
            return []

    # ------------------------------------------------------------------
    # Task generation: smell path
    # ------------------------------------------------------------------

    def _tasks_from_smells(
        self,
        repo_state: RepoState,
        smells: list[CodeSmell],
        retrieved_symbols: list[str],
    ) -> list[AgentTask]:
        """
        One REFACTOR task + one DOC task per unique target method.

        Fix: originally created one task per smell, resulting in N refactor
        tasks for a method with N smells. A method can only be refactored once
        per run — multiple refactor tasks produce conflicting parallel edits.

        Correct behaviour: group all smells for the same target into a single
        refactor task whose notes describe all smells together. The Supervisor
        uses the prompt from RepoState.metadata (generated by PromptingEngine)
        which already covers all smells for the method.
        """
        tasks: list[AgentTask] = []

        # Group smells by target location
        by_target: dict[str, list[CodeSmell]] = {}
        for smell in smells:
            by_target.setdefault(smell.location, []).append(smell)

        for location, location_smells in by_target.items():
            # Use highest priority among all smells for this target
            priority = min(
                _compute_priority(s, repo_state) for s in location_smells
            )

            # Consolidate all smell notes into one description
            smell_summary = "; ".join(
                f"{s.smell_type}(sev={s.severity.name})"
                for s in sorted(location_smells, key=lambda s: s.severity, reverse=True)
            )

            fn = next(
                (f for f in repo_state.functions if f.name == location),
                None,
            )
            notes = (
                f"smells=[{smell_summary}] "
                f"smell_count={len(location_smells)}"
            )
            if fn:
                notes += (
                    f" | complexity={fn.complexity}"
                    f" loc={fn.loc}"
                    f" cohesion={fn.cohesion_score:.2f}"
                    f" callers={len(fn.called_by)}"
                )
            if retrieved_symbols:
                notes += f" | retrieved_context={retrieved_symbols[0]}"

            # ONE refactor task per target (covers all smells)
            refactor_id = _task_id("refactor", location)
            refactor_task = AgentTask(
                task_id=refactor_id,
                kind=TaskKind.REFACTOR,
                target=location,
                priority=priority,
                depends_on=[],
                status=TaskStatus.PENDING,
                notes=notes,
            )
            tasks.append(refactor_task)

            # ONE doc task per target (depends on refactor completing)
            doc_id = _task_id("doc", location)
            doc_task = AgentTask(
                task_id=doc_id,
                kind=TaskKind.DOCUMENT,
                target=location,
                priority=min(10, priority + 1),
                depends_on=[refactor_id],
                status=TaskStatus.PENDING,
                notes=f"Document post-refactor. {notes}",
            )
            tasks.append(doc_task)

        return tasks

    # ------------------------------------------------------------------
    # Task generation: no-smell structural path
    # ------------------------------------------------------------------

    def _tasks_from_structure(self, repo_state: RepoState) -> list[AgentTask]:
        """
        When no smells are detected, emit documentation tasks for functions
        that either lack a docstring or have high complexity (>= 5).
        Falls back to ALL functions so at least one task is always emitted.
        """
        targets = [
            fn for fn in repo_state.functions
            if not fn.docstring or fn.complexity >= 5
        ]
        if not targets:
            targets = list(repo_state.functions)
        targets.sort(key=lambda f: f.complexity, reverse=True)

        return [
            AgentTask(
                task_id=_task_id("doc", fn.name),
                kind=TaskKind.DOCUMENT,
                target=fn.name,
                priority=min(10, max(1, 5 - fn.complexity // 3)),
                depends_on=[],
                status=TaskStatus.PENDING,
                notes=(
                    f"No docstring. complexity={fn.complexity} "
                    f"loc={fn.loc} cohesion={fn.cohesion_score:.2f}"
                ),
            )
            for fn in targets
        ]

    # ------------------------------------------------------------------
    # Re-planning: evaluator feedback path
    # ------------------------------------------------------------------

    def _replan_tasks(
        self,
        repo_state: RepoState,
        retrieved_symbols: list[str],
    ) -> list[AgentTask]:
        """
        Emit retry tasks for FAILED tasks and for refactor results whose
        confidence fell below the threshold.

        Research basis (LLM Planner Agent survey, 2025): iterative re-planning
        with evaluation feedback is the standard pattern in plan-and-execute
        architectures; the planner should only re-emit tasks that genuinely
        failed, not the entire plan.
        """
        retry_tasks: list[AgentTask] = []

        # Collect targets with low-confidence refactor results
        low_conf_targets = {
            r.target_name
            for r in repo_state.refactor_results
            if r.confidence < _REPLAN_CONFIDENCE_THRESHOLD
        }

        # Re-emit FAILED tasks
        for task in repo_state.tasks:
            if task.status == TaskStatus.FAILED:
                retry_id = _task_id(task.kind.value, task.target, suffix="retry")
                retry_tasks.append(AgentTask(
                    task_id=retry_id,
                    kind=task.kind,
                    target=task.target,
                    priority=max(1, task.priority - 1),   # escalate on retry
                    depends_on=[],
                    status=TaskStatus.PENDING,
                    notes=f"[RETRY] Original task {task.task_id} failed. {task.notes}",
                ))

        # Re-emit refactor + doc for low-confidence results
        for target in low_conf_targets:
            refactor_retry_id = _task_id("refactor", target, suffix="lowconf")
            doc_retry_id = _task_id("doc", target, suffix="lowconf")
            ctx = retrieved_symbols[0] if retrieved_symbols else ""
            retry_tasks.extend([
                AgentTask(
                    task_id=refactor_retry_id,
                    kind=TaskKind.REFACTOR,
                    target=target,
                    priority=2,
                    depends_on=[],
                    status=TaskStatus.PENDING,
                    notes=(
                        f"[LOW_CONF_RETRY] Refactor confidence below "
                        f"{_REPLAN_CONFIDENCE_THRESHOLD}. "
                        f"retrieved_context={ctx}"
                    ),
                ),
                AgentTask(
                    task_id=doc_retry_id,
                    kind=TaskKind.DOCUMENT,
                    target=target,
                    priority=3,
                    depends_on=[refactor_retry_id],
                    status=TaskStatus.PENDING,
                    notes=f"[LOW_CONF_RETRY] Re-document after low-confidence refactor.",
                ),
            ])

        return retry_tasks

    # ------------------------------------------------------------------
    # Structured logging  (replaces print("DEBUG ---") calls)
    # ------------------------------------------------------------------

    def _log_plan(self, tasks: list[AgentTask], *, replan: bool) -> None:
        mode = "REPLAN" if replan else "PLAN"
        logger.info("Planner [%s] — %d task(s)", mode, len(tasks))
        for t in tasks:
            logger.debug(
                "  %s  kind=%-10s  target=%-40s  priority=%d  deps=%s",
                t.task_id,
                t.kind.value,
                t.target,
                t.priority,
                t.depends_on or "[]",
            )