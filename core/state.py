from __future__ import annotations

"""
state.py — Shared context model for an Autonomous Multi-Agent Platform
targeting Java code refactoring and documentation.

Design rationale
----------------
Research on LLM-based multi-agent systems (AutoGen, MetaGPT, SWE-bench
agents, CoDT, A-MOP) consistently shows that agent performance degrades
when the shared context is:
  - Structurally flat  (no call graph / dependency edges)
  - Semantically thin  (no cohesion metrics, no role signals)
  - Provenanceless     (no agent attribution, no confidence scores)
  - Monolithically hashed (execution state entangled with structural identity)

Every design decision below addresses one or more of those failure modes.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(int, Enum):
    """
    Typed severity replaces a bare int.

    Agents and evaluators compare Severity members symbolically, which
    prevents magic-number bugs and makes prompt-visible context self-describing
    (an LLM reading the state sees "HIGH" not "3").
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"

class TaskKind(str, Enum):
    REFACTOR = "refactor"
    DOCUMENT = "document"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"

# ---------------------------------------------------------------------------
# Stable hash utility
# ---------------------------------------------------------------------------

def _stable_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Core code units — richer relational context for agents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionUnit:
    """
    Represents a single Java method or top-level function.

    Research finding: agents tasked with smell detection and documentation
    perform significantly better when the context includes *relational* signals
    (what does this function call? what calls it?) alongside structural ones
    (lines, complexity). Call graph edges are the minimum viable relational
    context for refactoring agents (cf. CoDT, 2024; LLMRefactor, 2023).

    Fields
    ------
    name            Fully-qualified name, e.g. "com.example.Foo.process".
    params          Parameter names only; types live in the Java AST layer.
    docstring       Existing Javadoc, if any. Agents read this to avoid
                    overwriting intentional documentation.
    file_path       Relative path within the repo root.
    start_line      1-indexed, inclusive.
    end_line        1-indexed, inclusive.
    calls           Names of functions directly invoked in the body.
                    Enables call-graph traversal without a separate index.
    called_by       Reverse edges — callers of this function.
                    Critical for impact analysis before refactoring.
    complexity      McCabe cyclomatic complexity. Smell detectors use this
                    as a primary signal for "long method" and "complex method".
    loc             Lines of code (non-blank, non-comment).
    cohesion_score  LCOM variant in [0.0, 1.0]; 1.0 = perfectly cohesive.
                    Low cohesion is the primary signal for "feature envy"
                    and "god class" smells at the method level.
    """
    name: str
    params: list[str]
    docstring: str | None

    file_path: str = ""
    start_line: int = 0
    end_line: int = 0

    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)  # ← reverse edges added

    complexity: int = 0
    loc: int = 0
    cohesion_score: float = 1.0  # ← replaces bare loc as a smell signal

@dataclass(frozen=True)
class ClassUnit:
    """
    Represents a Java class or interface.

    Fields
    ------
    name            Fully-qualified class name.
    methods         All methods belonging to this class.
    docstring       Existing class-level Javadoc.
    file_path       Relative path within the repo root.
    superclass      Direct superclass name, if any.
    interfaces      Implemented interface names.
    is_abstract     True for abstract classes and interfaces.
    lcom            Class-level LCOM4 score. High LCOM → god-class smell.
    instability     Ce / (Ca + Ce) in [0.0, 1.0]. High instability + high
                    responsibility = fragility smell.
    """
    name: str
    methods: list[FunctionUnit]
    docstring: str | None

    file_path: str = ""
    superclass: str | None = None
    interfaces: list[str] = field(default_factory=list)
    is_abstract: bool = False
    lcom: float = 0.0          # ← class-level cohesion metric
    instability: float = 0.0   # ← coupling/stability metric

# ---------------------------------------------------------------------------
# Agent communication units — typed, attributed, and confidence-bearing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeSmell:
    """
    A smell finding produced by the AnalyzerAgent.

    Research finding: evaluator agents cannot meaningfully weight findings
    without knowing *which agent* produced them and *how confident* it was.
    Confidence also enables the Evaluator to request a second opinion only
    for borderline findings (confidence < 0.6), avoiding wasted LLM calls.

    Fields
    ------
    smell_type      e.g. "LongMethod", "GodClass", "FeatureEnvy".
    location        Fully-qualified name of the affected element.
    description     Natural-language explanation for the refactoring agent.
    severity        Typed enum — not a bare int.
    agent_id        ID of the agent that produced this finding.
    confidence      [0.0, 1.0]. Evaluator discards findings below threshold.
    reasoning       One-sentence chain-of-thought. Improves evaluator accuracy
                    and provides an audit trail.
    """
    smell_type: str
    location: str
    description: str
    severity: Severity

    agent_id: str = "analyzer"       # ← provenance
    confidence: float = 1.0          # ← confidence score
    reasoning: str = ""              # ← chain-of-thought trace

@dataclass(frozen=True)
class AgentTask:
    """
    A unit of work passed between agents.

    Replaces the opaque `List[Any]` tasks field.

    Research finding: typed task records with explicit priority and dependency
    edges are the minimum structure needed for a Planner agent to build a
    valid execution DAG. Without depends_on, agents execute in arbitrary order
    and produce conflicting edits (cf. A-MOP, 2024; MetaGPT ablation).

    Fields
    ------
    task_id         Stable identifier; used in depends_on edges.
    kind            What type of work this is.
    target          Fully-qualified name of the element to act on.
    priority        Lower = higher priority (like Unix nice).
    depends_on      task_ids that must be DONE before this task starts.
    status          Lifecycle state.
    assigned_to     Agent ID that owns this task, or None if unassigned.
    notes           Optional context injected by the Planner or Analyzer.
    """
    task_id: str
    kind: TaskKind
    target: str

    priority: int = 5
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    notes: str = ""

@dataclass(frozen=True)
class RefactorResult:
    """
    Outcome from the RefactorAgent for a single task.

    Fields
    ------
    task_id         Links back to the AgentTask that triggered this.
    target_name     Fully-qualified name of the refactored element.
    success         Whether the refactoring succeeded.
    changes         Unified diff or structured description of the edit.
    agent_id        Producing agent.
    confidence      Self-assessed confidence in the correctness of the edit.
    """
    task_id: str
    target_name: str
    success: bool
    changes: str

    agent_id: str = "refactor"
    confidence: float = 1.0

@dataclass(frozen=True)
class DocumentationResult:
    """
    Outcome from the DocAgent for a single target.

    Fields
    ------
    task_id         Links back to the AgentTask that triggered this.
    target_name     Fully-qualified name.
    docstring       Generated Javadoc.
    agent_id        Producing agent.
    confidence      Self-assessed confidence.
    """
    task_id: str
    target_name: str
    docstring: str

    agent_id: str = "doc"
    confidence: float = 1.0

@dataclass(frozen=True)
class ProvenanceEntry:
    """
    One record in the state's audit log.

    Research finding: multi-agent debugging is nearly impossible without
    provenance. A linear append-only log of which agent triggered each
    state transition, at what version, with what reasoning, reduces
    debugging time dramatically and enables replay (cf. LangGraph tracing).
    """
    version: int
    agent_id: str
    action: str        # e.g. "added_smell", "completed_task:task-42"
    summary: str = ""  # brief human-readable description

# ---------------------------------------------------------------------------
# Shared multi-agent state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoState:
    """
    The single shared context object passed between all agents.

    Design principles
    -----------------
    1. Immutable  — agents never mutate state in place. They call evolve()
       to produce a new version, making concurrent reads safe and enabling
       full history replay.

    2. Dual hashing — structural_hash captures only the parsed code and
       metadata (what the repo *is*). execution_hash captures the full
       state including agent outputs (what has been *done*). Separating
       these means a Planner can detect "same code, different execution
       stage" vs "code has changed", which matters for cache invalidation
       and incremental re-analysis.

    3. Typed tasks — AgentTask replaces List[Any], giving Planner agents
       a well-typed dependency graph to schedule against.

    4. Provenance log — append-only list of ProvenanceEntry records.
       Never cleared across versions.

    5. evaluation_scores keys convention:
         "<agent_id>/<metric>"  e.g. "refactor/correctness", "doc/coverage"
       This namespace prevents key collisions between agents.
    """

    # ---- Parsed repo ----
    raw_code: str
    classes: list[ClassUnit]
    functions: list[FunctionUnit]
    imports: list[str]
    metadata: dict[str, Any]

    # ---- Agent outputs ----
    smells: list[CodeSmell]
    tasks: list[AgentTask]           # ← typed, replaces List[Any]

    refactor_results: list[RefactorResult]
    documentation_results: list[DocumentationResult]

    evaluation_scores: dict[str, float]

    # ---- Execution tracking ----
    completed_tasks: list[str]       # task_ids only

    # ---- Provenance ----
    provenance_log: list[ProvenanceEntry] = field(default_factory=list)

    # ---- Version ----
    version: int = 0

    # ---- Computed hashes (set in __post_init__) ----
    structural_hash: str = field(init=False)
    execution_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # structural_hash: identity of the *code* only.
        # Changing a docstring or adding a smell must NOT change this hash.
        object.__setattr__(
            self,
            "structural_hash",
            _stable_hash({
                "raw_code": self.raw_code,
                "classes": [
                    asdict(c)
                    for c in sorted(self.classes, key=lambda x: x.name)
                ],
                "functions": [
                    asdict(f)
                    for f in sorted(self.functions, key=lambda x: x.name)
                ],
                "imports": sorted(self.imports),
                "metadata": self.metadata,
            }),
        )

        # execution_hash: full snapshot including agent outputs.
        # Used by Evaluator to detect whether a new run is needed.
        object.__setattr__(
            self,
            "execution_hash",
            _stable_hash({
                "structural_hash": self.structural_hash,
                "smells": [asdict(s) for s in self.smells],
                "tasks": [asdict(t) for t in self.tasks],
                "refactor_results": [asdict(r) for r in self.refactor_results],
                "documentation_results": [
                    asdict(d) for d in self.documentation_results
                ],
                "evaluation_scores": self.evaluation_scores,
                "completed_tasks": sorted(self.completed_tasks),
            }),
        )

    # -----------------------------------------------------------------------
    # Immutable state evolution
    # -----------------------------------------------------------------------

    def evolve(self, agent_id: str, action: str, summary: str = "", **kwargs) -> "RepoState":
        new_state = RepoState(
            raw_code=kwargs.get("raw_code", self.raw_code),
            classes=kwargs.get("classes", self.classes),
            functions=kwargs.get("functions", self.functions),
            imports=kwargs.get("imports", self.imports),
            metadata=kwargs.get("metadata", self.metadata),

            smells=kwargs.get("smells", self.smells),
            tasks=kwargs.get("tasks", self.tasks),

            refactor_results=kwargs.get("refactor_results", self.refactor_results),
            documentation_results=kwargs.get("documentation_results", self.documentation_results),

            # ✅ THIS IS THE FIX
            evaluation_scores=kwargs.get("evaluation_scores", self.evaluation_scores),

            completed_tasks=kwargs.get("completed_tasks", self.completed_tasks),

            provenance_log=self.provenance_log + [
                ProvenanceEntry(
                    version=self.version + 1,
                    agent_id=agent_id,
                    action=action,
                    summary=summary,
                )
            ],

            version=self.version + 1,


    )

        return new_state

    # -----------------------------------------------------------------------
    # Convenience query helpers — agents call these instead of filtering lists
    # -----------------------------------------------------------------------

    def pending_tasks(self) -> list[AgentTask]:
        """Tasks that are pending AND whose dependencies are all complete."""
        done = set(self.completed_tasks)
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in done for dep in t.depends_on)
        ]

    def smells_for(self, location: str) -> list[CodeSmell]:
        """All smells reported for a given fully-qualified location."""
        return [s for s in self.smells if s.location == location]

    def high_confidence_smells(self, threshold: float = 0.7) -> list[CodeSmell]:
        """Smells the Evaluator should act on without a second opinion."""
        return [s for s in self.smells if s.confidence >= threshold]

    def results_for_task(self, task_id: str) -> list[RefactorResult]:
        """All refactor results linked to a specific task."""
        return [r for r in self.refactor_results if r.task_id == task_id]

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_repo_state(
    raw_code: str,
    classes: list[ClassUnit],
    functions: list[FunctionUnit],
    imports: list[str],
    metadata: dict[str, Any] | None = None,
) -> RepoState:
    """
    Construct the initial RepoState for a newly ingested repository.

    The metadata dict is copied defensively to avoid shared-reference bugs
    when callers mutate their own dict after construction.
    """
    safe_metadata: dict[str, Any] = dict(metadata) if metadata else {}
    safe_metadata["repo_hash"] = hashlib.sha256(
        raw_code.encode()
    ).hexdigest()

    return RepoState(
        raw_code=raw_code,
        classes=classes,
        functions=functions,
        imports=imports,
        metadata=safe_metadata,
        smells=[],
        tasks=[],
        refactor_results=[],
        documentation_results=[],
        evaluation_scores={},
        completed_tasks=[],
        provenance_log=[],
        version=0,
    )