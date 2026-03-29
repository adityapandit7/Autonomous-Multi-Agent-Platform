from __future__ import annotations

import sys as _sys, os as _os
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

"""
evaluator_agent.py — Evaluator Agent

Position in pipeline (architecture diagram)
--------------------------------------------
  Supervisor → RefactorAgent | DocAgent
      ↓               ↓
          → [THIS FILE] EvaluatorAgent
              ↓                    ↓
      Update RepoState      Feedback loop → Planner (replan=True)

Role
----
The Evaluator is NOT a new evaluator — the metric logic already exists in:
  evaluator/refactor_evaluator.py  (analyze_refactoring, save_evaluation_report)
  evaluator/doc_evaluator.py       (evaluate_documentation, save_doc_evaluation_report)

What was missing was the *agent wrapper* that:
  1. Runs both evaluators against the results stored in RepoState.
  2. Writes scores back into RepoState.evaluation_scores via evolve() so the
     Supervisor's feedback loop has something to read.
  3. Emits a structured EvaluationSummary so pipeline.py and the Supervisor
     can make routing decisions without parsing raw eval dicts.
  4. Saves evaluation artefacts to the run directory (mirrors what pipeline.py
     was doing inline, now done in one place).

Research basis
--------------
- Evaluation-Driven Development (arXiv 2411.13768, 2025): evaluators must
  feed results back into the agent loop as structured signals, not just write
  reports to disk. Confidence thresholds gate progression to the next stage.
- Multi-dimensional eval frameworks (Galileo, 2025; EmergentMind 2025):
  evaluation must cover correctness, style, semantic preservation, and
  documentation quality — not just a single score.
- CI/CD-integrated evals (DEV Community, 2025): deterministic checks
  (AST validity, JSON schema) run first; statistical/model-based metrics run
  second — fail fast on hard errors before spending time on soft metrics.
- Human escalation (arXiv 2411.13768, 2025): when confidence is in a
  borderline band, the system should flag for human review rather than
  silently accepting or rejecting.

evaluation_scores key namespace
---------------------------------
  "refactor/<metric>"  e.g. "refactor/confidence", "refactor/codebleu"
  "doc/<metric>"       e.g. "doc/confidence", "doc/coverage"

This matches the convention documented in state.py.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluator.refactor_evaluator import analyze_refactoring, save_evaluation_report
from evaluator.doc_evaluator import evaluate_documentation, save_doc_evaluation_report
from core.state import RepoState, TaskKind

logger = logging.getLogger(__name__)

AGENT_ID = "evaluator"

# ---------------------------------------------------------------------------
# Thresholds  (mirrors pipeline.py's _print_final verdict logic)
# ---------------------------------------------------------------------------

REFACTOR_ACCEPT    = 0.75   # confidence >= this → ACCEPTED
REFACTOR_COND      = 0.65   # confidence >= this → CONDITIONALLY ACCEPTED
DOC_ACCEPT         = 0.75
DOC_COND           = 0.65

# Borderline band: flag for human review but don't block the pipeline
HUMAN_REVIEW_BAND  = (0.60, 0.65)

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@dataclass
class AgentVerdict:
    """
    Per-agent evaluation verdict.

    Fields
    ------
    confidence      Overall confidence score in [0.0, 1.0].
    status          Human-readable status string from the underlying evaluator.
    accepted        True if confidence >= the ACCEPT threshold.
    conditional     True if confidence is in the CONDITIONALLY ACCEPTED band.
    needs_replan    True if confidence is below the replan threshold.
    needs_human     True if confidence is in the borderline human-review band.
    raw             Full raw evaluation dict from the underlying evaluator.
    """
    confidence: float
    status: str
    accepted: bool
    conditional: bool
    needs_replan: bool
    needs_human: bool
    raw: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationSummary:
    """
    Structured output of EvaluatorAgent.run().

    Consumers (Supervisor, pipeline.py) read this instead of parsing raw dicts.
    """
    refactor: AgentVerdict | None = None
    doc: AgentVerdict | None = None

    @property
    def any_needs_replan(self) -> bool:
        return (
            (self.refactor is not None and self.refactor.needs_replan)
            or (self.doc is not None and self.doc.needs_replan)
        )

    @property
    def any_needs_human(self) -> bool:
        return (
            (self.refactor is not None and self.refactor.needs_human)
            or (self.doc is not None and self.doc.needs_human)
        )

# ---------------------------------------------------------------------------
# Evaluator Agent
# ---------------------------------------------------------------------------

class EvaluatorAgent:
    """
    Wraps the existing refactor and doc evaluators with RepoState write-back.

    Parameters
    ----------
    run_dir     Path to the current run's output directory.  Evaluation
                artefacts (.txt, .json) are written here, matching the
                file names pipeline.py already expects.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, repo_state):
        """
        Evaluate all completed results in repo_state.
        Returns (new_state, summary) with evaluation_scores populated.
        """
        scores = dict(repo_state.evaluation_scores)
        summary = EvaluationSummary()

        # Evaluate refactor results
        if repo_state.refactor_results:
            refactor_verdict = self._evaluate_refactor(repo_state, scores)
            summary.refactor = refactor_verdict
            raw = refactor_verdict.raw
            scores["refactor/confidence"]     = refactor_verdict.confidence
            scores["refactor/ast_valid"]      = float(raw.get("ast_validity", {}).get("valid", False))
            scores["refactor/style"]          = raw.get("style_metrics", {}).get("score", 0.0)
            scores["refactor/codebleu"]       = raw.get("codebleu_score", 0.0)
            scores["refactor/semantic"]       = raw.get("semantic_preservation", {}).get("score", 0.0)
            scores["refactor/improvement"]    = raw.get("improvement", {}).get("score", 0.0)
            scores["refactor/loc_original"]   = raw.get("improvement", {}).get("loc_original", 0)
            scores["refactor/loc_refactored"] = raw.get("improvement", {}).get("loc_refactored", 0)
            logger.info("Refactor evaluation: confidence=%.3f", refactor_verdict.confidence)

        # Evaluate documentation results
        if repo_state.documentation_results:
            doc_verdict = self._evaluate_doc(repo_state, scores)
            summary.doc = doc_verdict
            raw = doc_verdict.raw
            scores["doc/confidence"]   = doc_verdict.confidence
            scores["doc/coverage"]     = raw.get("coverage", {}).get("score", 0.0)
            scores["doc/completeness"] = raw.get("completeness", {}).get("score", 0.0)
            logger.info("Doc evaluation: confidence=%.3f", doc_verdict.confidence)

        if summary.any_needs_human:
            logger.warning("One or more results in the borderline band — consider human review.")

        parts = []
        if summary.refactor:
            parts.append(f"refactor_conf={summary.refactor.confidence:.3f}")
        if summary.doc:
            parts.append(f"doc_conf={summary.doc.confidence:.3f}")

        new_state = repo_state.evolve(
            agent_id=AGENT_ID,
            action="evaluation_complete",
            summary=" ".join(parts) or "partial evaluation",
            evaluation_scores=scores,
        )
        return new_state, summary

    # ------------------------------------------------------------------
    # Refactor evaluation
    # ------------------------------------------------------------------

    def _evaluate_refactor(
        self,
        repo_state: RepoState,
        scores: dict[str, float],
    ) -> AgentVerdict:
        """
        Run analyze_refactoring() against all refactor results in state.

        For multi-result runs (post-replan), we evaluate each result and
        aggregate by taking the minimum confidence — the weakest result
        determines whether replanning is needed.

        Research (arXiv 2411.13768): evaluate at the system level, not just
        per-output; the weakest link determines overall pipeline confidence.
        """
        all_evals: list[dict] = []

        for r_result in repo_state.refactor_results:
            raw_eval = analyze_refactoring(
                repo_state.raw_code,
                r_result.changes,
            )
            all_evals.append(raw_eval)

            # Save per-result artefacts
            txt_path = self.run_dir / f"EVALUATION_refactor_{r_result.task_id}.txt"
            json_path = self.run_dir / f"EVALUATION_refactor_{r_result.task_id}.json"

            save_evaluation_report(
                raw_eval,
                repo_state.raw_code,
                r_result.changes,
                str(txt_path),
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(raw_eval, f, indent=2, default=str)

        # Aggregate: minimum confidence across all results (weakest-link)
        # Also save a combined summary at the legacy filename for compatibility
        primary = all_evals[0]  # first result = primary for the combined report
        if len(repo_state.refactor_results) == 1:
            save_evaluation_report(
                primary,
                repo_state.raw_code,
                repo_state.refactor_results[0].changes,
                str(self.run_dir / "EVALUATION_refactor.txt"),
            )
            with open(self.run_dir / "EVALUATION_refactor.json", "w", encoding="utf-8") as f:
                json.dump(primary, f, indent=2, default=str)

        min_conf = min(e["confidence"]["score"] for e in all_evals)
        primary_conf = primary["confidence"]["score"]

        # Write to shared scores dict (namespace: "refactor/<metric>")

        return self._make_verdict(
            confidence=min_conf,
            status=primary["confidence"]["status"],
            accept_threshold=REFACTOR_ACCEPT,
            cond_threshold=REFACTOR_COND,
            raw=primary,
        )

    # ------------------------------------------------------------------
    # Doc evaluation
    # ------------------------------------------------------------------

    def _evaluate_doc(
        self,
        repo_state: RepoState,
        scores: dict[str, float],
    ) -> AgentVerdict:
        """
        Run evaluate_documentation() against all doc results in state.

        Uses the same parsed context dict that DocAgent used, reconstructed
        from RepoState so no re-parsing is needed.
        """
        parsed_ctx = self._build_parsed_context(repo_state)
        all_evals: list[dict] = []

        for d_result in repo_state.documentation_results:
            raw_eval = evaluate_documentation(d_result.docstring, parsed_ctx)
            all_evals.append(raw_eval)

            txt_path  = self.run_dir / f"EVALUATION_doc_{d_result.task_id}.txt"
            json_path = self.run_dir / f"EVALUATION_doc_{d_result.task_id}.json"

            save_doc_evaluation_report(raw_eval, d_result.docstring, str(txt_path))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(raw_eval, f, indent=2, default=str)

        primary = all_evals[0]
        if len(repo_state.documentation_results) == 1:
            save_doc_evaluation_report(
                primary,
                repo_state.documentation_results[0].docstring,
                str(self.run_dir / "EVALUATION_doc.txt"),
            )
            with open(self.run_dir / "EVALUATION_doc.json", "w", encoding="utf-8") as f:
                json.dump(primary, f, indent=2, default=str)

        min_conf = min(e["confidence"]["score"] for e in all_evals)


        return self._make_verdict(
            confidence=min_conf,
            status=primary["confidence"]["status"],
            accept_threshold=DOC_ACCEPT,
            cond_threshold=DOC_COND,
            raw=primary,
        )

    # ------------------------------------------------------------------
    # Verdict factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_verdict(
        confidence: float,
        status: str,
        accept_threshold: float,
        cond_threshold: float,
        raw: dict,
    ) -> AgentVerdict:
        return AgentVerdict(
            confidence=confidence,
            status=status,
            accepted=confidence >= accept_threshold,
            conditional=cond_threshold <= confidence < accept_threshold,
            needs_replan=confidence < cond_threshold,
            needs_human=HUMAN_REVIEW_BAND[0] <= confidence < HUMAN_REVIEW_BAND[1],
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Context helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parsed_context(repo_state: RepoState) -> dict:
        """
        Reconstruct the 'parsed' dict that evaluate_documentation() expects,
        sourced from RepoState so no re-parsing is needed.
        """
        return {
            "functions": [
                {
                    "name":       f.name,
                    "params":     f.params,
                    "docstring":  f.docstring,
                    "complexity": f.complexity,
                    "loc":        f.loc,
                }
                for f in repo_state.functions
            ],
            "classes": [
                {"name": c.name, "docstring": c.docstring}
                for c in repo_state.classes
            ],
            "imports": repo_state.imports,
        }