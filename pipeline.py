# pipeline.py — Orchestrated Multi-Agent Pipeline
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  INTEGRATION STRATEGY                                               │
# │                                                                     │
# │  This file runs TODAY using your friend's legacy models.            │
# │  Each new agent has a clearly marked TEST POINT (search "TP-N").    │
# │  When a module is ready, flip its AGENT_FLAGS entry to True         │
# │  and the pipeline automatically routes through the new agent.       │
# │                                                                     │
# │  Test points:                                                       │
# │    TP-1  core/state.py        → RepoState construction              │
# │    TP-2  core/hybrid_retriever→ symbol index build                  │
# │    TP-3  core/planner_agent   → task DAG generation                 │
# │    TP-4  core/supervisor      → agent dispatch + retry              │
# │    TP-5  core/evaluator       → score writing into RepoState        │
# └─────────────────────────────────────────────────────────────────────┘
#
# Flow:
#   input.java
#     → [Block 2] parser/java_parser.py          → parsed metrics
#     → [Block 3] prompt_engine/                 → smell report + 2 prompts
#     → [Block 4] RepoState + PlannerAgent        (TP-1, TP-2, TP-3)
#     → [Block 5] SupervisorAgent dispatch        (TP-4)
#         → agents/refactor_agent.py             → refactored Java
#         → agents/doc_agent.py                  → documentation
#     → [Block 6] EvaluatorAgent                 (TP-5)
#     → outputs/run_TIMESTAMP/
#
# Output files:
#   parsed_analysis.json          parser metrics per method
#   smell_report.txt              detected smells with reasons
#   refactoring_plan.txt          action plan per smell
#   PROMPT_refactor_agent.txt     exact prompt sent to Refactor Agent
#   PROMPT_doc_agent.txt          exact prompt sent to Doc Agent
#   refactored_code.java          Refactor Agent output
#   documentation.md              Doc Agent output
#   EVALUATION_refactor.txt       refactor evaluation (human-readable)
#   EVALUATION_refactor.json      refactor evaluation (machine-readable)
#   EVALUATION_doc.txt            doc evaluation (human-readable)
#   EVALUATION_doc.json           doc evaluation (machine-readable)
#   summary.json                  one-page summary of all scores
#   provenance_log.json           full agent-level audit trail
 
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
 
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
 
logger = logging.getLogger(__name__)
 
# ── Core (always available) ───────────────────────────────────────────────────
#from conftest import parsed
from parser.java_parser import JavaParser
from prompt_engine.prompting_engine import PromptingEngine
from smell_detector.smell_detector import SmellDetector
from agents.refactor_agent import RefactorAgent
from agents.doc_agent import DocAgent
from evaluator.refactor_evaluator import save_evaluation_report
from evaluator.doc_evaluator import save_doc_evaluation_report
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"        # ← add this too
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────
#  AGENT FEATURE FLAGS
#  Set a flag to True once the corresponding module is implemented and tested.
#  All False → pipeline runs identically to your current working version.
# ─────────────────────────────────────────────────────────────────────────────
AGENT_FLAGS = {
    "use_repo_state":    True,   # TP-1  core/state.py
    "use_symbol_index":  True,   # TP-2  core/hybrid_retriever.py
    "use_planner":       True,   # TP-3  core/planner_agent.py
    "use_supervisor":    True,   # TP-4  core/supervisor.py
    "use_evaluator":     True,   # TP-5  core/evaluator.py  (new EvaluatorAgent)
}
 
# ─────────────────────────────────────────────────────────────────────────────
#  Conditional imports — each wrapped so missing modules don't break the run
# ─────────────────────────────────────────────────────────────────────────────
_state_module = None
_retriever_module = None
_planner_class = None
_supervisor_class = None
_evaluator_class = None
 
if AGENT_FLAGS["use_repo_state"]:
    try:
        from core import state as _state_module
        from core.state import (
            create_repo_state, CodeSmell, Severity, FunctionUnit, ClassUnit,
        )
        logger.info("[AGENT] core/state.py loaded ✓")
    except ImportError as e:
        logger.error("[AGENT] core/state.py MISSING — flip use_repo_state=False. %s", e)
        AGENT_FLAGS["use_repo_state"] = False
 
if AGENT_FLAGS["use_symbol_index"]:
    try:
        from core.hybrid_retriever import hybrid_retrieve, build_symbol_index
        _retriever_module = True
        logger.info("[AGENT] core/hybrid_retriever.py loaded ✓")
    except ImportError as e:
        logger.error("[AGENT] core/hybrid_retriever.py MISSING — flip use_symbol_index=False. %s", e)
        AGENT_FLAGS["use_symbol_index"] = False
 
if AGENT_FLAGS["use_planner"]:
    try:
        from core.planner_agent import PlannerAgent as _PlannerClass
        print("PLANNER FILE:", _PlannerClass.__module__)
        _planner_class = _PlannerClass
        logger.info("[AGENT] core/planner_agent.py loaded ✓")
    except ImportError as e:
        logger.error("[AGENT] core/planner_agent.py MISSING — flip use_planner=False. %s", e)
        AGENT_FLAGS["use_planner"] = False
 
if AGENT_FLAGS["use_supervisor"]:
    try:
        from core.supervisor import SupervisorAgent as _SupervisorClass
        _supervisor_class = _SupervisorClass
        logger.info("[AGENT] core/supervisor.py loaded ✓")
    except ImportError as e:
        logger.error("[AGENT] core/supervisor.py MISSING — flip use_supervisor=False. %s", e)
        AGENT_FLAGS["use_supervisor"] = False
 
if AGENT_FLAGS["use_evaluator"]:
    try:
        from core.evaluator import EvaluatorAgent as _EvaluatorClass
        _evaluator_class = _EvaluatorClass
        logger.info("[AGENT] core/evaluator.py loaded ✓")
    except ImportError as e:
        logger.error("[AGENT] core/evaluator.py MISSING — flip use_evaluator=False. %s", e)
        AGENT_FLAGS["use_evaluator"] = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Helper: severity string → enum  (only used when TP-1 is active)
# ─────────────────────────────────────────────────────────────────────────────
def _severity_from_str(s: str):
    if not AGENT_FLAGS["use_repo_state"]:
        return s
    from core.state import Severity
    return {
        "critical": Severity.CRITICAL,
        "high":     Severity.HIGH,
        "medium":   Severity.MEDIUM,
        "low":      Severity.LOW,
    }.get(s.lower(), Severity.MEDIUM)
 
 
def _build_function_units(parsed: dict) -> list:
    """TP-1 helper — convert parser dicts → typed FunctionUnit instances.

    KEY MAPPING (parser output keys → FunctionUnit fields):
      lineno        → start_line
      param_count   → params (converted to list of placeholder names)
      conditional_count → complexity (best available proxy)
      loc           → loc  (unchanged)
    """
    if not AGENT_FLAGS["use_repo_state"]:
        return parsed.get("functions", [])
    from core.state import FunctionUnit
    return [
        FunctionUnit(
            name=f.get("name", "unknown"),
            # Parser gives param_count (int); FunctionUnit expects list[str]
            # Build placeholder param names so len(params) is correct
            params=[f"p{i}" for i in range(f.get("param_count", 0))],
            docstring=f.get("docstring"),
            file_path=f.get("file_path", ""),
            # Parser uses "lineno" not "start_line"
            start_line=f.get("lineno", f.get("start_line", 0)),
            end_line=f.get("end_line", 0),
            calls=f.get("calls", []),
            called_by=f.get("called_by", []),
            # Parser uses "conditional_count" not "complexity"
            complexity=f.get("conditional_count", f.get("complexity", 0)),
            loc=f.get("loc", 0),
            cohesion_score=f.get("cohesion_score", 1.0),
        )
        for f in parsed.get("functions", [])
    ]
 
 
def _build_class_units(parsed: dict) -> list:
    """TP-1 helper — convert parser dicts → typed ClassUnit instances."""
    if not AGENT_FLAGS["use_repo_state"]:
        return parsed.get("classes", [])
    from core.state import ClassUnit
    return [
        ClassUnit(
            name=c.get("name", "unknown"),
            methods=[],
            docstring=c.get("docstring"),
            file_path=c.get("file_path", ""),
            superclass=c.get("superclass"),
            interfaces=c.get("interfaces", []),
            is_abstract=c.get("is_abstract", False),
            lcom=c.get("lcom", 0.0),
            instability=c.get("instability", 0.0),
        )
        for c in parsed.get("classes", [])
    ]
 
 
def _build_code_smells(raw_smells: list[dict]) -> list:
    """TP-1 helper — convert SmellDetector raw dicts → typed CodeSmell instances."""
    if not AGENT_FLAGS["use_repo_state"]:
        return raw_smells   # pass raw dicts through
    from core.state import CodeSmell
    return [
        CodeSmell(
            smell_type=s.get("name", "unknown"),
            location=s.get("function", "unknown"),
            description=(
                f"{s.get('metric', '')}={s.get('value', '')} "
                f"(threshold={s.get('threshold', '')})"
            ),
            severity=_severity_from_str(s.get("severity", "medium")),
            agent_id="smell_detector",
            confidence=s.get("confidence", 1.0),
            reasoning=s.get("reason", ""),
        )
        for s in raw_smells
    ]
 
 
def _load_config() -> dict:
    cfg = ROOT / "config.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    return {}
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Simple provenance log used by LEGACY path (no RepoState)
# ─────────────────────────────────────────────────────────────────────────────
class _SimpleProvenanceLog:
    """Minimal audit trail for the legacy path so provenance_log.json is always written."""
 
    def __init__(self):
        self._entries = []
        self._version = 0
 
    def record(self, agent_id: str, action: str, summary: str):
        self._version += 1
        self._entries.append({
            "version": self._version,
            "agent":   agent_id,
            "action":  action,
            "summary": summary,
        })
 
    def to_list(self) -> list:
        return list(self._entries)
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═════════════════════════════════════════════════════════════════════════════
class Pipeline:
 
    def __init__(self):
        config = _load_config()
 
        self.refactor_adapter = str(
            ROOT / config.get("models", {}).get("refactor_adapter_path", "models/refactor_agent_final")
        )
        self.doc_adapter = str(
            ROOT / config.get("models", {}).get("doc_adapter_path", "models/doc_agent_final")
        )
        self.output_dir = ROOT / config.get("pipeline", {}).get("output_dir", "outputs")
        self.output_dir.mkdir(exist_ok=True, parents=True)
 
        dacos_path = config.get("dacos", {}).get("path", "") or None
        max_in     = config.get("models", {}).get("max_input_length",  512)
        max_out    = config.get("models", {}).get("max_output_length", 256)
 
        # ── YOUR FRIEND'S LEGACY MODELS (always initialised) ─────────────
        # These are the models that currently work. They remain active
        # regardless of which AGENT_FLAGS are set. Supervisor wraps them
        # when TP-4 is enabled; otherwise the pipeline calls them directly.
        self.parser         = JavaParser()
        self.engine         = PromptingEngine(model_type="codet5p-770m",
                                              dacos_folder=dacos_path)
        self.detector       = SmellDetector(dacos_folder=dacos_path)
        self.refactor_agent = RefactorAgent(
            adapter_path=self.refactor_adapter,
            max_input_length=max_in,
            max_output_length=max_out,
        )
        self.doc_agent = DocAgent(
            adapter_path=self.doc_adapter,
            max_input_length=max_in,
            max_output_length=256,
        )
 
        # Placeholders — filled lazily per run once symbol index exists
        self.retriever    = None
        self.symbol_index = None
 
        logger.info("[Pipeline] Initialised. Active flags: %s", AGENT_FLAGS)
        print("\n=== MODEL PATHS ===")
        print("Refactor model:", self.refactor_adapter)
        print("Doc model     :", self.doc_adapter)
        print("===================\n")
 
    # ─────────────────────────────────────────────────────────────────────
    def run(self, source_code: str, mode: str = "both", disable_llm: bool = False) -> dict:
 
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{ts}"
        run_dir.mkdir(exist_ok=True)
 
        result = {"timestamp": ts, "run_dir": str(run_dir),
                  "mode": mode, "success": False}
 
        plog = _SimpleProvenanceLog()   # lightweight log used by legacy path
        # ── DEBUG STEP TRACKER ─────────────────────────────
        step = 0
        def checkpoint(name):
            nonlocal step
            step += 1
            print(f"\n[STEP {step}] {name}")

        print("\n=== EXECUTION MODE ===")
        for k, v in AGENT_FLAGS.items():
            print(f"{k:20}: {v}")
        print("======================\n")
 
        # ══════════════════════════════════════════════════════════════════
        # BLOCK 2 — PARSE
        # ══════════════════════════════════════════════════════════════════
        checkpoint("Parsing input code")
        logger.info("[Block 2]  Parsing ...")
        parsed = self.parser.parse(source_code)
        print(f"Functions parsed: {len(parsed.get('functions', []))}")
        print(f"Classes parsed  : {len(parsed.get('classes', []))}")
        parsed["original_code"] = source_code
 
        if not parsed.get("parse_success", True):
            logger.error("Parse failed: %s", parsed.get("error"))
            return result
 
        logger.info(
            "           %d method(s)  %d class(es)  parser=%s",
            len(parsed["functions"]),
            len(parsed["classes"]),
            parsed["parser_used"],
        )
        plog.record("parser", "parse_complete",
                    f"{len(parsed['functions'])} functions, {len(parsed['classes'])} classes")
 
        (run_dir / "parsed_analysis.json").write_text(
            json.dumps({k: v for k, v in parsed.items() if k != "original_code"},
                       indent=2, default=str),
            encoding="utf-8",
        )
 
        # ══════════════════════════════════════════════════════════════════
        # BLOCK 3 — SMELL DETECTION + PROMPT GENERATION
        # ══════════════════════════════════════════════════════════════════
        checkpoint("Detecting smells + generating prompts")
        logger.info("[Block 3]  Detecting smells + generating prompts ...")
        raw_smells      = self.detector.detect_smells(parsed)
        print(f"Detected smells: {len(raw_smells)}")
        report          = self.detector.generate_report(parsed)
        plan            = self.engine.generate_refactoring_plan(parsed)
        prompts         = self.engine.generate_prompts(source_code, parsed, "both")
        refactor_prompt = prompts.get("refactor_prompt", "")
        doc_prompt      = prompts.get("documentation_prompt", "")
 
        logger.info("           %d smell(s) detected", len(raw_smells))
        for s in raw_smells:
            logger.info(
                "              [%s] %s — %s  (%s=%s, threshold=%s)",
                s["severity"], s["name"], s["function"],
                s["metric"], s["value"], s["threshold"],
            )
        plog.record("smell_detector", "smells_detected",
                    f"{len(raw_smells)} smell(s)")
 
        (run_dir / "smell_report.txt").write_text(report, encoding="utf-8")
        (run_dir / "refactoring_plan.txt").write_text(plan, encoding="utf-8")
        (run_dir / "PROMPT_refactor_agent.txt").write_text(refactor_prompt, encoding="utf-8")
        (run_dir / "PROMPT_doc_agent.txt").write_text(doc_prompt, encoding="utf-8")
        result["smells"] = [
            {"name": s["name"], "severity": s["severity"], "function": s["function"]}
            for s in raw_smells
        ]
 
        # ══════════════════════════════════════════════════════════════════
        # BLOCK 4 — REPOSTATE + TASK DAG
        # ══════════════════════════════════════════════════════════════════
        # ┌──────────────────────────────────────────────────────────────┐
        # │  TP-1  core/state.py → RepoState                            │
        # │  Set AGENT_FLAGS["use_repo_state"] = True to activate.      │
        # │  Expected: create_repo_state(...) returns a RepoState obj.  │
        # └──────────────────────────────────────────────────────────────┘
        checkpoint("Building RepoState")
        repo_state  = None
        smells = _build_code_smells(raw_smells)   # typed if TP-1 active
 
        if AGENT_FLAGS["use_repo_state"]:
            logger.info("[TP-1]    Building RepoState ...")
            try:
                from core.state import create_repo_state
                functions  = _build_function_units(parsed)
                classes    = _build_class_units(parsed)
                repo_state = create_repo_state(
                    raw_code=source_code,
                    classes=classes,
                    functions=functions,
                    imports=parsed.get("imports", []),
                    metadata={
                        "parser_used":     parsed.get("parser_used"),
                        "total_loc":       parsed.get("total_loc"),
                        "mode":            mode,
                        # Store prompts so Supervisor uses PromptingEngine format
                        # (matches CodeT5+ training format exactly)
                        "refactor_prompt": refactor_prompt,
                        "doc_prompt":      doc_prompt,
                    },
                )
                repo_state = repo_state.evolve(
                    agent_id="pipeline",
                    action="smells_attached",
                    summary=f"{len(smells)} smell(s) from SmellDetector",
                    smells=smells,
                )
                logger.info("[TP-1]    RepoState v%s created ✓", repo_state.version)
                plog.record("pipeline", "repo_state_created",
                            f"version={repo_state.version}")
            except Exception as exc:
                logger.error("[TP-1]    RepoState FAILED — falling back to legacy path. %s", exc)
                AGENT_FLAGS["use_repo_state"] = False
                repo_state = None
        else:
            logger.info("[Block 4]  RepoState disabled (TP-1 off) — using legacy dicts")
        print("\n================ REPO STATE DEBUG ================")

        if repo_state:
            print(f"Version      : {repo_state.version}")
            print(f"Functions    : {len(repo_state.functions)}")
            print(f"Classes      : {len(repo_state.classes)}")
            print(f"Smells       : {len(repo_state.smells)}")
            print(f"Tasks        : {len(getattr(repo_state, 'tasks', []))}")
        else:
            print("RepoState NOT created")

        print("=================================================\n")
        print("\n--- FUNCTIONS ---")
        for f in repo_state.functions[:3]:  # limit to first 3
            print(f"- {f.name} | LOC={f.loc} | complexity={f.complexity}")

        print("\n--- SMELLS ---")
        for s in repo_state.smells:
            print(f"- {s.smell_type} in {s.location} [{s.severity}]")

        print("\n--- METADATA ---")
        print(repo_state.metadata)
        print("=================================================\n")
        

        def serialize(obj):
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        with open(run_dir / "repo_state_debug.json", "w", encoding="utf-8") as f:
            json.dump(repo_state, f, default=serialize, indent=2)
 
        # ┌──────────────────────────────────────────────────────────────┐
        # │  TP-2  core/hybrid_retriever.py → symbol index              │
        # │  Set AGENT_FLAGS["use_symbol_index"] = True to activate.    │
        # │  Expected: build_symbol_index(parsed) returns an index obj. │
        # └──────────────────────────────────────────────────────────────┘
        checkpoint("Building Symbol Index")
        if AGENT_FLAGS["use_symbol_index"]:
            logger.info("[TP-2]    Building symbol index ...")
            try:
                from core.hybrid_retriever import build_symbol_index
                self.symbol_index = build_symbol_index(parsed)
                logger.info("[TP-2]    Symbol index built ✓ (%d symbols)",
                            len(self.symbol_index) if hasattr(self.symbol_index, "__len__") else -1)
                plog.record("hybrid_retriever", "symbol_index_built",
                            f"index ready")
            except Exception as exc:
                logger.error("[TP-2]    Symbol index FAILED. %s", exc)
                AGENT_FLAGS["use_symbol_index"] = False
        else:
            logger.info("[Block 4]  Symbol index disabled (TP-2 off)")
 
        # ┌──────────────────────────────────────────────────────────────┐
        # │  TP-3  core/planner_agent.py → task DAG                     │
        # │  Set AGENT_FLAGS["use_planner"] = True to activate.         │
        # │  Expected: planner.run(repo_state, code_smells) returns     │
        # │            updated RepoState with .tasks populated.         │
        # └──────────────────────────────────────────────────────────────┘
        checkpoint("Running PlannerAgent")
        if AGENT_FLAGS["use_planner"] and repo_state is not None:
            logger.info("[TP-3]    Running PlannerAgent ...")
            try:
                from core.planner_agent import PlannerAgent
                planner    = PlannerAgent(
                    engine=self.engine,
                    retriever=self.retriever,
                    symbol_index=self.symbol_index,
                )
                repo_state = planner.run(repo_state)
                print("\n=== AFTER PLANNER ===")
                print(f"Tasks: {len(repo_state.tasks)}")
                logger.info("[TP-3]    PlannerAgent produced %d task(s) ✓",
                            len(repo_state.tasks))
                plog.record("planner_agent", "dag_built",
                            f"{len(repo_state.tasks)} task(s)")
            except Exception as exc:
                logger.error("[TP-3]    PlannerAgent FAILED — skipping DAG. %s", exc)
                AGENT_FLAGS["use_planner"] = False
        elif AGENT_FLAGS["use_planner"] and repo_state is None:
            logger.warning("[TP-3]    Planner requested but RepoState is None — enable TP-1 first.")
        else:
            logger.info("[Block 4]  PlannerAgent disabled (TP-3 off)")


        # ══════════════════════════════════════════════════════════════════
        # BLOCK 5 — AGENT DISPATCH
        # ══════════════════════════════════════════════════════════════════
        # ┌──────────────────────────────────────────────────────────────┐
        # │  TP-4  core/supervisor.py → SupervisorAgent                 │
        # │  Set AGENT_FLAGS["use_supervisor"] = True to activate.      │
        # │  Expected: supervisor.run(repo_state) returns updated       │
        # │            RepoState with .refactor_results /               │
        # │            .documentation_results populated.                │
        # │                                                              │
        # │  LEGACY FALLBACK: calls refactor_agent + doc_agent directly │
        # │  (your friend's models) when supervisor is off.             │
        # └──────────────────────────────────────────────────────────────┘
        checkpoint("Dispatching agents (Supervisor or Legacy)")
        refactored_code = None
        documentation   = None
    
        if mode in ("refactor", "both", "document"):
 
            if AGENT_FLAGS["use_supervisor"] and repo_state is not None:
                # ── TP-4 ACTIVE: Supervisor wraps legacy agents ───────────
                logger.info("[TP-4]    SupervisorAgent dispatching tasks ...")
                try:
                    from core.supervisor import SupervisorAgent
                    supervisor = SupervisorAgent(
                        refactor_agent=self.refactor_agent,
                        doc_agent=self.doc_agent,
                        planner_agent=planner if AGENT_FLAGS["use_planner"] else None,
                    )
                    repo_state = supervisor.run(repo_state)
                    logger.info("[TP-4]    SupervisorAgent finished ✓")
                    plog.record("supervisor", "dispatch_complete",
                                "all tasks dispatched")
 
                    # Read outputs from RepoState
                    if repo_state.refactor_results:
                        refactored_code = repo_state.refactor_results[0].changes
                    if repo_state.documentation_results:
                        documentation = repo_state.documentation_results[0].docstring
 
                except Exception as exc:
                    logger.error("[TP-4]    SupervisorAgent FAILED — falling back to legacy. %s", exc)
                    AGENT_FLAGS["use_supervisor"] = False
                    # fall through to legacy below
 
            if not AGENT_FLAGS["use_supervisor"]:
                # ── LEGACY PATH: direct call to your friend's models ─────
                logger.info("[Block 5]  Legacy path — calling models directly ...")
                print("⚠️ Using LEGACY path (direct LLM calls)")
                if disable_llm:
                    logger.info("           🚫 LLM calls disabled — using mock outputs")
                    if mode in ("refactor", "both"):
                        refactored_code = source_code
                        plog.record("refactor_agent", "skipped", "LLM disabled")
                    if mode in ("document", "both"):
                        documentation = "# Mock Documentation\n\nLLM disabled."
                        plog.record("doc_agent", "skipped", "LLM disabled")
                else:
                    if mode in ("refactor", "both"):
                        logger.info("           Calling RefactorAgent (legacy) ...")
                        try:
                            refactor_result = self.refactor_agent.run(
                                prompt=refactor_prompt,
                                original_code=source_code,
                            )
                            refactored_code = refactor_result.get("refactored_code", "")
                            plog.record("refactor_agent", "refactored",
                                        f"output length={len(refactored_code)}")
                        except Exception as exc:
                            logger.error("           RefactorAgent call FAILED. %s", exc)

                    if mode in ("document", "both"):
                        logger.info("           Calling DocAgent (legacy) ...")
                        try:
                            doc_result = self.doc_agent.run(
                                prompt=doc_prompt,
                                parsed_code=parsed,   # ← pass parsed dict, not source_code
                            )
                            documentation = doc_result.get("documentation", "")
                            plog.record("doc_agent", "documented",
                                        f"output length={len(documentation)}")
                        except Exception as exc:
                            logger.error("           DocAgent call FAILED. %s", exc)
 
        # ══════════════════════════════════════════════════════════════════
        # BLOCK 6 — EVALUATION
        # ══════════════════════════════════════════════════════════════════
        # ┌──────────────────────────────────────────────────────────────┐
        # │  TP-5  core/evaluator.py → EvaluatorAgent                   │
        # │  Set AGENT_FLAGS["use_evaluator"] = True to activate.       │
        # │  Expected: evaluator.run(repo_state) returns                │
        # │            (updated_repo_state, eval_summary).              │
        # │  Legacy fallback: calls save_evaluation_report /            │
        # │                   save_doc_evaluation_report directly.      │
        # └──────────────────────────────────────────────────────────────┘
        checkpoint("Evaluating outputs")
        logger.info("[Block 6]  Evaluating results ...")
        eval_summary = None
 
        if AGENT_FLAGS["use_evaluator"] and repo_state is not None:
            try:
                from core.evaluator import EvaluatorAgent
                evaluator  = EvaluatorAgent(run_dir=run_dir)
                repo_state, eval_summary = evaluator.run(repo_state)
                logger.info("[TP-5]    EvaluatorAgent finished ✓  scores=%s",
                            repo_state.evaluation_scores)
                plog.record("evaluator_agent", "evaluation_complete",
                            f"scores={repo_state.evaluation_scores}")
            except Exception as exc:
                logger.error("[TP-5]    EvaluatorAgent FAILED — using legacy evaluators. %s", exc)
                AGENT_FLAGS["use_evaluator"] = False
 
        if not AGENT_FLAGS["use_evaluator"]:
            # ── LEGACY EVALUATORS ─────────────────────────────────────────
            if refactored_code:
                try:
                    save_evaluation_report(
                        original_code=source_code,
                        refactored_code=refactored_code,
                        run_dir=run_dir,
                    )
                    plog.record("refactor_evaluator", "evaluation_saved",
                                "EVALUATION_refactor.* written")
                except Exception as exc:
                    logger.error("           Refactor evaluation FAILED. %s", exc)
 
            if documentation:
                try:
                    save_doc_evaluation_report(
                        source_code=source_code,
                        documentation=documentation,
                        run_dir=run_dir,
                    )
                    plog.record("doc_evaluator", "evaluation_saved",
                                "EVALUATION_doc.* written")
                except Exception as exc:
                    logger.error("           Doc evaluation FAILED. %s", exc)
 
        # ══════════════════════════════════════════════════════════════════
        # BLOCK 7 — COLLECT RESULTS + SAVE
        # ══════════════════════════════════════════════════════════════════
        if refactored_code:
            (run_dir / "refactored_code.java").write_text(
                refactored_code, encoding="utf-8"
            )
            result["refactored_code"]     = refactored_code
            result["refactor_used_model"] = True
 
            # Read eval scores from EvaluatorAgent (TP-5) or legacy JSON
            result["refactor_evaluation"] = self._read_refactor_eval(
                run_dir, eval_summary
            )
 
        if documentation:
            (run_dir / "documentation.md").write_text(
                documentation, encoding="utf-8"
            )
            result["documentation"]  = documentation
            result["doc_used_model"] = True
 
            result["doc_evaluation"] = self._read_doc_eval(
                run_dir, eval_summary
            )
 
        result["success"] = True
 
        # Final summary + print
        self._save_summary(result, run_dir, parsed, repo_state, plog)
        self._print_final(result, run_dir)
 
        # ── Provenance log ────────────────────────────────────────────────
        # If RepoState is active its log takes precedence; otherwise use
        # the simple plog collected along the legacy path.
        if repo_state is not None and hasattr(repo_state, "provenance_log"):
            provenance = [
                {"version": e.version, "agent": e.agent_id,
                 "action": e.action, "summary": e.summary}
                for e in repo_state.provenance_log
            ]
        else:
            provenance = plog.to_list()
 
        with open(run_dir / "provenance_log.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        self._last_repo_state = repo_state  # store for potential inspection after the run
        return result
 
    # ─────────────────────────────────────────────────────────────────────
    #  Eval helpers (read from new EvaluatorAgent OR legacy JSON files)
    # ─────────────────────────────────────────────────────────────────────
    def _read_refactor_eval(self, run_dir: Path, eval_summary) -> dict | None:
        """Pull refactor scores from EvaluatorAgent summary or legacy JSON."""
        if eval_summary is not None and hasattr(eval_summary, "refactor") \
                and eval_summary.refactor:
            re = eval_summary.refactor
            return {
                "confidence_score":  re.confidence,
                "confidence_status": re.status,
                "ast_valid":         bool(re.raw.get("ast_validity", {}).get("valid")),
                "style_score":       re.raw.get("style_metrics", {}).get("score", 0.0),
                "codebleu":          re.raw.get("codebleu_score", 0.0),
                "semantic_score":    re.raw.get("semantic_preservation", {}).get("score", 0.0),
                "improvement_score": re.raw.get("improvement", {}).get("score", 0.0),
                "loc_original":      re.raw.get("improvement", {}).get("loc_original", 0),
                "loc_refactored":    re.raw.get("improvement", {}).get("loc_refactored", 0),
            }
        # Fall back to legacy JSON written by save_evaluation_report
        legacy = run_dir / "EVALUATION_refactor.json"
        if legacy.exists():
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                return {
                    "confidence_score":  data.get("confidence", 0.0),
                    "confidence_status": data.get("status", "unknown"),
                    "ast_valid":         data.get("ast_valid", False),
                    "style_score":       data.get("style_score", 0.0),
                    "codebleu":          data.get("codebleu", 0.0),
                    "semantic_score":    data.get("semantic_score", 0.0),
                    "improvement_score": data.get("improvement_score", 0.0),
                    "loc_original":      data.get("loc_original", 0),
                    "loc_refactored":    data.get("loc_refactored", 0),
                }
            except Exception:
                pass
        return None
 
    def _read_doc_eval(self, run_dir: Path, eval_summary) -> dict | None:
        """Pull doc scores from EvaluatorAgent summary or legacy JSON."""
        if eval_summary is not None and hasattr(eval_summary, "doc") \
                and eval_summary.doc:
            de = eval_summary.doc
            return {
                "confidence_score":   de.confidence,
                "confidence_status":  de.status,
                "coverage_score":     de.raw.get("coverage", {}).get("score", 0.0),
                "completeness_score": de.raw.get("completeness", {}).get("score", 0.0),
            }
        legacy = run_dir / "EVALUATION_doc.json"
        if legacy.exists():
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                return {
                    "confidence_score":   data.get("confidence", 0.0),
                    "confidence_status":  data.get("status", "unknown"),
                    "coverage_score":     data.get("coverage_score", 0.0),
                    "completeness_score": data.get("completeness_score", 0.0),
                }
            except Exception:
                pass
        return None
 
    # ─────────────────────────────────────────────────────────────────────
    def _save_summary(self, result, run_dir, parsed, repo_state, plog):
        summary = {
            "timestamp": result["timestamp"],
            "mode":      result["mode"],
            "agent_flags_active": {k: v for k, v in AGENT_FLAGS.items() if v},
            "input": {
                "parser":  parsed.get("parser_used"),
                "loc":     parsed.get("total_loc"),
                "methods": len(parsed.get("functions", [])),
                "classes": len(parsed.get("classes", [])),
                "smells":  result.get("smells", []),
            },
            "refactor":          result.get("refactor_evaluation"),
            "doc":               result.get("doc_evaluation"),
            "evaluation_scores": (
                repo_state.evaluation_scores
                if repo_state and hasattr(repo_state, "evaluation_scores")
                else {}
            ),
            "tasks_completed": (
                len(repo_state.completed_tasks)
                if repo_state and hasattr(repo_state, "completed_tasks")
                else "n/a"
            ),
            "tasks_total": (
                len(repo_state.tasks)
                if repo_state and hasattr(repo_state, "tasks")
                else "n/a"
            ),
            "state_version": (
                repo_state.version
                if repo_state and hasattr(repo_state, "version")
                else "legacy"
            ),
            "files": sorted(p.name for p in run_dir.iterdir()),
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
 
    # ─────────────────────────────────────────────────────────────────────
    def _print_final(self, result, run_dir):
        print("\n" + "=" * 65)
        print("  FINAL RESULT")
        print("=" * 65)
 
        active = [k for k, v in AGENT_FLAGS.items() if v]
        print(f"\n  Active agents : {active if active else ['legacy path']}")
 
        if "refactor_evaluation" in result and result["refactor_evaluation"]:
            re  = result["refactor_evaluation"]
            loc = f"{re['loc_original']} → {re['loc_refactored']}"
            v   = ("ACCEPTED"              if re["confidence_score"] >= 0.75
                   else "CONDITIONALLY ACCEPTED" if re["confidence_score"] >= 0.65
                   else "REJECTED")
            print(f"\n  REFACTORING")
            print(f"     Confidence  : {re['confidence_score']:.3f}  {re['confidence_status']}")
            print(f"     AST Valid   : {'YES' if re['ast_valid'] else 'NO'}")
            print(f"     Style       : {re['style_score']:.3f}")
            print(f"     CodeBLEU    : {re['codebleu']:.3f}")
            print(f"     Semantic    : {re['semantic_score']:.3f}")
            print(f"     Improvement : {re['improvement_score']:.3f}")
            print(f"     LOC change  : {loc}")
            print(f"     Verdict     : {v}")
 
        if "doc_evaluation" in result and result["doc_evaluation"]:
            de = result["doc_evaluation"]
            v  = ("ACCEPTED"              if de["confidence_score"] >= 0.75
                  else "CONDITIONALLY ACCEPTED" if de["confidence_score"] >= 0.65
                  else "REJECTED")
            print(f"\n  DOCUMENTATION")
            print(f"     Confidence   : {de['confidence_score']:.3f}  {de['confidence_status']}")
            print(f"     Coverage     : {de['coverage_score']:.3f}")
            print(f"     Completeness : {de['completeness_score']:.3f}")
            print(f"     Verdict      : {v}")
 
        print(f"\n  Outputs : {run_dir}")
        print(f"\n  Files:")
        descs = {
            "parsed_analysis.json":      "Parser metrics (LOC, params, conditionals)",
            "smell_report.txt":          "Detected smells with reasons",
            "refactoring_plan.txt":      "Action plan per smell",
            "PROMPT_refactor_agent.txt": "Exact prompt sent to Refactor Agent",
            "PROMPT_doc_agent.txt":      "Exact prompt sent to Doc Agent",
            "refactored_code.java":      "Refactor Agent output",
            "documentation.md":          "Doc Agent output",
            "EVALUATION_refactor.txt":   "Refactor evaluation (human-readable)",
            "EVALUATION_refactor.json":  "Refactor scores (machine-readable)",
            "EVALUATION_doc.txt":        "Doc evaluation (human-readable)",
            "EVALUATION_doc.json":       "Doc scores (machine-readable)",
            "summary.json":              "One-page summary of all scores",
            "provenance_log.json":       "Agent-level audit trail",
        }
        for f in sorted(run_dir.iterdir()):
            desc = descs.get(f.name, "")
            print(f"    {f.name:<42} {desc}")
 
        print("=" * 65)