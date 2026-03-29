"""
pipeline_api.py — Callable wrapper around Pipeline for HTTP/API use.

Drop this file next to pipeline.py and main.py (project root).

Public surface
--------------
run_pipeline(source_code, mode, filename) -> PipelineResult (dict)

PipelineResult schema
---------------------
{
  "success": bool,
  "mode": "refactor" | "document" | "both",
  "filename": str,
  "originalContent": str,
  "processedContent": str,          # primary output (for backward-compat)
  "refactoredContent": str | None,  # set when mode in ("refactor", "both")
  "documentedContent": str | None,  # set when mode in ("document", "both")
  "artifacts": {
    "runDir": str | None,
    "parsedAnalysis": object | None,
    "smellReport": str | None,
    "refactoringPlan": str | None,
    "prompts": {
      "refactor": str | None,
      "doc": str | None
    },
    "evaluation": {
      "refactor": object | None,
      "doc": object | None,
      "summary": object | None
    }
  },
  "error": None
}

Error result schema
-------------------
{
  "success": False,
  "error": {
    "message": str,
    "type": "validation" | "runtime" | "pipeline",
    "details": str | None
  }
}
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Literal

# ── Silence noisy loggers before importing pipeline ──────────────────────────
import os
import warnings
import logging

os.environ.setdefault("TRANSFORMERS_OFFLINE",          "1")
os.environ.setdefault("HF_HUB_OFFLINE",               "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",        "false")
warnings.filterwarnings("ignore")

for _lib in ["transformers", "peft", "torch", "safetensors", "accelerate",
             "huggingface_hub", "httpx", "httpcore", "urllib3", "filelock"]:
    logging.getLogger(_lib).setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────

VALID_MODES = {"refactor", "document", "both"}

# Module-level singleton so the heavy model weights are loaded only once.
_pipeline_instance = None


def _get_pipeline():
    """Return a cached Pipeline instance (loads models on first call)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        from pipeline import Pipeline          # noqa: PLC0415
        _pipeline_instance = Pipeline()
    return _pipeline_instance


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    source_code: str | None,
    mode: str | None,
) -> str | None:
    """Return an error message string, or None if inputs are valid."""
    if not source_code or not source_code.strip():
        return "source_code must be a non-empty string"
    if mode not in VALID_MODES:
        return f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers — read files written by Pipeline.run()
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except Exception:
        return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def _collect_artifacts(run_dir_str: str | None, raw_result: dict) -> dict:
    """
    Build the `artifacts` sub-object from the run directory + raw pipeline result.

    Parameters
    ----------
    run_dir_str : path string returned by pipeline.run() (may be None on early failure)
    raw_result  : the dict returned by pipeline.run()
    """
    artifacts: dict = {
        "runDir":         run_dir_str,
        "parsedAnalysis": None,
        "smellReport":    None,
        "refactoringPlan": None,
        "prompts": {"refactor": None, "doc": None},
        "evaluation": {"refactor": None, "doc": None, "summary": None},
    }

    if not run_dir_str:
        return artifacts

    run_dir = Path(run_dir_str)

    artifacts["parsedAnalysis"] = _read_json(run_dir / "parsed_analysis.json")
    artifacts["smellReport"]    = _read_text(run_dir / "smell_report.txt")
    artifacts["refactoringPlan"] = _read_text(run_dir / "refactoring_plan.txt")
    artifacts["prompts"]["refactor"] = _read_text(run_dir / "PROMPT_refactor_agent.txt")
    artifacts["prompts"]["doc"]      = _read_text(run_dir / "PROMPT_doc_agent.txt")

    artifacts["evaluation"]["refactor"] = raw_result.get("refactor_evaluation")
    artifacts["evaluation"]["doc"]      = raw_result.get("doc_evaluation")
    artifacts["evaluation"]["summary"]  = _read_json(run_dir / "summary.json")

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    source_code: str,
    mode: Literal["refactor", "document", "both"],
    filename: str = "uploaded_file.java",
    disable_llm: bool = False,
) -> dict:
    """
    Run the multi-agent pipeline and return a JSON-serialisable dict.

    Parameters
    ----------
    source_code : raw Java source code (string)
    mode        : "refactor" | "document" | "both"
    filename    : original filename (for display only, not written to disk here)
    disable_llm : pass True to skip actual LLM calls (dry-run / testing)

    Returns
    -------
    dict conforming to PipelineResult schema (see module docstring)
    """
    # ── Input validation ──────────────────────────────────────────────────
    err = _validate_inputs(source_code, mode)
    if err:
        return {
            "success": False,
            "error": {"message": err, "type": "validation", "details": None},
        }

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        pipeline  = _get_pipeline()
        raw       = pipeline.run(source_code, mode=mode, disable_llm=disable_llm)
    except Exception as exc:
        return {
            "success": False,
            "error": {
                "message": str(exc),
                "type": "pipeline",
                "details": traceback.format_exc(),
            },
        }

    if not raw.get("success"):
        return {
            "success": False,
            "error": {
                "message": "Pipeline completed but reported failure",
                "type": "pipeline",
                "details": raw,
            },
        }

    # ── Extract outputs ───────────────────────────────────────────────────
    # Prefer reading from the files pipeline.run() wrote to disk —
    # these are the canonical outputs (refactored_code.java, documentation.md)
    run_dir_str        = raw.get("run_dir")
    run_dir            = Path(run_dir_str) if run_dir_str else None

    refactored_content = None
    documented_content = None

    if run_dir:
        rf = run_dir / "refactored_code.java"
        dc = run_dir / "documentation.md"
        if rf.exists():
            refactored_content = rf.read_text(encoding="utf-8")
        if dc.exists():
            documented_content = dc.read_text(encoding="utf-8")

    # Fall back to what pipeline.run() returned in memory
    if refactored_content is None:
        refactored_content = raw.get("refactored_code")
    if documented_content is None:
        documented_content = raw.get("documentation")

    # processedContent: primary output (backward-compat with legacy server.js)
    if mode == "refactor":
        processed_content = refactored_content or source_code
    elif mode == "document":
        processed_content = documented_content or source_code
    else:  # both
        processed_content = documented_content or refactored_content or source_code

    # ── Assemble artifacts ────────────────────────────────────────────────
    artifacts = _collect_artifacts(raw.get("run_dir"), raw)

    return {
        "success":          True,
        "mode":             mode,
        "filename":         filename,
        "originalContent":  source_code,
        "processedContent": processed_content,
        "refactoredContent": refactored_content,
        "documentedContent": documented_content,
        "artifacts":        artifacts,
        "error":            None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: thin Flask/FastAPI HTTP service
# Run: python pipeline_api.py --serve [--port 5001]
# ─────────────────────────────────────────────────────────────────────────────

def _start_http_server(port: int = 5001):
    """
    Minimal Flask HTTP service so Node can proxy to Python instead of
    spawning a child process.  Start with:  python pipeline_api.py --serve
    """
    try:
        from flask import Flask, request, jsonify   # pip install flask
    except ImportError:
        raise SystemExit("Flask not installed.  Run: pip install flask")

    app = Flask(__name__)

    @app.post("/run")
    def run():
        """
        POST /run
        Body (JSON): { "sourceCode": str, "mode": str, "filename": str }
        Returns: PipelineResult JSON
        """
        data        = request.get_json(force=True, silent=True) or {}
        source_code = data.get("sourceCode", "")
        mode        = data.get("mode", "")
        filename    = data.get("filename", "uploaded.java")
        disable_llm = bool(data.get("disableLlm", False))

        result = run_pipeline(source_code, mode, filename, disable_llm)
        status = 200 if result["success"] else (
            400 if result.get("error", {}).get("type") == "validation" else 500
        )
        return jsonify(result), status

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    print(f"[pipeline_api] HTTP service starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser()
    p.add_argument("--serve",  action="store_true", help="Start HTTP server")
    p.add_argument("--port",   type=int, default=5001)
    p.add_argument("--stdin",  action="store_true",
                   help="Read JSON payload {sourceCode, mode, filename} from stdin")
    p.add_argument("--file",   help="Java file to process (CLI mode)")
    p.add_argument("--mode",   choices=list(VALID_MODES), default="both")
    args = p.parse_args()

    if args.serve:
        _start_http_server(args.port)

    elif args.stdin:
        # Node sends: {"sourceCode": "...", "mode": "...", "filename": "..."}
        import json as _json
        payload     = _json.loads(sys.stdin.read())
        source_code = payload.get("sourceCode", "")
        mode        = payload.get("mode", args.mode)
        filename    = payload.get("filename", "uploaded.java")
        out = run_pipeline(source_code, mode, filename)
        # Print a unique delimiter so Node can find the JSON even if
        # pipeline.py printed debug lines to stdout beforehand
        print("__PIPELINE_JSON__")
        print(_json.dumps(out, default=str))

    elif args.file:
        src = Path(args.file).read_text(encoding="utf-8")
        out = run_pipeline(src, args.mode, Path(args.file).name)
        print(json.dumps(out, indent=2, default=str))

    else:
        p.print_help()
        sys.exit(1)