import os as _os
#from unittest import result

#from conftest import repo_state


# Prevent transformers from contacting HuggingFace Hub during model load
_os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
_os.environ.setdefault('HF_HUB_OFFLINE', '1')

#!/usr/bin/env python3
"""
main.py — Entry point for the Autonomous Multi-Agent Platform.

Usage:
    python main.py --file SimpleTest.java --mode both
    python main.py --file OrderProcessor.java --mode refactor
    python main.py --file SimpleTest.java --mode document
"""

# ── Must be the very first lines — before ANY import ─────────────────────────
import os
import warnings

# Fix 1: Prevents safetensors background thread from calling HuggingFace API
# This kills the 50-line ConnectError / OSError traceback entirely
os.environ["TRANSFORMERS_OFFLINE"]        = "1"

# Fix 2: Suppresses tqdm progress bars from safetensors / HF Hub loading
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Fix 3: Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"]      = "false"

# Fix 4: Suppress all Python warnings
warnings.filterwarnings("ignore")

# Fix 5: Silence all third-party loggers — must be before any transformers import
import logging

_SILENT = [
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "transformers.safetensors_conversion",
    "peft",
    "torch",
    "safetensors",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "huggingface_hub.utils._headers",
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
]
for _lib in _SILENT:
    logging.getLogger(_lib).setLevel(logging.CRITICAL)

# Pipeline logger — WARNING only (our own messages)
logging.basicConfig(level=logging.WARNING, format="  ⚠  %(message)s")

# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ───────────────────────────────────────────────────────────────────────────
# Args
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Parse command-line arguments for the Autonomous Multi-Agent Platform.

    This CLI allows you to:
    - Run the pipeline on a Java file
    - Choose execution mode (refactor / document / both)
    - Enable dry-run mode (no LLM calls)
    - Query the processed code using RAG

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="""
Autonomous Multi-Agent Platform for Code Refactoring and Documentation.

Pipeline Flow:
  1. Parse Java code
  2. Detect code smells
  3. Generate refactoring + documentation prompts
  4. (Optional) Execute agents
  5. Evaluate outputs
  6. Enable RAG-based querying

Examples:
  python main.py --file SimpleTest.java --mode both
  python main.py --file OrderProcessor.java --mode refactor
  python main.py --file SimpleTest.java --mode both --dry-run
  python main.py --file SimpleTest.java --ask "What does this class do?"
"""
    )

    # ── Core execution ─────────────────────────────────────────────
    parser.add_argument(
        "--file", "-f",
        default="SimpleTest.java",
        help="Path to the Java file to process (default: SimpleTest.java)"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["refactor", "document", "both"],
        default="both",
        help="Execution mode: refactor code, generate documentation, or both"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable all LLM/model calls (runs full pipeline with mock outputs)"
    )

    # ── RAG / Query features ───────────────────────────────────────
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Ask a single question about the processed code (RAG)"
    )

    parser.add_argument(
        "--query",
        action="store_true",
        help="Enter interactive query mode after pipeline execution"
    )

    parser.add_argument(
        "--query-mode",
        choices=["answer", "explain", "smell"],
        default="answer",
        help="Query mode: answer (default), explain code, or analyze smells"
    )

    return parser.parse_args()




# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def load_source(path: Path) -> str:
    if not path.exists():
        print(f"\n❌ File not found: {path}")
        sys.exit(1)

    code = path.read_text(encoding="utf-8")
    if not code.strip():
        print(f"\n❌ File is empty: {path}")
        sys.exit(1)

    return code


def run_query(question, pipeline, query_mode):
    try:
        from core.ragquery import query_repo
    except ImportError:
        print("⚠ RAG not available")
        return

    repo_state = getattr(pipeline, "_last_repo_state", None)
    symbol_index = getattr(pipeline, "symbol_index", None)

    if not repo_state or not symbol_index:
        print("⚠ No repo state available for querying")
        return

    print(f"\nQ: {question}\n")

    response = query_repo(
        query=question,
        retriever=getattr(pipeline, "retriever", None),
        symbol_index=symbol_index,
        llm=getattr(pipeline, "doc_agent", None),
        mode=query_mode,
    )

    print(f"A: {response.answer}")
    print(f"\nSources: {response.sources}")


def interactive_loop(pipeline, query_mode):
    print("\nEnter questions (type 'exit' to quit)\n")

    while True:
        q = input("> ").strip()
        if not q or q.lower() in ["exit", "quit"]:
            break
        run_query(q, pipeline, query_mode)


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Autonomous Multi-Agent Platform")
    print("=" * 60)
    print(f"  File : {args.file}")
    print(f"  Mode : {args.mode}")

    source_path = ROOT / args.file
    source_code = load_source(source_path)

    print(f"  Lines: {len(source_code.splitlines())}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        from pipeline import Pipeline
        pipeline = Pipeline()
        
        result = pipeline.run(source_code, mode=args.mode, disable_llm=args.dry_run)
        repo_state = getattr(pipeline, "_last_repo_state", None)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
    print("\n--- DEBUG: REPOSTATE ---")
    print(type(repo_state))

    if repo_state:
        print(f"RepoState Version: {repo_state.version}")
        print(f"Functions: {len(repo_state.functions)}")
        print(f"Smells: {len(repo_state.smells)}")
    else:
        print("RepoState NOT FOUND")



    if not result.get("success"):
        print("\n❌ Pipeline completed with errors")
        return 1

    print("\n✅ Pipeline completed successfully\n")

    # ── RAG ───────────────────────────────────────────────────────────────
    if args.ask:
        run_query(args.ask, pipeline, args.query_mode)

    if args.query:
        interactive_loop(pipeline, args.query_mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
