"""
run_ollama_test.py — Main entry point for Ollama testing

Usage:
    python run_ollama_test.py --pipeline_run ..\outputs\run_20260319_140941
    python run_ollama_test.py --pipeline_run ..\outputs\run_20260319_140941 --models codellama:7b gemma3:12b
"""
import os, sys, json, argparse, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ollama_client    import is_ollama_running, list_pulled_models, MODELS
from ollama_refactor  import run_refactor_test
from ollama_doc       import run_doc_test
from ollama_evaluator import evaluate_refactored_code, evaluate_documentation


def run_full_test(refactor_prompt, doc_prompt, original_code,
                  original_loc, models, output_dir):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*62)
    print("  Ollama Testing — Autonomous Multi-Agent Platform")
    print("="*62)
    print(f"  Models : {', '.join(models)}")
    print(f"  Output : {output_dir}")
    print(f"  Time   : {timestamp}")
    print("="*62)

    if not is_ollama_running():
        print("\nERROR: Ollama is not running.")
        print("  Start it:  ollama serve")
        sys.exit(1)

    pulled    = list_pulled_models()
    available = []
    for m in models:
        short = m.split(":")[0]
        if any(short in p for p in pulled):
            available.append(m)
        else:
            print(f"  SKIP {m} — not pulled. Run: ollama pull {m}")

    if not available:
        print("No models available.")
        sys.exit(1)

    print(f"\nTesting {len(available)} model(s): {', '.join(available)}")

    # ── Refactoring ────────────────────────────────────────────────────────
    print("\n" + "─"*62)
    print("  [1/4] REFACTORING")
    print("─"*62)
    refactor_results = run_refactor_test(refactor_prompt, available, output_dir)

    # ── Evaluate refactoring ───────────────────────────────────────────────
    print("\n  [2/4] EVALUATING REFACTORING")
    refactor_evals = {}
    for model, res in refactor_results.items():
        safe = model.replace(":", "_").replace("/","_").replace(".","_")
        mdir = os.path.join(output_dir, safe)
        if res.get("success") and res.get("output_cleaned"):
            ev = evaluate_refactored_code(original_code, res["output_cleaned"], original_loc)
            refactor_evals[model] = ev
            with open(f"{mdir}/eval_refactor.json", "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2, ensure_ascii=False)
            c = ev["confidence"]["score"]
            l = ev["logic_correctness"]["score"]
            a = ev["ast_validity"]["valid"]
            x = ev["extract_method"]["extracted_methods"]
            print(f"  {model:<30}  conf={c:.3f}  logic={l:.3f}  ast={a}  extracted={x}")
        else:
            print(f"  {model:<30}  FAILED")

    # ── Documentation ─────────────────────────────────────────────────────
    print("\n" + "─"*62)
    print("  [3/4] DOCUMENTATION")
    print("─"*62)
    doc_results = run_doc_test(doc_prompt, available, output_dir)

    # ── Evaluate documentation ─────────────────────────────────────────────
    print("\n  [4/4] EVALUATING DOCUMENTATION")
    doc_evals = {}
    for model, res in doc_results.items():
        safe = model.replace(":", "_").replace("/","_").replace(".","_")
        mdir = os.path.join(output_dir, safe)
        if res.get("success") and res.get("output_cleaned"):
            ev = evaluate_documentation(res["output_cleaned"])
            doc_evals[model] = ev
            with open(f"{mdir}/eval_doc.json", "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2, ensure_ascii=False)
            c  = ev["confidence"]["score"]
            pc = ev["parameter_coverage"]["score"]
            rt = ev["completeness"]["checks"].get("has_return_tag", False)
            print(f"  {model:<30}  conf={c:.3f}  params={pc:.3f}  @return={rt}")
        else:
            print(f"  {model:<30}  FAILED")

    # ── Summary JSON ───────────────────────────────────────────────────────
    summary = {
        "timestamp":     timestamp,
        "models_tested": available,
        "refactoring":   {m: {"confidence": v["confidence"]["score"],
                               "logic":      v["logic_correctness"]["score"],
                               "ast_valid":  v["ast_validity"]["valid"],
                               "extracted":  v["extract_method"]["extracted_methods"],
                               "loc_original":   v["improvement"]["loc_original"],
                               "loc_refactored": v["improvement"]["loc_refactored"]}
                          for m, v in refactor_evals.items()},
        "documentation": {m: {"confidence":    v["confidence"]["score"],
                               "param_coverage":v["parameter_coverage"]["score"],
                               "has_return":    v["completeness"]["checks"].get("has_return_tag"),
                               "completeness":  v["completeness"]["score"]}
                          for m, v in doc_evals.items()},
    }
    spath = os.path.join(output_dir, f"ollama_summary_{timestamp}.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Final table ────────────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  FINAL RESULTS")
    print("="*62)
    print(f"  {'Model':<30} {'Ref Conf':>9} {'Logic':>7} {'Doc Conf':>9} {'Params':>7}")
    print("  " + "-"*58)
    for model in available:
        rc = refactor_evals.get(model, {}).get("confidence", {}).get("score", 0)
        rl = refactor_evals.get(model, {}).get("logic_correctness", {}).get("score", 0)
        dc = doc_evals.get(model, {}).get("confidence", {}).get("score", 0)
        dp = doc_evals.get(model, {}).get("parameter_coverage", {}).get("score", 0)
        print(f"  {model:<30} {rc:>9.3f} {rl:>7.3f} {dc:>9.3f} {dp:>7.3f}")

    print(f"\n  Output structure:")
    if os.path.exists(output_dir):
        for entry in sorted(os.listdir(output_dir)):
            full = os.path.join(output_dir, entry)
            if os.path.isdir(full):
                files = os.listdir(full)
                print(f"    {output_dir}/{entry}/")
                for fn in sorted(files):
                    print(f"      {fn}")
    print(f"\n  Summary: {spath}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline_run", required=True,
                   help="Path to outputs/run_TIMESTAMP from main pipeline")
    p.add_argument("--models", nargs="+",
                   default=["codellama:7b","deepseek-coder:6.7b","gemma3:12b","qwen2.5vl:7b"])
    p.add_argument("--output_dir", default="ollama_outputs")
    args = p.parse_args()

    run_dir = args.pipeline_run
    for fn in ["PROMPT_refactor_agent.txt", "PROMPT_doc_agent.txt"]:
        if not os.path.exists(f"{run_dir}/{fn}"):
            print(f"ERROR: {run_dir}/{fn} not found")
            sys.exit(1)

    with open(f"{run_dir}/PROMPT_refactor_agent.txt", encoding="utf-8") as f:
        refactor_prompt = f.read()
    with open(f"{run_dir}/PROMPT_doc_agent.txt", encoding="utf-8") as f:
        doc_prompt = f.read()

    original_loc = 40
    parsed_path  = f"{run_dir}/parsed_analysis.json"
    if os.path.exists(parsed_path):
        original_loc = json.load(open(parsed_path, encoding="utf-8")).get("total_loc", 40)

    # Extract original code from refactor prompt
    lines = refactor_prompt.split("\n")
    code_lines, in_code = [], False
    for line in lines:
        if line.strip().startswith("public") and not in_code: in_code = True
        if in_code:
            if line.strip().startswith("Strategy:"): break
            code_lines.append(line)
    original_code = "\n".join(code_lines)

    run_full_test(refactor_prompt, doc_prompt, original_code,
                  original_loc, args.models, args.output_dir)
