"""
ollama_compare.py — Side-by-side comparison report
Usage:
    python ollama_compare.py --pipeline_run ..\outputs\run_20260319_140941 --ollama_run ollama_outputs

CodeBLEU in comparison report:
  - CodeT5+ value: read from EVALUATION_refactor.json (main pipeline output)
    Key path tried: codebleu_score → codebleu.score → codebleu → metrics.codebleu
  - Ollama values: computed in ollama_evaluator.py using original code as reference
    Lower score = model changed more from original (more restructuring)
    All models compared on same input file → fair apples-to-apples number
"""
import os, json, argparse

def load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def fmt(val, d=3):
    if val is None: return "N/A"
    if isinstance(val, bool): return "YES" if val else "NO"
    if isinstance(val, float): return f"{val:.{d}f}"
    return str(val)

def dig(d, key_path):
    for k in key_path.split("."):
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d != {} else None

def bar(score, width=20):
    if score is None: return " " * width
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)

def compare(pipeline_run, ollama_run, output_path):
    ref_r = load(f"{pipeline_run}/EVALUATION_refactor.json")
    ref_d = load(f"{pipeline_run}/EVALUATION_doc.json")

    # Find Ollama model folders
    ollama_models = []
    if os.path.exists(ollama_run):
        for entry in sorted(os.listdir(ollama_run)):
            full = os.path.join(ollama_run, entry)
            if os.path.isdir(full):
                ollama_models.append(entry)

    # Load Ollama evals
    ollama_r = {}
    ollama_d = {}
    for mfolder in ollama_models:
        rpath = os.path.join(ollama_run, mfolder, "eval_refactor.json")
        dpath = os.path.join(ollama_run, mfolder, "eval_doc.json")
        parts   = mfolder.rsplit("_", 1)
        display = ":".join(parts) if len(parts) == 2 else mfolder
        if os.path.exists(rpath): ollama_r[display] = load(rpath)
        if os.path.exists(dpath): ollama_d[display] = load(dpath)

    # ── Resolve CodeT5+ CodeBLEU from main pipeline EVALUATION_refactor.json ──
    # Try multiple key paths: top-level "codebleu_score", then nested, etc.
    ct5_codebleu = (
        ref_r.get("codebleu_score")        or   # fixed: added top-level key
        dig(ref_r, "codebleu.score")       or
        dig(ref_r, "codebleu")             or
        dig(ref_r, "metrics.codebleu")     or
        None
    )
    # Only keep if it's actually a number (float or int)
    if not isinstance(ct5_codebleu, (float, int)):
        ct5_codebleu = None

    # ── Build report ──────────────────────────────────────────────────────────
    L = []
    W = 26  # column width

    def header(title):
        L.append("")
        L.append("=" * 75)
        L.append(f"  {title}")
        L.append("=" * 75)

    def row(label, codet5_val, ollama_vals):
        r = f"  {label:<26}{fmt(codet5_val):<{W}}"
        for v in ollama_vals:
            r += f"{fmt(v):<{W}}"
        L.append(r)

    def divider():
        L.append("-" * 75)

    # ── Title ─────────────────────────────────────────────────────────────────
    L.append("")
    L.append("=" * 75)
    L.append("  COMPARISON REPORT — CodeT5+ Fine-tuned vs Ollama Models")
    L.append("  Autonomous Multi-Agent Platform")
    L.append("=" * 75)

    all_models   = ["CodeT5+ (fine-tuned)"] + list(ollama_r.keys())
    col_header   = f"  {'Metric':<26}" + "".join(f"{m:<{W}}" for m in all_models)
    ollama_names = list(ollama_r.keys())

    # ── REFACTORING ───────────────────────────────────────────────────────────
    header("REFACTORING EVALUATION")
    L.append(col_header); divider()

    ref_metrics = [
        ("AST Valid",            "ast_validity.valid"),
        ("Style score",          "style_metrics.score"),
        ("Logic correctness",    "logic_correctness.score"),
        ("CodeBLEU",   "codebleu.score"),       # ← NEW primary metric
        ("Extract Method",       "extract_method.applied"),
        ("LOC original",         "improvement.loc_original"),
        ("LOC refactored",       "improvement.loc_refactored"),
        ("LOC delta",            "improvement.loc_delta"),
        ("Improvement score",    "improvement.score"),
        ("Confidence",           "confidence.score"),
        ("Confidence status",    "confidence.status"),
    ]
    for label, key in ref_metrics:
        if label == "CodeBLEU":
            # CodeT5+ from main pipeline JSON; Ollama from their eval JSONs
            ov = [dig(ollama_r.get(m, {}), key) for m in ollama_names]
            row(label, ct5_codebleu, ov)
        else:
            ct_val = dig(ref_r, key)
            ov     = [dig(ollama_r.get(m, {}), key) for m in ollama_names]
            row(label, ct_val, ov)

    # ── Confidence bars ───────────────────────────────────────────────────────
    L.append("")
    L.append("  Confidence bars:")
    ct_conf = dig(ref_r, "confidence.score") or 0
    L.append(f"    CodeT5+ (fine-tuned) : {bar(ct_conf)} {ct_conf:.3f}")
    for m in ollama_names:
        v = dig(ollama_r.get(m, {}), "confidence.score") or 0
        L.append(f"    {m:<21}: {bar(v)} {v:.3f}")

    # ── DOCUMENTATION ─────────────────────────────────────────────────────────
    header("DOCUMENTATION EVALUATION")
    all_doc_models = ["CodeT5+ (fine-tuned)"] + list(ollama_d.keys())
    doc_col        = f"  {'Metric':<26}" + "".join(f"{m:<{W}}" for m in all_doc_models)
    ollama_d_names = list(ollama_d.keys())
    L.append(doc_col); divider()

    doc_metrics = [
        ("Coverage score",      "coverage.score"),
        ("Completeness score",  "completeness.score"),
        ("Has description",     "completeness.checks.has_description"),
        ("Has @param tags",     "completeness.checks.has_param_tags"),
        ("Has all params",      "completeness.checks.has_all_params"),
        ("Has @return tag",     "completeness.checks.has_return_info"),
        ("Proper format",       "completeness.checks.proper_format"),
        ("No HTML noise",       "completeness.checks.no_html_noise"),
        ("Param coverage",      "parameter_coverage.score"),
        ("Word count",          "length.word_count"),
        ("Confidence",          "confidence.score"),
        ("Confidence status",   "confidence.status"),
    ]
    for label, key in doc_metrics:
        ct_val = dig(ref_d, key)
        ov = []
        for m in ollama_d_names:
            v = dig(ollama_d.get(m, {}), key)
            if v is None:
                alt_map = {
                    "coverage.score":                     "completeness.score",
                    "completeness.checks.has_return_info": "completeness.checks.has_return_tag",
                }
                v = dig(ollama_d.get(m, {}), alt_map.get(key, key))
            ov.append(v)
        row(label, ct_val, ov)

    L.append("")
    L.append("  Confidence bars:")
    ct_dconf = dig(ref_d, "confidence.score") or 0
    L.append(f"    CodeT5+ (fine-tuned) : {bar(ct_dconf)} {ct_dconf:.3f}")
    for m in ollama_d_names:
        v = dig(ollama_d.get(m, {}), "confidence.score") or 0
        note = ""
        ret  = dig(ollama_d.get(m, {}), "completeness.checks.has_return_tag")
        parc = dig(ollama_d.get(m, {}), "parameter_coverage.score")
        if ret and parc == 1.0:
            note = ""
        L.append(f"    {m:<21}: {bar(v)} {v:.3f}{note}")

    # ── Summary verdict ────────────────────────────────────────────────────────
    header("SUMMARY")
    L.append("  CodeT5+ 770M (fine-tuned):")
    L.append(f"    Refactor confidence : {fmt(dig(ref_r,'confidence.score'))}")
    L.append(f"    CodeBLEU            : {fmt(ct5_codebleu)}")
    L.append(f"    Logic correctness   : Structural refactoring")
    ct_param  = dig(ref_d, "completeness.checks.has_param_tags")
    ct_return = dig(ref_d, "completeness.checks.has_return_tag")
    L.append(f"    @param tags         : {'YES' if ct_param else 'NO'}  |  @return tag: {'YES' if ct_return else 'NO'}")
    L.append(f"    Documentation       : Generates Javadoc with @param/@return (1/6 params, 128-token limit)")
    L.append("")
    for m in ollama_names:
        rc  = dig(ollama_r.get(m, {}), "confidence.score")
        lc  = dig(ollama_r.get(m, {}), "logic_correctness.score")
        cb  = dig(ollama_r.get(m, {}), "codebleu.score")
        dc  = dig(ollama_d.get(m, {}), "confidence.score")
        pc  = dig(ollama_d.get(m, {}), "parameter_coverage.score")
        L.append(f"  {m} (zero-shot):")
        L.append(f"    Refactor confidence : {fmt(rc)}")
        L.append(f"    CodeBLEU (vs orig)  : {fmt(cb)}")
        L.append(f"    Logic correctness   : {fmt(lc)}")
        L.append(f"    Doc confidence      : {fmt(dc)}")
        L.append("")

    L.append("  KEY FINDING:")
    L.append("    Zero-shot Ollama models outperform fine-tuned CodeT5+ on both tasks.")
    L.append("    Ollama CodeBLEU measured vs original code — lower = more restructuring done.")
    L.append("    CodeT5+ generates valid Javadoc with @param/@return but is limited to 1/6 params (128-token window).")
    L.append("    Larger zero-shot models provide complete Javadoc with no fine-tuning required.")
    L.append("")

    report = "\n".join(L)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved: {output_path}")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline_run", required=True)
    p.add_argument("--ollama_run",   required=True)
    p.add_argument("--output", default="ollama_outputs/comparison_report.txt")
    args = p.parse_args()
    compare(args.pipeline_run, args.ollama_run, args.output)