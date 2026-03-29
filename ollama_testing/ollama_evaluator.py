"""
ollama_evaluator.py — Evaluates Ollama outputs with same metrics as main pipeline

CodeBLEU notes:
  - Requires: pip install codebleu tree-sitter==0.21.3 tree-sitter-java
  - Reference used: original (smelly) code
  - Interpretation: lower score = model changed more from original (more refactoring)
  - This is consistent across all models on the same input file
"""
import re

try:
    import javalang
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

try:
    from codebleu import calc_codebleu
    HAS_CODEBLEU = True
except ImportError:
    HAS_CODEBLEU = False
    print("WARNING: codebleu not installed — CodeBLEU metric will be skipped.")
    print("  Fix: pip install codebleu tree-sitter==0.21.3 tree-sitter-java")


def evaluate_refactored_code(original_code, refactored_code, original_loc):
    # ── AST validity ──────────────────────────────────────────────────────
    ast_valid = False
    ast_error = None
    if HAS_JAVALANG:
        try:
            test = refactored_code
            if "class " not in test:
                test = f"public class Test {{\n{test}\n}}"
            javalang.parse.parse(test)
            ast_valid = True
        except Exception as e:
            ast_error = str(e)[:100]
    else:
        o = refactored_code.count("{")
        c = refactored_code.count("}")
        ast_valid = (o == c and o > 0)
        ast_error = None if ast_valid else f"Braces: {o} open, {c} close"

    # ── Style ─────────────────────────────────────────────────────────────
    style_checks = {
        "camel_case_methods":   bool(re.search(r"\b[a-z][a-zA-Z0-9]+\s*\(", refactored_code)),
        "brace_same_line":      ") {" in refactored_code,
        "spaces_not_tabs":      "\t" not in refactored_code,
        "has_semicolons":       ";" in refactored_code,
        "minimal_trailing_ws":  not bool(re.search(r"[ \t]+\n", refactored_code)),
        "no_integer_discounts": not bool(re.search(r"discount\s*[+\-]?=\s*[1-9][0-9]*;", refactored_code)),
    }
    style_score = round(sum(style_checks.values()) / len(style_checks), 3)

    # ── Logic correctness ─────────────────────────────────────────────────
    logic_checks = {
        "uses_equals_not_==":      '== "' not in refactored_code,
        "discount_values_decimal": not bool(re.search(r"discount\s*[+\-]?=\s*[1-9][0-9]*;", refactored_code)),
        "balanced_braces":         refactored_code.count("{") == refactored_code.count("}"),
        "has_return_statement":    "return" in refactored_code,
        "preserves_method_name":   any(m in refactored_code for m in
                                       ["applyDiscount","calculateDiscount","computeDiscount"]),
    }
    logic_score = round(sum(logic_checks.values()) / len(logic_checks), 3)

    # ── LOC ───────────────────────────────────────────────────────────────
    ref_loc    = len([l for l in refactored_code.split("\n") if l.strip()])
    loc_delta  = original_loc - ref_loc
    improvement= round(min(max(loc_delta / original_loc, 0), 1), 3) if loc_delta > 0 else 0.0

    # ── Extract Method ────────────────────────────────────────────────────
    method_count      = len(re.findall(r"\b(?:private|protected|public)\s+\w+\s+\w+\s*\(", refactored_code))
    extracted_methods = max(0, method_count - 1)

    # ── CodeBLEU ──────────────────────────────────────────────────────────
    # Reference = original smelly code. Lower score = model changed more from original.
    # This is consistent across all models on the same input file.
    codebleu_score      = None
    codebleu_components = {}
    codebleu_note       = "not computed"

    if HAS_CODEBLEU:
        try:
            cb = calc_codebleu(
                references=[[original_code]],
                predictions=[refactored_code],
                lang="java",
                weights=(0.25, 0.25, 0.25, 0.25),
            )
            codebleu_score      = round(cb["codebleu"], 4)
            codebleu_components = {
                k: round(v, 4) for k, v in cb.items() if k != "codebleu"
            }
            codebleu_note = "vs original (lower = more restructured)"
        except Exception as e:
            codebleu_note = f"failed: {str(e)[:120]}"

    # ── Confidence ────────────────────────────────────────────────────────
    confidence = round(
        0.25 * (1.0 if ast_valid else 0.0) +
        0.20 * style_score +
        0.30 * logic_score +
        0.15 * improvement +
        0.10 * min(extracted_methods / 2, 1.0),
        3
    )
    status = ("EXCELLENT" if confidence >= 0.8 else
              "GOOD"      if confidence >= 0.6 else
              "FAIR"      if confidence >= 0.4 else "POOR")

    return {
        "ast_validity":      {"valid": ast_valid, "method": "javalang" if HAS_JAVALANG else "brace_balance", "error": ast_error},
        "style_metrics":     {"score": style_score, "checks": style_checks, "passed": sum(style_checks.values()), "total": len(style_checks)},
        "logic_correctness": {"score": logic_score, "checks": logic_checks, "passed": sum(logic_checks.values()), "total": len(logic_checks)},
        "improvement":       {"loc_original": original_loc, "loc_refactored": ref_loc, "loc_delta": loc_delta, "score": improvement},
        "extract_method":    {"extracted_methods": extracted_methods, "applied": extracted_methods > 0},
        "codebleu":          {"score": codebleu_score, "components": codebleu_components, "note": codebleu_note},
        "confidence":        {"score": confidence, "status": status},
    }


def evaluate_documentation(javadoc, method_params=None):
    if method_params is None:
        method_params = ["price","customerType","couponCode","isSeasonal","quantity","isMember"]

    doc_lower = javadoc.lower()

    completeness_checks = {
        "has_description":  bool(re.search(r"/\*\*\s*\n\s*\*\s+\w", javadoc)),
        "has_return_tag":   "@return" in javadoc,
        "has_param_tags":   "@param" in javadoc,
        "has_all_params":   all(p.lower() in doc_lower for p in method_params),
        "proper_format":    javadoc.strip().startswith("/**") and javadoc.strip().endswith("*/"),
        "no_html_noise":    "<p>" not in javadoc and "</p>" not in javadoc,
    }
    comp_score = round(sum(completeness_checks.values()) / len(completeness_checks), 3)

    params_covered = [p for p in method_params if p.lower() in doc_lower]
    param_coverage = round(len(params_covered) / len(method_params), 3)
    param_tags     = re.findall(r"@param\s+(\w+)", javadoc)

    word_count = len(javadoc.split())
    line_count = len(javadoc.strip().split("\n"))
    length_ok  = 10 <= word_count <= 500

    confidence = round(
        0.35 * comp_score +
        0.35 * param_coverage +
        0.20 * (1.0 if completeness_checks["has_return_tag"] else 0.0) +
        0.10 * (1.0 if length_ok else 0.5),
        3
    )
    status = ("EXCELLENT" if confidence >= 0.8 else
              "GOOD"      if confidence >= 0.6 else
              "FAIR"      if confidence >= 0.4 else "POOR")

    return {
        "completeness":       {"score": comp_score, "checks": completeness_checks, "passed": sum(completeness_checks.values()), "total": len(completeness_checks)},
        "parameter_coverage": {"score": param_coverage, "covered": params_covered, "missing": [p for p in method_params if p.lower() not in doc_lower], "param_tags_found": param_tags},
        "length":             {"word_count": word_count, "line_count": line_count, "ok": length_ok},
        "confidence":         {"score": confidence, "status": status},
    }