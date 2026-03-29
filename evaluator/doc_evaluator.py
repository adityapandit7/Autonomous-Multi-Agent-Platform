# evaluator/doc_evaluator.py
#
# Block 6b: Documentation Evaluator — Full Javadoc Standard
#
# Updated completeness checks (was 5, now 8):
#   OLD: has_overview, has_parameters, has_return_info, has_method_docs, has_javadoc_style
#   NEW: + has_param_tags, has_return_tag, has_description, no_html_noise
#
# The new checks distinguish between:
#   - has_parameters: any mention of parameters (word "parameter" in text)
#   - has_param_tags: actual @param Javadoc tags present
#   - has_return_info: return mentioned anywhere
#   - has_return_tag:  actual @return tag present
#
# This distinction matters because CodeT5+ passes has_parameters (template table)
# but fails has_param_tags (@param tags absent) — the score now reflects this correctly.

import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Coverage — % of methods documented
# ═══════════════════════════════════════════════════════════════════════════════
def compute_coverage(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    functions = parsed_code.get("functions", [])
    if not functions:
        return {"score": 1.0, "covered": 0, "total": 0,
                "covered_functions": [], "missing_functions": []}

    doc_lower = documentation.lower()
    covered, missing = [], []
    for func in functions:
        name = func.get("name", "")
        (covered if name.lower() in doc_lower else missing).append(name)

    score = len(covered) / len(functions)
    return {
        "score":             round(score, 3),
        "covered":           len(covered),
        "total":             len(functions),
        "covered_functions": covered,
        "missing_functions": missing,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Completeness — 8 checks covering full Javadoc standard
# ═══════════════════════════════════════════════════════════════════════════════
def compute_completeness(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    """
    8 checks for Javadoc completeness.

    Checks split into basic (always needed) and Javadoc-specific:
      Basic:
        has_overview       — has an overview/description section
        has_method_docs    — has per-method documentation (### headers)
        has_description    — has a meaningful description (>= 5 words in first sentence)
        no_html_noise      — no raw HTML tags (<p>, </p> etc.) in output

      Javadoc-specific:
        has_parameters     — parameters mentioned anywhere (word "param" or table)
        has_param_tags     — actual @param tags present (strict Javadoc format)
        has_return_info    — return value mentioned anywhere
        has_return_tag     — actual @return tag present (strict Javadoc format)
    """
    doc_lower = documentation.lower()

    # Basic checks
    has_overview    = any(kw in doc_lower for kw in
                          ["overview", "description", "this module", "this class", "##"])
    has_method_docs = "###" in documentation

    # Description quality — first non-heading non-table line with >= 5 words
    desc_lines = [l.strip() for l in documentation.splitlines()
                  if l.strip() and not l.startswith("#")
                  and not l.startswith("|") and not l.startswith(">")
                  and not l.startswith("**Signature") and not l.startswith("---")]
    has_description = any(len(l.split()) >= 5 for l in desc_lines[:10])

    # HTML noise check — raw <p> tags indicate model output formatting issue
    no_html_noise = not bool(re.search(r"<[a-zA-Z][^>]*>", documentation))

    # Parameter checks — two levels
    has_parameters = any(kw in doc_lower for kw in
                         ["@param", "param", "parameter", "parameters", "arguments",
                          "parameters |"])
    has_param_tags = "@param" in documentation   # strict: actual tag

    # Return checks — two levels
    has_return_info = any(kw in doc_lower for kw in
                          ["@return", "return", "returns", "output", "result"])
    has_return_tag  = "@return" in documentation  # strict: actual tag

    checks = {
        "has_overview":    has_overview,
        "has_method_docs": has_method_docs,
        "has_description": has_description,
        "no_html_noise":   no_html_noise,
        "has_parameters":  has_parameters,
        "has_param_tags":  has_param_tags,
        "has_return_info": has_return_info,
        "has_return_tag":  has_return_tag,
    }

    score = sum(checks.values()) / len(checks)
    return {
        "score":  round(score, 3),
        "checks": checks,
        "passed": sum(checks.values()),
        "total":  len(checks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BLEU-4
# ═══════════════════════════════════════════════════════════════════════════════
def compute_bleu(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if BLEU_AVAILABLE:
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            smoothing  = SmoothingFunction().method1
            return round(sentence_bleu([ref_tokens], hyp_tokens,
                                       smoothing_function=smoothing), 3)
        except Exception:
            pass
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words:
        return 0.0
    return round(len(ref_words & hyp_words) / len(ref_words), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L
# ═══════════════════════════════════════════════════════════════════════════════
def compute_rouge_l(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            result = scorer.score(reference, hypothesis)
            return round(result["rougeL"].fmeasure, 3)
        except Exception:
            pass
    a, b = reference.lower(), hypothesis.lower()
    if not a or not b:
        return 0.0
    common = sum(1 for c in b if c in a)
    return round(min((2 * common) / (len(a) + len(b)), 1.0), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Length score
# ═══════════════════════════════════════════════════════════════════════════════
def compute_length_score(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    word_count  = len(documentation.split())
    line_count  = len(documentation.splitlines())
    func_count  = len(parsed_code.get("functions", []))
    expected_min = max(20, func_count * 20)
    expected_max = max(200, func_count * 100)

    if expected_min <= word_count <= expected_max:
        score = 1.0
    elif word_count < expected_min:
        score = word_count / expected_min
    else:
        score = expected_max / word_count

    return {
        "score":        round(score, 3),
        "word_count":   word_count,
        "line_count":   line_count,
        "expected_min": expected_min,
        "expected_max": expected_max,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence score
# ═══════════════════════════════════════════════════════════════════════════════
def compute_doc_confidence(coverage: Dict, completeness: Dict,
                           length: Dict, bleu: float, rouge_l: float,
                           has_reference: bool) -> Dict[str, Any]:
    components = {
        "coverage":     coverage["score"],
        "completeness": completeness["score"],
        "length":       length["score"],
    }
    weights = {
        "coverage":     0.35,
        "completeness": 0.35,
        "length":       0.30,
    }

    if has_reference:
        components["bleu"]    = bleu
        components["rouge_l"] = rouge_l
        weights = {
            "coverage":     0.20,
            "completeness": 0.20,
            "length":       0.15,
            "bleu":         0.20,
            "rouge_l":      0.25,
        }

    score = max(0.0, min(1.0, sum(weights[k] * components[k] for k in components)))

    if score >= 0.85:   status = "✅ EXCELLENT"
    elif score >= 0.75: status = "✅ GOOD"
    elif score >= 0.65: status = "⚠ MODERATE"
    elif score >= 0.50: status = "⚠ MARGINAL"
    else:               status = "❌ POOR"

    return {
        "score":      round(score, 3),
        "status":     status,
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights":    weights,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation function
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_documentation(documentation: str, parsed_code: dict,
                            reference_doc: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of generated Java documentation.

    Args:
        documentation: The generated documentation string
        parsed_code:   The parsed AST structure (dict) or source code string
        reference_doc: Optional reference documentation for BLEU/ROUGE

    Returns:
        Full evaluation results dict
    """
    # Handle both dict and string input for parsed_code (pipeline compat)
    if isinstance(parsed_code, str):
        parsed_code = {"functions": [], "classes": []}

    has_reference = reference_doc is not None and len(reference_doc.strip()) > 0

    coverage     = compute_coverage(documentation, parsed_code)
    completeness = compute_completeness(documentation, parsed_code)
    length       = compute_length_score(documentation, parsed_code)
    bleu         = compute_bleu(reference_doc or "", documentation) if has_reference else 0.0
    rouge_l      = compute_rouge_l(reference_doc or "", documentation) if has_reference else 0.0
    confidence   = compute_doc_confidence(coverage, completeness, length,
                                          bleu, rouge_l, has_reference)

    return {
        "coverage":      coverage,
        "completeness":  completeness,
        "length":        length,
        "bleu":          bleu,
        "rouge_l":       rouge_l,
        "has_reference": has_reference,
        "confidence":    confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report saver
# ═══════════════════════════════════════════════════════════════════════════════
def save_doc_evaluation_report(analysis: Dict[str, Any], documentation: str,
                                output_file: str = "doc_evaluation_report.txt") -> None:
    """Save documentation evaluation to a readable report file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines += ["=" * 60, "JAVA DOCUMENTATION QUALITY EVALUATION REPORT",
              "=" * 60, f"Generated: {timestamp}", ""]

    cov = analysis["coverage"]
    lines += ["METHOD COVERAGE", "-" * 40,
              f"Score  : {cov['score']}/1.0",
              f"Covered: {cov['covered']}/{cov['total']} methods"]
    if cov.get("missing_functions"):
        lines.append(f"Missing: {', '.join(cov['missing_functions'])}")
    lines.append("")

    comp = analysis["completeness"]
    lines += ["COMPLETENESS", "-" * 40,
              f"Score : {comp['score']}/1.0  ({comp['passed']}/{comp['total']} checks passed)"]
    for check, passed in comp["checks"].items():
        lines.append(f"  {'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
    lines.append("")

    ln = analysis["length"]
    lines += ["LENGTH & READABILITY", "-" * 40,
              f"Score      : {ln['score']}/1.0",
              f"Word count : {ln['word_count']}  (expected {ln['expected_min']}–{ln['expected_max']})",
              f"Line count : {ln['line_count']}", ""]

    if analysis["has_reference"]:
        lines += ["BLEU & ROUGE (vs reference)", "-" * 40,
                  f"BLEU-4  : {analysis['bleu']}",
                  f"ROUGE-L : {analysis['rouge_l']}", ""]

    conf = analysis["confidence"]
    lines += ["OVERALL CONFIDENCE", "-" * 40,
              f"Score  : {conf['score']}/1.0",
              f"Status : {conf['status']}", "", "Component Scores:"]
    for name, score in conf["components"].items():
        weight = conf["weights"][name] * 100
        bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  {name.replace('_',' ').title():<15} {bar}  {score:.3f}  ({weight:.0f}%)")
    lines.append("")

    lines += ["=" * 60, "VERDICT", "=" * 60]
    score = conf["score"]
    if score >= 0.75:
        lines.append("✅ Documentation ACCEPTED — Good to excellent quality.")
    elif score >= 0.65:
        lines.append("⚠  Documentation CONDITIONALLY ACCEPTED — Review recommended.")
    else:
        lines.append("❌ Documentation REJECTED — Insufficient quality.")
    lines.append("")

    lines += ["=" * 60, "DOCUMENTATION PREVIEW (first 20 lines)", "=" * 60]
    for i, line in enumerate(documentation.splitlines()[:20], 1):
        lines.append(f"  {i:2d}: {line}")
    if len(documentation.splitlines()) > 20:
        lines.append(f"  ... ({len(documentation.splitlines())} lines total)")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.error(f"Could not save doc report: {e}")
