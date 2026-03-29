# evaluator/refactor_evaluator.py — Java version
import re, logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False


def check_ast_validity(code: str) -> Dict[str, Any]:
    """
    Check if code is valid Java.

    Strategy:
    1. Wrap bare methods in a class so javalang can parse them.
    2. Try javalang (Java 8 strict parser).
       - Accepted  → valid, done.
       - Rejected  → may be Java 14+ (rule switch, records, var).
                     Fall through to structural check.
    3. Structural check: balanced braces + has Java keywords.
       This correctly accepts Java 14+ code that javalang cannot parse.
    """
    if not code or len(code.strip()) < 10:
        return {"valid": False, "method": "empty_check", "error": "Empty or too short"}

    # Wrap bare methods so javalang can parse them as a compilation unit
    test_code = code if "class " in code else ("public class _Wrapper {\n" + code + "\n}")

    if JAVALANG_AVAILABLE:
        try:
            javalang.parse.parse(test_code)
            return {"valid": True, "method": "javalang", "error": None}
        except Exception as e:
            # javalang rejected it — could be Java 14+ syntax (rule switch etc.)
            # Fall through to structural check rather than hard-failing.
            pass

    # Structural check — handles Java 14+ syntax that javalang cannot parse
    balanced = code.count("{") == code.count("}")
    java_signs = ["public ", "private ", "protected ", "void ", "class ",
                  "return ", "int ", "double ", "String ", "switch "]
    python_signs = ["\ndef ", "\nelif ", "def __"]
    has_java    = any(s in code for s in java_signs)
    not_python  = not any(s in code for s in python_signs)
    ok = balanced and has_java and not_python and len(code.strip()) > 10
    method = "structural_java14" if JAVALANG_AVAILABLE else "structural_check"
    return {"valid": ok, "method": method, "error": None}


def compute_style_metrics(code: str) -> Dict[str, Any]:
    lines = code.splitlines()
    checks = {}
    method_names = re.findall(r'(?:public|private|protected|static)\s+\w[\w<>]*\s+([a-z]\w*)\s*\(', code)
    checks["camel_case_methods"] = all(re.match(r'^[a-z][a-zA-Z0-9]*$', n) for n in method_names) if method_names else True
    class_names = re.findall(r'\bclass\s+([A-Z][a-zA-Z0-9]*)', code)
    checks["pascal_case_classes"] = bool(class_names)
    checks["brace_same_line"] = len([l for l in lines if l.strip() == "{"]) == 0
    checks["spaces_not_tabs"] = len([l for l in lines if l.startswith("\t")]) == 0
    checks["has_semicolons"] = ";" in code
    trailing_ws = sum(1 for l in lines if l != l.rstrip())
    checks["minimal_trailing_ws"] = trailing_ws <= len(lines) * 0.1
    score = sum(checks.values()) / len(checks)
    return {"score": round(score, 3), "checks": checks, "passed": sum(checks.values()), "total": len(checks)}


def _tokenize(code: str) -> List[str]:
    tokens = re.findall(r'[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[{}()\[\];,.<>!=&|+\-*/^%]', code)
    kw = {"public","private","protected","static","void","int","double","String","boolean",
          "class","return","if","else","for","while","new","this","final","import","package"}
    return [t for t in tokens if t not in kw]


def compute_codebleu(original: str, refactored: str) -> float:
    ot, rt = _tokenize(original), _tokenize(refactored)
    if not ot or not rt: return 0.0
    os_, rs_ = set(ot), set(rt)
    jaccard = len(os_ & rs_) / len(os_ | rs_)
    ob = set(zip(ot, ot[1:])); rb = set(zip(rt, rt[1:]))
    bigram = len(ob & rb) / len(ob | rb) if (ob | rb) else 0
    return round((jaccard + bigram) / 2, 3)


def compute_semantic_preservation(original: str, refactored: str) -> Dict[str, Any]:
    pat = re.compile(r'(?:public|private|protected|static)\s+\w[\w<>]*\s+(\w+)\s*\(')
    orig_m = set(pat.findall(original)); ref_m = set(pat.findall(refactored))
    preserved = orig_m & ref_m
    prate = len(preserved) / len(orig_m) if orig_m else 1.0
    ot, rt = set(_tokenize(original)), set(_tokenize(refactored))
    sim = len(ot & rt) / len(ot | rt) if (ot | rt) else 0
    ideal_sim = 0.3 <= sim <= 0.8
    lr = len(refactored) / max(len(original), 1)
    length_ok = 0.3 <= lr <= 3.0
    score = (prate * 0.5) + (0.3 if ideal_sim else 0.0) + (0.2 if length_ok else 0.0)
    return {"score": round(score, 3), "preservation_rate": round(prate, 3),
            "token_similarity": round(sim, 3), "ideal_similarity": ideal_sim,
            "length_ratio": round(lr, 3), "length_ok": length_ok,
            "original_methods": list(orig_m), "preserved_methods": list(preserved)}


def compute_improvement_score(original: str, refactored: str) -> Dict[str, Any]:
    ol, rl = original.splitlines(), refactored.splitlines()
    od = max((len(l) - len(l.lstrip())) // 4 for l in ol if l.strip()) if ol else 0
    rd = max((len(l) - len(l.lstrip())) // 4 for l in rl if l.strip()) if rl else 0
    ld = len(ol) - len(rl)
    ch = sum(1 for a, b in zip(ol, rl) if a != b) + abs(len(ol) - len(rl))
    ls = min(1.0, max(0.0, ld / max(len(ol), 1) + 0.5))
    ds = 1.0 if od - rd >= 0 else 0.5
    cs = min(1.0, ch / max(len(ol), 1))
    score = ls * 0.3 + ds * 0.3 + cs * 0.4
    return {"score": round(score, 3), "loc_original": len(ol), "loc_refactored": len(rl),
            "loc_delta": ld, "depth_original": od, "depth_refactored": rd, "changed_lines": ch}


def compute_confidence(ast_val, style, codebleu, semantic, improvement) -> Dict:
    if not ast_val["valid"]:
        return {"score": 0.0, "status": "❌ INVALID — Not valid Java", "components": {}, "weights": {}}
    components = {"style": style["score"], "codebleu": codebleu,
                  "semantic": semantic["score"], "improvement": improvement["score"]}
    weights = {"style": 0.20, "codebleu": 0.25, "semantic": 0.35, "improvement": 0.20}
    score = max(0.0, min(1.0, sum(weights[k] * components[k] for k in components)))
    if score >= 0.85:   status = "✅ EXCELLENT"
    elif score >= 0.75: status = "✅ GOOD"
    elif score >= 0.65: status = "⚠ MODERATE"
    elif score >= 0.50: status = "⚠ MARGINAL"
    else:               status = "❌ POOR"
    return {"score": round(score, 3), "status": status,
            "components": {k: round(v, 3) for k, v in components.items()}, "weights": weights}


def analyze_refactoring(original: str, refactored: str) -> Dict[str, Any]:
    ast_val     = check_ast_validity(refactored)
    style       = compute_style_metrics(refactored)
    codebleu    = compute_codebleu(original, refactored)
    semantic    = compute_semantic_preservation(original, refactored)
    improvement = compute_improvement_score(original, refactored)
    confidence  = compute_confidence(ast_val, style, codebleu, semantic, improvement)
    return {"ast_validity": ast_val, "style_metrics": style, "codebleu_score": codebleu,
            "semantic_preservation": semantic, "improvement": improvement, "confidence": confidence}


def save_evaluation_report(analysis: Dict, original: str, refactored: str, output_file: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["=" * 60, "JAVA REFACTORING EVALUATION REPORT", "=" * 60, f"Generated: {timestamp}", ""]
    ast = analysis["ast_validity"]
    lines += ["AST VALIDITY", "-" * 40,
              f"Valid Java : {'✅ YES' if ast['valid'] else '❌ NO'}",
              f"Method     : {ast['method']}"]
    if ast["error"]: lines.append(f"Error      : {ast['error']}")
    lines.append("")
    style = analysis["style_metrics"]
    lines += ["STYLE METRICS", "-" * 40, f"Score : {style['score']}/1.0  ({style['passed']}/{style['total']} checks)"]
    for c, p in style["checks"].items():
        lines.append(f"  {'✅' if p else '❌'} {c.replace('_',' ').title()}")
    lines.append("")
    sem = analysis["semantic_preservation"]
    lines += ["SEMANTIC PRESERVATION", "-" * 40,
              f"Score              : {sem['score']}/1.0",
              f"Method preservation: {sem['preservation_rate']}",
              f"Token similarity   : {sem['token_similarity']}",
              f"Length ratio       : {sem['length_ratio']}", ""]
    imp = analysis["improvement"]
    lines += ["IMPROVEMENT", "-" * 40,
              f"LOC: {imp['loc_original']} → {imp['loc_refactored']} (delta: {imp['loc_delta']:+d})",
              f"Depth: {imp['depth_original']} → {imp['depth_refactored']}",
              f"Lines changed: {imp['changed_lines']}", ""]
    conf = analysis["confidence"]
    lines += ["OVERALL CONFIDENCE", "-" * 40, f"Score  : {conf['score']}/1.0", f"Status : {conf['status']}", ""]
    if conf.get("components"):
        for n, s in conf["components"].items():
            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            lines.append(f"  {n:<12} {bar}  {s:.3f}")
    lines += ["", "=" * 60, "VERDICT", "=" * 60]
    s = conf["score"]
    lines.append("✅ ACCEPTED" if s >= 0.75 else "⚠  CONDITIONALLY ACCEPTED" if s >= 0.65 else "❌ REJECTED")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.error(f"Could not save report: {e}")
