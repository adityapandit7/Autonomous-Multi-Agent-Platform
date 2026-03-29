# parser/java_parser.py
#
# Block 2: Java Source Code Parser
#
# Primary:  javalang  — pure Python Java 8 parser (accurate AST-based metrics)
# Fallback: regex     — used if javalang is unavailable or file uses Java 9+ syntax
#
# HONEST LIMITATIONS:
#   - javalang supports Java 8 syntax only.
#   - Java 9+ features trigger regex fallback automatically.
#   - Regex param counting is approximate for deeply nested generics.
#   - Responsibility count is heuristic, not static analysis.

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger.warning("[JavaParser] javalang not installed — pip install javalang")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_strings(code):
    """Replace Java string literal contents with empty strings.
    Prevents keywords inside strings from being counted as code constructs.
    Example: "flagging for review"  ->  ""   so 'for' is not a loop hit.
    """
    return re.sub(r'"[^"\n]*"', '""', code)


def _extract_method_body(source_code, start_line):
    """Extract lines belonging to a method by brace-matching.

    start_line is 1-based.  Scanning starts there and continues until braces
    balance back to zero.  Lines before the first opening brace are included
    (they form the signature) but do not affect the brace counter.
    """
    lines = source_code.splitlines()
    brace_count = 0
    body = []
    started = False

    for idx in range(max(0, start_line - 1), len(lines)):
        line = lines[idx]
        body.append(line)
        opens  = line.count("{")
        closes = line.count("}")
        brace_count += opens - closes
        if opens > 0:
            started = True
        if started and brace_count <= 0:
            break

    return body


def _count_nesting_depth(lines):
    """Maximum brace nesting depth inside a method body."""
    depth = max_depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, depth)
    return max(0, max_depth)


def _count_conditionals(lines):
    """Count conditional branches.

    Rules:
    - Each 'if'  (incl. those in 'else if') counts as one branch.
    - Each standalone 'else' (not followed by 'if') counts as one branch.
    - Each 'case' in a switch counts as one branch.
    - 'switch' itself is NOT a branch.
    String literals are stripped first to avoid false matches.
    """
    code = _strip_strings("\n".join(lines))
    count = 0
    count += len(re.findall(r'\bif\b',                   code))
    count += len(re.findall(r'\bcase\b',                 code))
    count += len(re.findall(r'\belse\b(?!\s*if\b)',      code))
    return count


def _count_loops(lines):
    """Count loop keywords (for / while / do), ignoring string literals."""
    code = _strip_strings("\n".join(lines))
    return len(re.findall(r'\b(?:for|while|do)\b', code))


def _estimate_responsibilities(lines):
    """Heuristic: count distinct responsibility types present in a method.

    Types detected:
      io          — System.out / logger / log. calls
      persistence — file / stream / database / connection / repository
      iteration   — for / while / do loops
      decision    — if / switch conditionals
      creation    — new SomeClass(...)
      error       — throw / catch / Exception
    """
    code = _strip_strings("\n".join(lines)).lower()
    resp = set()

    if re.search(r'system\.out|logger\.|log\.', code):
        resp.add("io")
    if re.search(r'\bfile\b|\bstream\b|\bdatabase\b|\bconnection\b|\brepository\b', code):
        resp.add("persistence")
    if re.search(r'\bfor\b|\bwhile\b|\bdo\b', code):
        resp.add("iteration")
    if re.search(r'\bif\b|\bswitch\b', code):
        resp.add("decision")
    if re.search(r'\bnew\s+\w+\s*\(', code):
        resp.add("creation")
    if re.search(r'\bthrow\b|\bcatch\b|\bexception\b', code):
        resp.add("error")

    return max(1, len(resp))


def _count_params_safe(params_str):
    """Count parameters correctly even when generics contain commas.

    Example:  'Map<String, Integer> map, String name'  ->  2  (not 3)
    """
    if not params_str.strip():
        return 0
    cleaned = re.sub(r'<[^>]*>', '', params_str)
    return len([p for p in cleaned.split(",") if p.strip()])


# ─────────────────────────────────────────────────────────────────────────────
# JavaParser class
# ─────────────────────────────────────────────────────────────────────────────

class JavaParser:
    """
    Parses a Java source file and returns a structured dict consumed by the
    rest of the pipeline (Prompting Engine, Agents, Evaluators).

    Output keys
    -----------
    parse_success : bool
    parser_used   : "javalang" | "regex"
    language      : "java"
    total_loc     : int
    classes       : list[dict]
    functions     : list[dict]   (named 'functions' for pipeline compatibility)

    Per-method dict keys
    --------------------
    name, lineno, loc, param_count, return_type, modifiers,
    nesting_depth, conditional_count, loop_count,
    responsibility_count, parser_used
    """

    def parse(self, source_code):
        if JAVALANG_AVAILABLE:
            result = self._parse_with_javalang(source_code)
            if result["parse_success"]:
                return result
            logger.warning("[JavaParser] javalang failed — falling back to regex.")
        return self._parse_with_regex(source_code)

    # ── javalang (primary) ───────────────────────────────────────────────────

    def _parse_with_javalang(self, source_code):
        try:
            tree = javalang.parse.parse(source_code)
        except Exception as e:
            return {"parse_success": False, "error": str(e),
                    "functions": [], "classes": [], "total_loc": 0,
                    "language": "java", "parser_used": "javalang"}

        functions = []
        classes   = []
        lines     = source_code.splitlines()

        for _, node in tree:

            if isinstance(node, javalang.tree.ClassDeclaration):
                lineno = node.position.line if node.position else 1
                body   = _extract_method_body(source_code, lineno)
                method_count = sum(
                    1 for _, n in node
                    if isinstance(n, javalang.tree.MethodDeclaration)
                )
                classes.append({
                    "name":         node.name,
                    "lineno":       lineno,
                    "loc":          len(body),
                    "method_count": method_count,
                })

            elif isinstance(node, javalang.tree.MethodDeclaration):
                lineno = node.position.line if node.position else 1
                body   = _extract_method_body(source_code, lineno)
                params = len(node.parameters) if node.parameters else 0

                # Extract plain type name from javalang's return_type node.
                # BasicType (double, int, boolean...) and ReferenceType (String, List...)
                # both have a .name attribute. Use it directly — regex on repr is fragile.
                if node.return_type is None:
                    return_type = "void"
                elif hasattr(node.return_type, "name"):
                    return_type = node.return_type.name or "void"
                else:
                    # Last resort: strip javalang repr to get type name
                    rt_str = str(node.return_type)
                    m = re.search(r"name='(\w+)'", rt_str)
                    return_type = m.group(1) if m else rt_str

                functions.append({
                    "name":                 node.name,
                    "lineno":               lineno,
                    "loc":                  len(body),
                    "param_count":          params,
                    "return_type":          return_type,
                    "modifiers":            sorted(node.modifiers) if node.modifiers else [],
                    "nesting_depth":        _count_nesting_depth(body),
                    "conditional_count":    _count_conditionals(body),
                    "loop_count":           _count_loops(body),
                    "responsibility_count": _estimate_responsibilities(body),
                    "parser_used":          "javalang",
                })

        return {
            "parse_success": True,
            "parser_used":   "javalang",
            "language":      "java",
            "total_loc":     len(lines),
            "classes":       classes,
            "functions":     functions,
        }

    # ── regex fallback ───────────────────────────────────────────────────────

    def _parse_with_regex(self, source_code):
        """Regex fallback for Java 9+ or when javalang is not installed."""
        lines     = source_code.splitlines()
        functions = []
        classes   = []

        class_pattern = re.compile(
            r'(?:(?:public|private|protected|abstract|final)\s+)*class\s+(\w+)'
        )
        method_pattern = re.compile(
            r'(?:(?:public|private|protected|static|final|synchronized|abstract|native)\s+)*'
            r'([\w<>\[\]]+)\s+'            # return type
            r'(\w+)\s*'                     # method name
            r'\(([^)]*)\)'                  # parameter list (single-line)
            r'(?:\s*throws\s+[\w,\s]+)?'   # optional throws
            r'\s*\{'                         # opening brace
        )

        # Sliding window: collapse WINDOW lines into one string per starting line
        # so multi-line signatures are matched as a single pattern.
        WINDOW  = 8
        windows = [
            " ".join(l.strip() for l in lines[i:i + WINDOW])
            for i in range(len(lines))
        ]

        seen = set()

        for i, line in enumerate(lines, 1):

            # ── Class detection ───────────────────────────────────────────
            cm = class_pattern.search(line)
            if cm and "class" in line and "=" not in line:
                name = cm.group(1)
                body = _extract_method_body(source_code, i)
                method_count = len(re.findall(
                    r'(?:public|private|protected)\s+\w[\w<>\[\]]*\s+\w+\s*\(',
                    "\n".join(body)
                ))
                classes.append({
                    "name":         name,
                    "lineno":       i,
                    "loc":          len(body),
                    "method_count": method_count,
                })
                continue

            # ── Method detection ──────────────────────────────────────────
            # Match against the window (captures multi-line signatures).
            # Then verify the method name appears on line i or i+1 to ensure
            # we are not detecting a method that actually starts several lines
            # ahead in the window.
            window = windows[i - 1]
            mm = method_pattern.search(window)
            if not mm:
                continue

            return_type = mm.group(1)
            method_name = mm.group(2)
            params_str  = mm.group(3)

            # Anchor check: method name must appear on line i or the next line
            anchor = " ".join(l.strip() for l in lines[i - 1: min(i + 1, len(lines))])
            if method_name not in anchor:
                continue

            # Skip keywords and constructors
            skip = {"if", "for", "while", "switch", "catch", "else",
                    "return", "new", "class", "interface", "enum"}
            if method_name.lower() in skip:
                continue
            if return_type[0].isupper() and return_type == method_name:
                continue  # likely a constructor
            if method_name in seen:
                continue
            seen.add(method_name)

            body = _extract_method_body(source_code, i)

            functions.append({
                "name":                 method_name,
                "lineno":               i,
                "loc":                  len(body),
                "param_count":          _count_params_safe(params_str),
                "return_type":          return_type,
                "modifiers":            [],
                "nesting_depth":        _count_nesting_depth(body),
                "conditional_count":    _count_conditionals(body),
                "loop_count":           _count_loops(body),
                "responsibility_count": _estimate_responsibilities(body),
                "parser_used":          "regex",
            })

        return {
            "parse_success": True,
            "parser_used":   "regex",
            "language":      "java",
            "total_loc":     len(lines),
            "classes":       classes,
            "functions":     functions,
        }
