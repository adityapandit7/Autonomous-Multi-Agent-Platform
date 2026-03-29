# prompt_engine/templates.py
#
# Prompt templates for the Refactor Agent and Doc Agent.
#
# Design principle:
#   CodeT5+ 770M is a seq2seq model with a 512-token input window.
#   It was fine-tuned on RCCT Java (code-to-code pairs).
#   Prompts must be CONCISE — the Java code itself must fit in the window.
#   Long "Requirements:" lists waste tokens and push code out of context.
#
# Each smell template includes:
#   - One-line task description
#   - The smell name and measured value
#   - The code
#   - A one-line strategy hint
#   - "Refactored Java code:" as the output trigger

# ─── Refactoring templates ────────────────────────────────────────────────────

REFACTOR_BASE_TEMPLATE = """\
Refactor the following Java code to improve quality.

{code}

Return ONLY the refactored Java code:"""


LONG_METHOD_TEMPLATE = """\
Refactor this Java Long Method ({loc} lines, threshold {threshold}) using Extract Method.

{code}

Strategy: Extract cohesive groups into private helper methods.
Return ONLY the refactored Java code:"""


LONG_PARAM_TEMPLATE = """\
Refactor this Java Long Parameter List ({param_count} params, threshold {threshold}) \
using a Parameter Object or Builder.

{code}

Strategy: Group related parameters into a new class.
Return ONLY the refactored Java code:"""


COMPLEX_CONDITIONAL_TEMPLATE = """\
Refactor these Java Complex Conditionals ({conditional_count} branches, threshold {threshold}) \
using Guard Clauses or a Map.

{code}

Strategy: Replace if-else chains with early returns or Map-based dispatch.
Return ONLY the refactored Java code:"""


MULTIFACETED_TEMPLATE = """\
Refactor this Java method ({resp_count} responsibilities, threshold {threshold}) \
applying Single Responsibility Principle.

{code}

Strategy: Extract each responsibility into a focused private method.
Return ONLY the refactored Java code:"""


# ─── Documentation template ───────────────────────────────────────────────────

DOCUMENTATION_BASE_TEMPLATE = """\
Generate Javadoc for the following Java code.

{code}

Return ONLY the Javadoc documentation:"""


# ─── Template lookup ─────────────────────────────────────────────────────────

_SMELL_TEMPLATE_MAP = {
    "Long Method":              LONG_METHOD_TEMPLATE,
    "Long Parameter List":      LONG_PARAM_TEMPLATE,
    "Complex Conditional":      COMPLEX_CONDITIONAL_TEMPLATE,
    "Multifaceted Abstraction": MULTIFACETED_TEMPLATE,
}


def get_refactor_template(smell_name: str) -> str:
    """Return the template for a given smell name, or the base template."""
    return _SMELL_TEMPLATE_MAP.get(smell_name, REFACTOR_BASE_TEMPLATE)
