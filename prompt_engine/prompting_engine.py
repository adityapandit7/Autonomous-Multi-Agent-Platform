# prompt_engine/prompting_engine.py
#
# Block 3: Prompting Engine
#
# Responsibilities:
#   1. Run the SmellDetector on parsed_code
#   2. Build a refactoring prompt that:
#        - Uses the template for the most severe smell
#        - Passes CORRECT metric values (loc for Long Method, param_count for
#          Long Param, etc.) — not the same value for everything
#        - Lists ALL detected smells in the prompt (not just the top one)
#   3. Build a documentation prompt
#   4. Generate a human-readable refactoring plan

import logging
from typing import Dict, List
from smell_detector import SmellDetector
from .templates import (
    get_refactor_template,
    REFACTOR_BASE_TEMPLATE,
    DOCUMENTATION_BASE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class PromptingEngine:

    def __init__(self, model_type: str = "codet5p-770m", dacos_folder=None):
        self.model_type = model_type
        self.detector   = SmellDetector(dacos_folder=dacos_folder)

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_prompts(self, raw_code: str, parsed_code: dict,
                         user_request: str = "both") -> Dict[str, str]:
        """
        Generate refactor and/or documentation prompts.

        Returns dict with keys:
          refactor_prompt      (if user_request in "refactor", "both")
          documentation_prompt (if user_request in "document", "both")
        """
        smells = self.detector.detect_smells(parsed_code)
        result = {}

        if user_request in ("refactor", "both"):
            result["refactor_prompt"] = self._build_refactor_prompt(
                raw_code, parsed_code, smells
            )
        if user_request in ("document", "both"):
            result["documentation_prompt"] = self._build_doc_prompt(
                raw_code, parsed_code
            )

        return result

    def generate_refactoring_plan(self, parsed_code: dict) -> str:
        """Human-readable refactoring plan saved to 3_refactoring_plan.txt."""
        smells  = self.detector.detect_smells(parsed_code)
        methods = parsed_code.get("functions", [])

        lines = [
            "JAVA REFACTORING PLAN",
            "=" * 50,
            f"Methods analysed : {len(methods)}",
            f"Smells detected  : {len(smells)}",
            "",
        ]

        if not smells:
            lines.append("No significant smells detected. Code quality is acceptable.")
        else:
            for i, smell in enumerate(smells, 1):
                lines += [
                    f"{i}. [{smell['severity'].upper()}] {smell['name']}",
                    f"   Method    : {smell.get('function', 'N/A')}",
                    f"   Measured  : {smell['metric']} = {smell['value']} "
                    f"(threshold: {smell['threshold']})",
                    f"   Action    : {smell['suggestion']}",
                    "",
                ]

        return "\n".join(lines)

    # ── Private builders ───────────────────────────────────────────────────

    def _build_refactor_prompt(self, raw_code: str, parsed_code: dict,
                                smells: List[dict]) -> str:
        """
        Build the refactoring prompt.

        Strategy:
        - Use the template matching the most severe (first) smell.
        - Pass the CORRECT metric value for that smell's template placeholder.
        - Append a secondary-smells line so the model knows all issues.
        """
        if not smells:
            return REFACTOR_BASE_TEMPLATE.format(code=raw_code)

        top   = smells[0]
        name  = top["name"]
        value = top["value"]
        threshold = top["threshold"]

        template = get_refactor_template(name)

        # Use only the method body of the most severe smell, not the entire file.
        # This keeps the prompt within the 512-token model limit so the code
        # actually fits and the model receives meaningful input.
        method_name = top.get("function", "")
        code_to_use = self._extract_method_code(raw_code, parsed_code, method_name) or raw_code

        # Build the correct keyword arguments for this specific template.
        fmt_kwargs = {
            "code":               code_to_use,
            "loc":                value if name == "Long Method"              else 0,
            "param_count":        value if name == "Long Parameter List"      else 0,
            "conditional_count":  value if name == "Complex Conditional"      else 0,
            "resp_count":         value if name == "Multifaceted Abstraction" else 0,
            "threshold":          threshold,
        }

        try:
            prompt = template.format(**fmt_kwargs)
        except KeyError:
            prompt = REFACTOR_BASE_TEMPLATE.format(code=raw_code)

        # Append secondary smells if there are more than one.
        # This gives the model the full picture without wasting tokens.
        if len(smells) > 1:
            other_names = [s["name"] for s in smells[1:]]
            unique_others = list(dict.fromkeys(other_names))  # preserve order, deduplicate
            prompt += f"\n\n// Additional smells also present: {', '.join(unique_others)}"

        return prompt

    def _extract_method_code(self, raw_code: str, parsed_code: dict,
                              method_name: str) -> str:
        """
        Extract just the source lines for a specific method.
        Keeps the prompt short so the model's 512-token window fits the code.
        Falls back to the full file if extraction fails.
        """
        if not method_name:
            return raw_code

        method = next(
            (f for f in parsed_code.get("functions", []) if f["name"] == method_name),
            None
        )
        if not method:
            return raw_code

        start = method.get("lineno", 1)
        loc   = method.get("loc",    len(raw_code.splitlines()))
        lines = raw_code.splitlines()
        # lineno is 1-based; extract start-1 through start-1+loc
        end   = min(start - 1 + loc, len(lines))
        extracted = "\n".join(lines[start - 1 : end])
        # Strip leading Javadoc/comment lines — the regex parser may point
        # to a comment line just before the actual method signature.
        extracted_lines = extracted.splitlines()
        code_start = 0
        for idx, ln in enumerate(extracted_lines):
            s = ln.strip()
            if s and not s.startswith("*") and not s.startswith("/*") and not s.startswith("//"):
                code_start = idx
                break
        extracted = "\n".join(extracted_lines[code_start:])
        return extracted if extracted.strip() else raw_code

    def _build_doc_prompt(self, raw_code: str, parsed_code: dict) -> str:
        """
        Build the documentation prompt.
        Includes a brief smell summary so the Doc Agent can mention
        quality issues in the generated Javadoc.
        The smell list is kept to one line — cheap in tokens, useful in output.
        """
        methods     = parsed_code.get("functions", [])
        method_list = ", ".join(m["name"] for m in methods) if methods else "none"

        # Detect smells for context — kept brief (one line only)
        smell_line = ""
        try:
            from smell_detector import SmellDetector
            smells = SmellDetector(dacos_folder=None).detect_smells(parsed_code)
            if smells:
                names = ", ".join(dict.fromkeys(s["name"] for s in smells))
                smell_line = f"Code quality issues: {names}.\n"
        except Exception:
            pass

        return (
            f"Generate Javadoc for the following Java code.\n"
            f"Methods: {method_list}\n"
            f"{smell_line}"
            f"\n{raw_code}\n\n"
            f"Return ONLY the Javadoc:"
        )
