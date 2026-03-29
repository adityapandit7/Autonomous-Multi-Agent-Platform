# prompt_engine/smell_detector.py
#
# Block 3a: Code Smell Detector
#
# Detects the 4 smells using metrics from the parser output.
# Thresholds come from DACOSKnowledgeBase — data-driven from real Java projects
# when the DACOS folder is provided, defaults otherwise.
#
# Smells detected:
#   Long Method           — loc > threshold
#   Long Parameter List   — param_count > threshold
#   Complex Conditional   — conditional_count > threshold
#   Multifaceted Abstraction — responsibility_count > threshold

import logging
from typing import List, Dict, Any
from knowledge_base.dacos_knowledge import DACOSKnowledgeBase

logger = logging.getLogger(__name__)


class SmellDetector:

    def __init__(self, dacos_folder=None):
        self.kb = DACOSKnowledgeBase(dacos_folder)

        # Flatten nested DACOS thresholds into a single dict for fast lookup.
        # DACOSKnowledgeBase stores: {"Long Method": {"threshold": 30, "severe": 50, "critical": 70}}
        nested = self.kb.thresholds

        lm = nested.get("Long Method",              {})
        lp = nested.get("Long Parameter List",      {})
        cc = nested.get("Complex Conditional",      {})
        ma = nested.get("Multifaceted Abstraction", {})

        self.t = {
            # Long Method
            "lm_medium":   lm.get("threshold", 30),
            "lm_high":     lm.get("severe",    50),
            "lm_critical": lm.get("critical",  70),
            # Long Parameter List
            "lp_medium":   lp.get("threshold",  5),
            "lp_high":     lp.get("severe",     8),
            "lp_critical": lp.get("critical",  12),
            # Complex Conditional
            "cc_medium":   cc.get("threshold",  5),
            "cc_high":     cc.get("severe",    10),
            "cc_critical": cc.get("critical",  15),
            # Multifaceted Abstraction
            "ma_medium":   ma.get("threshold",  2),
            "ma_high":     ma.get("severe",     3),
            "ma_critical": ma.get("critical",   5),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_smells(self, parsed_code: dict) -> List[Dict]:
        """
        Run all smell checks on every method in parsed_code.
        Returns a list of smell dicts sorted by severity (critical first).
        """
        smells = []
        for method in parsed_code.get("functions", []):
            smells.extend(self._check_long_method(method))
            smells.extend(self._check_long_parameter_list(method))
            smells.extend(self._check_complex_conditional(method))
            smells.extend(self._check_multifaceted_abstraction(method))

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        smells.sort(key=lambda s: severity_order.get(s["severity"], 99))
        return smells

    def generate_report(self, parsed_code: dict) -> str:
        """Human-readable smell report for saving to file."""
        smells  = self.detect_smells(parsed_code)
        methods = parsed_code.get("functions", [])
        classes = parsed_code.get("classes",  [])

        lines = [
            "=" * 60,
            "CODE SMELL ANALYSIS REPORT — Java",
            "=" * 60,
            f"Methods analysed : {len(methods)}",
            f"Classes analysed : {len(classes)}",
            f"Total LOC        : {parsed_code.get('total_loc', 0)}",
            f"Smells detected  : {len(smells)}",
            f"Parser used      : {parsed_code.get('parser_used', 'unknown')}",
            f"Thresholds from  : {'DACOS dataset' if self.kb.initialized else 'defaults'}",
            "",
        ]

        if not smells:
            lines.append("No significant code smells detected.")
        else:
            for smell in smells:
                icon = "🔴" if smell["severity"] == "critical" else \
                       "🟡" if smell["severity"] in ("high", "medium") else "🟢"
                lines += [
                    f"{icon} [{smell['severity'].upper()}] {smell['name']}",
                    f"   Method    : {smell.get('function', 'N/A')}",
                    f"   Reason    : {smell['reason']}",
                    f"   Refactor  : {smell['suggestion']}",
                    "",
                ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Smell checks ──────────────────────────────────────────────────────────

    def _check_long_method(self, method: dict) -> List[Dict]:
        loc  = method.get("loc", 0)
        name = method.get("name", "unknown")
        t    = self.t

        if   loc >= t["lm_critical"]: severity = "critical"
        elif loc >= t["lm_high"]:     severity = "high"
        elif loc >= t["lm_medium"]:   severity = "medium"
        else:                          return []

        return [{
            "name":       "Long Method",
            "function":   name,
            "severity":   severity,
            "metric":     "loc",
            "value":      loc,
            "threshold":  t["lm_medium"],
            "reason":     f"Method '{name}' has {loc} lines (threshold: {t['lm_medium']})",
            "suggestion": "Apply Extract Method — split into focused private helper methods.",
        }]

    def _check_long_parameter_list(self, method: dict) -> List[Dict]:
        params = method.get("param_count", 0)
        name   = method.get("name", "unknown")
        t      = self.t

        if   params >= t["lp_critical"]: severity = "critical"
        elif params >= t["lp_high"]:     severity = "high"
        elif params >= t["lp_medium"]:   severity = "medium"
        else:                             return []

        return [{
            "name":       "Long Parameter List",
            "function":   name,
            "severity":   severity,
            "metric":     "param_count",
            "value":      params,
            "threshold":  t["lp_medium"],
            "reason":     f"Method '{name}' has {params} parameters (threshold: {t['lp_medium']})",
            "suggestion": "Introduce a Parameter Object or use the Builder pattern.",
        }]

    def _check_complex_conditional(self, method: dict) -> List[Dict]:
        conds = method.get("conditional_count", 0)
        name  = method.get("name", "unknown")
        t     = self.t

        if   conds >= t["cc_critical"]: severity = "critical"
        elif conds >= t["cc_high"]:     severity = "high"
        elif conds >= t["cc_medium"]:   severity = "medium"
        else:                            return []

        return [{
            "name":       "Complex Conditional",
            "function":   name,
            "severity":   severity,
            "metric":     "conditional_count",
            "value":      conds,
            "threshold":  t["cc_medium"],
            "reason":     f"Method '{name}' has {conds} conditional branches (threshold: {t['cc_medium']})",
            "suggestion": "Apply Guard Clauses, Strategy pattern, or Map-based dispatch.",
        }]

    def _check_multifaceted_abstraction(self, method: dict) -> List[Dict]:
        # FIX: uses DACOS thresholds (self.t) instead of hardcoded values
        resp = method.get("responsibility_count", 1)
        name = method.get("name", "unknown")
        t    = self.t

        if   resp >= t["ma_critical"]: severity = "critical"
        elif resp >= t["ma_high"]:     severity = "high"
        elif resp >= t["ma_medium"]:   severity = "medium"
        else:                           return []

        return [{
            "name":       "Multifaceted Abstraction",
            "function":   name,
            "severity":   severity,
            "metric":     "responsibility_count",
            "value":      resp,
            "threshold":  t["ma_medium"],
            "reason":     f"Method '{name}' has {resp} distinct responsibilities (threshold: {t['ma_medium']})",
            "suggestion": "Apply Single Responsibility Principle — extract each responsibility into its own method.",
        }]
