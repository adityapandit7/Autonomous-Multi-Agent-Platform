# prompt_engine/dacos_knowledge.py
#
# DACOS Knowledge Base — data-driven code smell thresholds.
# When DACOS folder is provided, thresholds are calculated from real Java project data.
# When not available, sensible defaults are used that match smell_detector.py.

from .dacos_integration import get_dacos, init_dacos, DACOSDataset
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# ─── Default thresholds (aligned with smell_detector.py fallbacks) ───────────
# These are used when DACOS folder is not provided or fails to load.
# Values match real Java project analysis conventions.
DEFAULT_THRESHOLDS = {
    "Long Method":           {"threshold": 30, "severe": 50, "critical": 70},
    "Long Parameter List":   {"threshold":  5, "severe":  8, "critical": 12},
    "Complex Conditional":   {"threshold":  5, "severe": 10, "critical": 15},
    "Multifaceted Abstraction": {"threshold": 2, "severe": 3, "critical": 5},
}


class DACOSKnowledgeBase:
    """
    DACOS-powered knowledge base for code smell thresholds.
    Loads real thresholds from the DACOS dataset when available,
    falls back to defaults when not.
    """

    def __init__(self, dacos_folder: Optional[str] = None):
        self.dacos: Optional[DACOSDataset] = None
        self.dacos_folder = dacos_folder
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self.initialized = False

        if dacos_folder and dacos_folder != "SKIP":
            self._initialize_dacos()

        self.smells = self._build_smells()

        if self.initialized:
            logger.info("DACOS Knowledge Base initialized with real thresholds")
        else:
            logger.info("DACOS Knowledge Base initialized with default thresholds")

    def _initialize_dacos(self) -> bool:
        try:
            self.dacos = init_dacos(self.dacos_folder)
            if self.dacos:
                dacos_thresholds = self.dacos.get_smell_thresholds()
                for smell, values in dacos_thresholds.items():
                    if smell in self.thresholds:
                        self.thresholds[smell].update(values)
                    else:
                        self.thresholds[smell] = values
                self.initialized = True
                return True
        except Exception as e:
            logger.warning(f"Failed to initialize DACOS: {e}")
        return False

    def _build_smells(self) -> Dict:
        """
        Build smell definitions using thresholds from DACOS (or defaults).

        IMPORTANT — metric mapping:
          Long Method           → loc              (lines of code)
          Long Parameter List   → param_count      (number of parameters)
          Complex Conditional   → conditional_count (number of if/else/switch branches)
          Multifaceted Abstraction → responsibility_count (number of distinct responsibilities)
        """
        lm = self.thresholds.get("Long Method", {})
        lm_th, lm_sv, lm_cr = lm.get("threshold", 30), lm.get("severe", 50), lm.get("critical", 70)

        lp = self.thresholds.get("Long Parameter List", {})
        lp_th, lp_sv, lp_cr = lp.get("threshold", 5), lp.get("severe", 8), lp.get("critical", 12)

        cc = self.thresholds.get("Complex Conditional", {})
        cc_th, cc_sv, cc_cr = cc.get("threshold", 5), cc.get("severe", 10), cc.get("critical", 15)

        ma = self.thresholds.get("Multifaceted Abstraction", {})
        ma_th, ma_sv, ma_cr = ma.get("threshold", 2), ma.get("severe", 3), ma.get("critical", 5)

        return {
            "Long Method": {
                "name":        "Long Method",
                "description": f"Method exceeds {lm_th} lines — likely doing too many things.",
                "refactor_guidance": (
                    f"Apply Extract Method pattern. Split into focused private helper methods. "
                    f"Target: under {lm_th} lines per method."
                ),
                "metric": "loc",
                "condition": lambda f: f.get("loc", 0) > lm_th,
                "severity_levels": {
                    "critical": lambda f: f.get("loc", 0) > lm_cr,
                    "high":     lambda f: f.get("loc", 0) > lm_sv,
                    "medium":   lambda f: f.get("loc", 0) > lm_th,
                },
                "thresholds": {"threshold": lm_th, "severe": lm_sv, "critical": lm_cr},
            },

            "Long Parameter List": {
                "name":        "Long Parameter List",
                "description": f"Method has more than {lp_th} parameters — hard to read and test.",
                "refactor_guidance": (
                    f"Introduce a Parameter Object (new class grouping related params) "
                    f"or use the Builder pattern. Target: under {lp_th} parameters."
                ),
                "metric": "param_count",
                "condition": lambda f: f.get("param_count", 0) > lp_th,
                "severity_levels": {
                    "critical": lambda f: f.get("param_count", 0) > lp_cr,
                    "high":     lambda f: f.get("param_count", 0) > lp_sv,
                    "medium":   lambda f: f.get("param_count", 0) > lp_th,
                },
                "thresholds": {"threshold": lp_th, "severe": lp_sv, "critical": lp_cr},
            },

            # NOTE: Complex Conditional uses conditional_count (if/else/switch branches)
            # NOT responsibility_count — that belongs to Multifaceted Abstraction.
            "Complex Conditional": {
                "name":        "Complex Conditional",
                "description": f"Method has more than {cc_th} conditional branches — hard to follow.",
                "refactor_guidance": (
                    "Apply Guard Clauses (early returns to reduce nesting), "
                    "replace if-else chains with Map-based dispatch, "
                    "or use the Strategy pattern."
                ),
                "metric": "conditional_count",
                "condition": lambda f: f.get("conditional_count", 0) > cc_th,
                "severity_levels": {
                    "critical": lambda f: f.get("conditional_count", 0) > cc_cr,
                    "high":     lambda f: f.get("conditional_count", 0) > cc_sv,
                    "medium":   lambda f: f.get("conditional_count", 0) > cc_th,
                },
                "thresholds": {"threshold": cc_th, "severe": cc_sv, "critical": cc_cr},
            },

            # NOTE: Multifaceted Abstraction uses responsibility_count
            # (I/O + persistence + iteration + decision = 4 responsibilities)
            "Multifaceted Abstraction": {
                "name":        "Multifaceted Abstraction",
                "description": f"Method has more than {ma_th} distinct responsibilities — violates SRP.",
                "refactor_guidance": (
                    "Apply Single Responsibility Principle. "
                    "Extract each responsibility (I/O, calculation, persistence, etc.) "
                    "into its own focused private method."
                ),
                "metric": "responsibility_count",
                "condition": lambda f: f.get("responsibility_count", 1) > ma_th,
                "severity_levels": {
                    "critical": lambda f: f.get("responsibility_count", 1) > ma_cr,
                    "high":     lambda f: f.get("responsibility_count", 1) > ma_sv,
                    "medium":   lambda f: f.get("responsibility_count", 1) > ma_th,
                },
                "thresholds": {"threshold": ma_th, "severe": ma_sv, "critical": ma_cr},
            },
        }

    def get_smell_info(self, smell_name: str) -> Optional[Dict]:
        return self.smells.get(smell_name)

    def get_all_smells(self) -> Dict:
        return self.smells

    def get_dacos_context(self) -> str:
        if self.dacos and self.initialized:
            return self.dacos.generate_dacos_context()
        return "DACOS dataset not loaded. Using standard thresholds."

    def get_severity(self, smell_name: str, func: dict) -> str:
        smell = self.smells.get(smell_name)
        if not smell:
            return "unknown"
        levels = smell.get("severity_levels", {})
        if levels.get("critical", lambda x: False)(func): return "critical"
        elif levels.get("high",   lambda x: False)(func): return "high"
        elif levels.get("medium", lambda x: False)(func): return "medium"
        else:                                              return "low"

    def reload(self) -> bool:
        if self.dacos_folder and self.dacos_folder != "SKIP":
            return self._initialize_dacos()
        return False


# Backward compatibility
DACOS_KNOWLEDGE = DACOSKnowledgeBase().get_all_smells()
