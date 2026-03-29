from .prompting_engine import PromptingEngine
from smell_detector.smell_detector import SmellDetector
from knowledge_base.dacos_knowledge import DACOSKnowledgeBase
from .templates import REFACTOR_BASE_TEMPLATE, DOCUMENTATION_BASE_TEMPLATE, get_refactor_template

try:
    from knowledge_base.dacos_integration import init_dacos, get_dacos
except ImportError:
    def init_dacos(path): return None
    def get_dacos(): return None

try:
    from knowledge_base.dacos_evaluator import DACOSEvaluator
except ImportError:
    DACOSEvaluator = None

__all__ = [
    "PromptingEngine", "SmellDetector", "DACOSKnowledgeBase",
    "REFACTOR_BASE_TEMPLATE", "DOCUMENTATION_BASE_TEMPLATE",
    "get_refactor_template", "init_dacos", "get_dacos",
]
