# agents/refactor_agent.py
#
# Block 5a: Refactor Agent
# Fine-tuned CodeT5+ 770M with LoRA adapter for Java code refactoring.
# Trained on RCCT Java dataset.
#
# DESIGN PRINCIPLE:
#   This agent does NOT build its own prompts.
#   It RECEIVES the prompt from the Prompting Engine (pipeline.py).
#   This ensures smell context and DACOS thresholds are used correctly.
#
# HONEST LIMITATIONS:
#   - Trained on RCCT Java (Java 8 style). Acknowledge in paper.
#   - Runs on CPU by default — inference takes 1-3 minutes per run.
#   - If model output is invalid Java, falls back to the original code.

import re
import json
import shutil
import tempfile
import logging
from pathlib import Path

import torch
import safetensors.torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "Salesforce/codet5p-770m"


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer loader — patches the added_tokens format bug in codet5p-770m
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokenizer(source_dir: Path) -> AutoTokenizer:
    """
    Load the tokenizer from the adapter folder with an added_tokens format patch.
    codet5p-770m stores added_tokens as plain strings instead of dicts,
    which causes AutoTokenizer to crash. This patches them to the correct format.
    """
    tmp = Path(tempfile.mkdtemp(prefix="tok_refactor_"))
    try:
        for pattern in ["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"]:
            for f in source_dir.glob(pattern):
                shutil.copy(f, tmp / f.name)

        tok_json = tmp / "tokenizer.json"
        if tok_json.exists():
            with open(tok_json, encoding="utf-8") as fh:
                data = json.load(fh)

            fixed = []
            for entry in data.get("added_tokens", []):
                if isinstance(entry, str):
                    fixed.append({
                        "id": 0, "content": entry, "single_word": False,
                        "lstrip": False, "rstrip": False,
                        "normalized": False, "special": False,
                    })
                else:
                    fixed.append(entry)
            data["added_tokens"] = fixed

            with open(tok_json, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

        return AutoTokenizer.from_pretrained(str(tmp), use_fast=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# RefactorAgent
# ─────────────────────────────────────────────────────────────────────────────

class RefactorAgent:
    """
    Refactor Agent — fine-tuned CodeT5+ 770M with LoRA adapter for Java.
    Receives prompt from the Prompting Engine, returns refactored Java code.
    Falls back to the original code (cleaned) if model output is invalid.
    """

    def __init__(self, adapter_path: str,
                 max_input_length: int  = 512,
                 max_output_length: int = 256):
        self.adapter_path      = Path(adapter_path)
        self.max_input_length  = max_input_length
        self.max_output_length = max_output_length  # 256 is enough; 512 is very slow on CPU
        self.tokenizer = None
        self.model     = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded   = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load base model + LoRA adapter. Returns True if successful."""
        if self._loaded:
            return True

        if not self.adapter_path.exists():
            logger.error(f"[RefactorAgent] Adapter not found: {self.adapter_path}")
            logger.error("[RefactorAgent] Copy trained adapter files to models/refactor_agent_final/")
            return False

        try:
            logger.debug(f"[RefactorAgent] Device: {self.device.upper()}")
            logger.debug(f"[RefactorAgent] Loading base model {BASE_MODEL_NAME} ...")

            # Silence the noisy safetensors background conversion thread
            logging.getLogger("transformers.safetensors_conversion").setLevel(logging.ERROR)

            self.tokenizer = _load_tokenizer(self.adapter_path)

            base_model = T5ForConditionalGeneration.from_pretrained(
                BASE_MODEL_NAME,
                use_safetensors=False,
            )
            base_model = base_model.to(self.device)

            logger.debug(f"[RefactorAgent] Loading LoRA adapter from {self.adapter_path} ...")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                is_trainable=False,
            )

            # ── Key-remapping fix ──────────────────────────────────────────
            # PEFT version on Kaggle (where the model was trained) differs from
            # the local version. The saved keys use the old naming convention.
            # Remap them so load_state_dict matches correctly.
            weights_file = self.adapter_path / "adapter_model.safetensors"
            if weights_file.exists():
                state_dict = safetensors.torch.load_file(str(weights_file))
                remapped = {}
                for k, v in state_dict.items():
                    nk = k.replace("base_model.model.base_model.model.", "base_model.model.")
                    nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                    nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                    remapped[nk] = v
                result  = self.model.load_state_dict(remapped, strict=False)
                matched = len(remapped) - len(result.unexpected_keys)
                logger.debug(f"[RefactorAgent] ✅ LoRA loaded ({matched}/{len(remapped)} keys)")

            self.model.eval()
            self._loaded = True
            logger.debug("[RefactorAgent] ✅ Ready.")
            return True

        except Exception as e:
            logger.error(f"[RefactorAgent] Load failed: {e}")
            return False

    def run(self, prompt: str, original_code: str) -> dict:
        """
        Run the refactor agent.

        Args:
            prompt:        Full prompt from the Prompting Engine
            original_code: Raw original Java code (used as fallback)

        Returns:
            dict with keys:
              refactored_code : str
              used_model      : bool  — True if model output was accepted
              valid_java      : bool  — True if output passed Java validation
        """
        if not self._loaded:
            if not self.load():
                logger.warning("[RefactorAgent] Model unavailable — using fallback.")
                return self._fallback(original_code)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
                padding=True,
                return_attention_mask=True,
            ).to(self.device)

            logger.debug("[RefactorAgent] Running inference "
                        f"(input: {inputs['input_ids'].shape[1]} tokens, "
                        f"device: {self.device}) ...")

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_output_length,
                    num_beams=2,            # 2 beams: good quality, half the CPU time vs 4
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                )

            raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"[RefactorAgent] Raw output ({len(raw)} chars)")

            candidate = self._extract_code(raw, original_code)
            candidate = self._post_process_java(candidate)  # fix #comments, },  unclosed braces
            candidate = self._cleanup(candidate)

            if self._is_valid_java(candidate):
                logger.debug("[RefactorAgent] ✅ Model output accepted.")
                return {"refactored_code": candidate, "used_model": True, "valid_java": True}
            else:
                logger.warning("[RefactorAgent] Model output invalid Java — using fallback.")
                return self._fallback(original_code)

        except Exception as e:
            logger.error(f"[RefactorAgent] Inference error: {e}")
            return self._fallback(original_code)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_code(self, raw: str, fallback: str) -> str:
        """
        Strip prompt echoes and markdown fences from the model's raw output.
        Markers match the templates in prompt_engine/templates.py.
        """
        # Strip output trigger phrases — stop after the first match so that
        # splitting on "```java" does not then re-split on the closing "```".
        for marker in (
            "Return ONLY the refactored Java code:",  # all smell templates
            "Return ONLY the refactored Java code",   # without colon variant
            "Refactored Java code:",
        ):
            if marker in raw:
                raw = raw.split(marker, 1)[-1]
                break   # stop — don't apply more splits after finding the trigger

        # Remove any remaining markdown fences (``` or ```java) in one pass
        raw = re.sub(r"```[a-zA-Z]*", "", raw).strip()

        # Skip leading non-code lines — find first line that looks like Java.
        # Only break on a line that starts with a known Java keyword/modifier.
        # Do NOT break on arbitrary text lines (that was causing preambles to be included).
        lines    = raw.splitlines()
        starters = ("public ", "private ", "protected ", "class ", "import ",
                    "package ", "/*", "/**", "//", "@", "void ", "static ",
                    "abstract ", "final ", "synchronized ", "native ")
        start    = 0
        found    = False
        for i, line in enumerate(lines):
            s = line.strip()
            if s and any(s.startswith(p) for p in starters):
                start = i
                found = True
                break

        # If no Java starter found, take from the first non-empty line as last resort
        if not found:
            for i, line in enumerate(lines):
                if line.strip():
                    start = i
                    break

        candidate = "\n".join(lines[start:]).strip()
        # Use >= 10 so short but valid outputs like "public void m() {}" are accepted
        return candidate if len(candidate) >= 10 else fallback

    def _post_process_java(self, code: str) -> str:
        """
        Fix the specific syntax errors the model tends to produce:
        1. },   →  }         (invalid comma after closing brace)
        2. #... →  (removed)  (Python-style comments the model occasionally generates)
        3. Unclosed braces   (model hits max_new_tokens mid-method, close them)
        """
        import re
        lines = code.splitlines()
        fixed = []

        for line in lines:
            # Fix 1: strip Python # comments (leave // Java comments intact)
            result = []
            in_string = False
            for i, c in enumerate(line):
                if c == '"' and (i == 0 or line[i-1] != '\\'):
                    in_string = not in_string
                if c == '#' and not in_string:
                    break
                result.append(c)
            line = ''.join(result).rstrip()

            # Fix 2: remove comma after closing brace
            line = re.sub(r'\},\s*$', '}', line)
            line = re.sub(r'\},\s*(else|catch|finally)', r'} \1', line)

            fixed.append(line)

        code = '\n'.join(fixed)

        # Fix 3: close unclosed braces (model truncated mid-method)
        opens  = code.count('{')
        closes = code.count('}')
        missing = opens - closes
        if 0 < missing <= 10:   # sanity limit — don't add braces to garbage
            code = code.rstrip()
            for i in range(missing):
                indent = max(0, (missing - i - 1)) * 4
                code += '\n' + (' ' * indent) + '}'

        return code.strip()

    def _cleanup(self, code: str) -> str:
        """Light Java style cleanup — normalise indentation and blank lines."""
        try:
            lines = code.split("\n")
            fixed = []
            for line in lines:
                stripped = line.lstrip()
                if not stripped:
                    fixed.append("")
                    continue
                indent = len(line) - len(stripped)
                # Round up to the nearest 4-space block.
                # e.g. 3 spaces → 4, 5 spaces → 8, 7 spaces → 8
                # This handles code with inconsistent indentation.
                if indent == 0:
                    spaces = 0
                else:
                    spaces = ((indent + 3) // 4) * 4
                fixed.append(" " * spaces + stripped)
            code = "\n".join(fixed)
            code = re.sub(r"\n{3,}", "\n\n", code)   # max 1 blank line
            if not code.endswith("\n"):
                code += "\n"
            return code
        except Exception:
            return code

    def _fallback(self, original_code: str) -> dict:
        """
        Return the original code (lightly cleaned) when model output is invalid.
        valid_java is False — the model did not produce accepted output.
        The evaluator uses this flag to distinguish model success from fallback.
        """
        cleaned = self._cleanup(original_code)
        return {"refactored_code": cleaned, "used_model": False, "valid_java": False}

    def _is_valid_java(self, code: str) -> bool:
        """
        Validate model output using javalang (Java 8 strict parser).

        - Bare methods (no class keyword) are wrapped in a dummy class.
        - If javalang rejects the code → invalid, return False.
        - If javalang is not installed → fall back to structural heuristics.
        """
        if not code or len(code.strip()) < 20:
            return False

        try:
            import javalang
            test_code = (
                code if "class " in code
                else f"public class _Wrapper {{\n{code}\n}}"
            )
            javalang.parse.parse(test_code)
            return True

        except ImportError:
            pass   # javalang not installed — use heuristic below

        except Exception:
            return False   # javalang rejected it — genuinely invalid

        # ── Heuristic (only if javalang not installed) ────────────────────
        if code.count("{") != code.count("}"):
            return False
        python_signs = ["\ndef ", "\nelif ", "def __"]
        if any(s in code for s in python_signs):
            return False
        java_signs = ["public ", "private ", "protected ", "void ",
                      "class ", "return ", "int ", "double ", "String "]
        return any(s in code for s in java_signs)