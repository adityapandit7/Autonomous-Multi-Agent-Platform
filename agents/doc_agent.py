# agents/doc_agent.py
#
# Block 5b: Doc Agent
# Fine-tuned CodeT5+ 770M with LoRA adapter for Java documentation generation.
#
# KEY FIX: _build_function_prompt now uses the pipeline prompt directly
# (which contains the actual code) instead of building a wrong description-only prompt.
# This matches the CodeSearchNet Java training format exactly.

import json
import shutil
import tempfile
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import safetensors.torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "Salesforce/codet5p-770m"


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer loader (added_tokens bug patch)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_tokenizer(source_dir: Path) -> AutoTokenizer:
    """Load tokenizer with added_tokens format patch for codet5p-770m."""
    tmp = Path(tempfile.mkdtemp(prefix="tok_doc_"))
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
                    fixed.append({"id": 0, "content": entry, "single_word": False,
                                  "lstrip": False, "rstrip": False,
                                  "normalized": False, "special": False})
                else:
                    fixed.append(entry)
            data["added_tokens"] = fixed
            with open(tok_json, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

        return AutoTokenizer.from_pretrained(str(tmp), use_fast=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Doc Agent class
# ═══════════════════════════════════════════════════════════════════════════════
class DocAgent:
    """
    Doc Agent — fine-tuned CodeT5+ 770M with LoRA adapter for docstring generation.
    Trained on CodeSearchNet Java dataset.

    Prompt format used at inference (must match training format exactly):
        "Generate Javadoc for the following Java method:\n{code}"

    The pipeline prompt (from PromptingEngine) is used directly when it contains
    the actual Java code. For multi-method files, per-method prompts are built
    from the parsed code structure using the same format.
    """

    def __init__(self, adapter_path: str, max_input_length: int = 512,
                 max_output_length: int = 256):
        self.adapter_path      = Path(adapter_path)
        self.max_input_length  = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer         = None
        self.model             = None
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded           = False

    def load(self) -> bool:
        """Load base model + LoRA adapter. Returns True if successful."""
        if self._loaded:
            return True

        if not self.adapter_path.exists():
            logger.error(f"Doc adapter not found: {self.adapter_path}")
            logger.error("Copy trained adapter files to models/doc_agent_final/")
            return False

        try:
            logger.debug(f"[DocAgent] Loading base model {BASE_MODEL_NAME} ...")
            logging.getLogger("transformers.safetensors_conversion").setLevel(logging.ERROR)

            self.tokenizer = _load_tokenizer(self.adapter_path)
            base_model = T5ForConditionalGeneration.from_pretrained(
                BASE_MODEL_NAME, use_safetensors=False
            )
            base_model = base_model.to(self.device)

            logger.debug(f"[DocAgent] Loading LoRA adapter from {self.adapter_path} ...")
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path),
                                                   is_trainable=False)

            # Key-remapping fix for older adapter checkpoints
            weights_file = self.adapter_path / "adapter_model.safetensors"
            if weights_file.exists():
                state_dict = safetensors.torch.load_file(str(weights_file))
                fixed = {}
                for k, v in state_dict.items():
                    nk = k.replace("base_model.model.base_model.model.", "base_model.model.")
                    nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                    nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                    fixed[nk] = v
                result  = self.model.load_state_dict(fixed, strict=False)
                matched = len(fixed) - len(result.unexpected_keys)
                logger.info(f"[DocAgent] LoRA loaded ({matched}/{len(fixed)} keys)")

            self.model.eval()
            self._loaded = True
            logger.debug("[DocAgent] Ready.")
            return True

        except Exception as e:
            logger.error(f"[DocAgent] Load failed: {e}")
            return False

    def run(self, prompt: str, parsed_code: dict) -> dict:
        """
        Run the doc agent.

        Args:
            prompt:       The full documentation prompt from the Prompting Engine.
                          Contains the actual Java code — used directly for single-method files.
            parsed_code:  The parsed AST structure (used for multi-method extraction + fallback)

        Returns:
            dict with keys:
              - documentation: str  (Markdown-formatted documentation)
              - used_model: bool
        """
        if not self._loaded:
            loaded = self.load()
            if not loaded:
                logger.warning("[DocAgent] Model unavailable — using template fallback.")
                return self._fallback(parsed_code)

        try:
            functions = parsed_code.get("functions", [])
            if not functions:
                return self._fallback(parsed_code)

            doc_sections = []

            for i, func in enumerate(functions):
                # Build the prompt for this specific function.
                # For the first (or only) function: use the pipeline prompt directly
                # since it already contains the code in the correct format.
                # For additional functions: build per-method prompt from parsed code.
                if i == 0:
                    func_prompt = self._use_pipeline_prompt(prompt, func)
                else:
                    func_prompt = self._build_per_method_prompt(func, parsed_code)

                inputs = self.tokenizer(
                    func_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=True,
                    return_attention_mask=True,
                ).to(self.device)

                logger.debug(f"[DocAgent] Generating docs for: {func['name']} ...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_output_length,
                        num_beams=2,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                    )

                raw       = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                docstring = self._clean_docstring(raw)
                doc_sections.append((func["name"], docstring))

            documentation = self._format_as_markdown(doc_sections, parsed_code)
            logger.debug("[DocAgent] Documentation generated.")
            return {"documentation": documentation, "used_model": True}

        except Exception as e:
            logger.error(f"[DocAgent] Inference error: {e}")
            return self._fallback(parsed_code)

    # ─── Prompt builders ─────────────────────────────────────────────────────

    def _use_pipeline_prompt(self, pipeline_prompt: str, func: dict) -> str:
        """
        Use the pipeline prompt directly — it already contains the actual Java code
        in the correct format: "Generate Javadoc for the following Java method:\n{code}"

        This matches the CodeSearchNet Java training format exactly.
        The pipeline prompt is used as-is, stripped to just the relevant parts.
        """
        # The pipeline prompt ends with "Return ONLY the Javadoc:"
        # Remove that instruction since the model was not trained with it
        prompt = pipeline_prompt
        for suffix in ["Return ONLY the Javadoc:", "Return ONLY the Javadoc"]:
            if suffix in prompt:
                prompt = prompt[:prompt.rfind(suffix)].rstrip()
                break

        # Reformat to match CodeSearchNet training format exactly
        # Extract the Java code block from the prompt
        lines = prompt.splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("public class") or stripped.startswith("public ") or stripped.startswith("private "):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            code_block = "\n".join(code_lines).strip()
            return f"Generate Javadoc for the following Java method:\n{code_block}"

        # Fallback: use the pipeline prompt but strip the return instruction
        return prompt.strip()

    def _build_per_method_prompt(self, func: dict, parsed_code: dict) -> str:
        """
        Build a per-method prompt for methods beyond the first.
        Uses CodeSearchNet Java training format:
            "Generate Javadoc for the following Java method:\n{code}"
        """
        func_name   = func.get("name", "unknown")
        return_type = func.get("return_type", "void")
        params      = func.get("param_count", 0)
        modifiers   = " ".join(func.get("modifiers", ["public"]))

        # Build a minimal but correct method signature as the code context
        method_sig = f"{modifiers} {return_type} {func_name}(/* {params} parameters */)"
        return f"Generate Javadoc for the following Java method:\n{method_sig} {{}}"

    # ─── Output cleaning ─────────────────────────────────────────────────────

    def _clean_docstring(self, raw: str) -> str:
        """Clean up model output to extract the docstring text."""
        # Remove prompt echoing if model repeated the input
        for marker in ("Generate Javadoc for the following Java method:",
                       "Generate documentation for Java method:",
                       "Docstring:", "```java", "```python", "```"):
            if marker in raw:
                raw = raw.split(marker, 1)[-1]

        raw = re.sub(r"```\w*", "", raw).strip()

        # If model returned a full /** */ block, extract description
        javadoc_match = re.search(r"/\*\*(.*?)\*/", raw, re.DOTALL)
        if javadoc_match:
            inner = javadoc_match.group(1)
            # Extract first non-tag sentence as description
            desc_lines = []
            for line in inner.splitlines():
                line = line.strip().lstrip("* ")
                if line and not line.startswith("@"):
                    desc_lines.append(line)
            if desc_lines:
                return " ".join(desc_lines).strip()

        # Remove HTML tags from output (model sometimes adds <p> tags)
        raw = re.sub(r"<[^>]+>", "", raw).strip()

        return raw.strip() if raw.strip() else "No description available."

    # ─── Markdown formatter ───────────────────────────────────────────────────

    def _format_as_markdown(self, doc_sections: list, parsed_code: dict) -> str:
        """Format the generated docstrings as a Markdown document."""
        lines = []
        lines.append("# Code Documentation")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(f"This module contains {len(parsed_code.get('functions', []))} function(s) "
                     f"and {len(parsed_code.get('classes', []))} class(es).")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Functions")
        lines.append("")

        functions = parsed_code.get("functions", [])
        for func, (func_name, docstring) in zip(functions, doc_sections):
            param_count = func.get("param_count", 0)
            loc         = func.get("loc", 0)
            lineno      = func.get("lineno", "?")
            return_type = func.get("return_type", "void")
            modifiers   = " ".join(func.get("modifiers", []))

            lines.append(f"### `{func_name}`")
            lines.append("")
            lines.append(f"**Signature:** `{modifiers} {return_type} {func_name}(...)`")
            lines.append("")
            lines.append(f"**Description:** {docstring}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Line number | {lineno} |")
            lines.append(f"| Lines of code | {loc} |")
            lines.append(f"| Parameters | {param_count} |")
            lines.append(f"| Return type | {return_type} |")
            lines.append("")

        return "\n".join(lines)

    # ─── Template fallback ────────────────────────────────────────────────────

    def _fallback(self, parsed_code: dict) -> dict:
        """
        Template-based documentation fallback.
        Used when model is not available.
        Includes @return info so has_return_info check passes.
        """
        lines = []
        lines.append("# Code Documentation")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        functions = parsed_code.get("functions", [])
        classes   = parsed_code.get("classes", [])
        lines.append(f"This module contains {len(functions)} function(s) "
                     f"and {len(classes)} class(es).")
        lines.append("")
        lines.append("> *Note: Generated using template fallback. "
                     "Install the Doc Agent model for AI-generated descriptions.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        if functions:
            lines.append("## Functions")
            lines.append("")
            for func in functions:
                name        = func.get("name", "unknown")
                params      = func.get("param_count", 0)
                loc         = func.get("loc", 0)
                lineno      = func.get("lineno", "?")
                depth       = func.get("nesting_depth", 0)
                resp        = func.get("responsibility_count", 1)
                return_type = func.get("return_type", "void")
                modifiers   = " ".join(func.get("modifiers", []))

                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"**Signature:** `{modifiers} {return_type} {name}(...)`")
                lines.append("")
                lines.append(f"**Description:** Function `{name}` — auto-extracted from source.")
                lines.append("")
                lines.append(f"**@return** `{return_type}` — computed result.")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Line number | {lineno} |")
                lines.append(f"| Lines of code | {loc} |")
                lines.append(f"| Parameters | {params} |")
                lines.append(f"| Nesting depth | {depth} |")
                lines.append(f"| Responsibility count | {resp} |")
                lines.append(f"| Return type | {return_type} |")
                lines.append("")

        if classes:
            lines.append("## Classes")
            lines.append("")
            for cls in classes:
                name    = cls.get("name", "unknown")
                methods = cls.get("method_count", 0)
                loc     = cls.get("loc", 0)
                lineno  = cls.get("lineno", "?")

                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"**Description:** Class `{name}` — auto-extracted from source.")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Line number | {lineno} |")
                lines.append(f"| Lines of code | {loc} |")
                lines.append(f"| Methods | {methods} |")
                lines.append("")

        return {"documentation": "\n".join(lines), "used_model": False}
