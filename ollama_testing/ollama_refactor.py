"""
ollama_refactor.py — Refactoring via Ollama
"""
import os, re, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ollama_client import generate, is_ollama_running, list_pulled_models

SYSTEM = (
    "You are an expert Java developer performing code refactoring. "
    "Follow the instructions exactly. Return ONLY valid Java code. "
    "No markdown, no explanation, no ```java blocks. Start with the Java code directly."
)

def build_prompt(base_prompt):
    return (f"{SYSTEM}\n\n{base_prompt}\n\n"
            "IMPORTANT: Return ONLY Java code. No explanations. No backticks.")

def clean_java(raw):
    raw = re.sub(r"```(?:java)?\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    # Remove non-Java unicode noise
    raw = raw.encode("ascii", errors="replace").decode("ascii").replace("?", "")
    lines = raw.strip().split("\n")
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if any(s.startswith(k) for k in ["public","private","protected","//","/*","import","class"]):
            start = i
            break
    return "\n".join(lines[start:]).strip()

def run_refactor_test(prompt, models, output_dir):
    if not is_ollama_running():
        print("ERROR: Ollama not running. Start with: ollama serve")
        return {}

    pulled  = list_pulled_models()
    results = {}
    full_prompt = build_prompt(prompt)

    for model in models:
        print(f"\n  [{model}] Running refactoring ...")
        short = model.split(":")[0]
        if not any(short in p for p in pulled):
            print(f"  SKIP — not pulled. Run: ollama pull {model}")
            results[model] = {"success": False, "error": "not pulled", "output_cleaned": ""}
            continue

        res = generate(model, full_prompt, temperature=0.1, max_tokens=1024)

        if res["success"]:
            res["output_cleaned"] = clean_java(res["response"])
            print(f"  Done in {res['duration_seconds']}s")
            preview = res['output_cleaned'][:150].replace('\n',' ')
            print(f"  Preview: {preview} ...")
        else:
            res["output_cleaned"] = ""
            print(f"  FAILED: {res['error']}")

        results[model] = res

        # Save per-model outputs
        if output_dir:
            safe  = model.replace(":", "_").replace("/", "_").replace(".", "_")
            mdir  = os.path.join(output_dir, safe)
            os.makedirs(mdir, exist_ok=True)

            if res.get("output_cleaned"):
                with open(f"{mdir}/refactored_code.java", "w", encoding="utf-8") as f:
                    f.write(res["output_cleaned"])

            with open(f"{mdir}/result_refactor.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in res.items() if k != "response"},
                          f, indent=2, ensure_ascii=False)

    return results
