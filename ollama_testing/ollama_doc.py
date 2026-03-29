"""
ollama_doc.py — Documentation via Ollama
"""
import os, re, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ollama_client import generate, is_ollama_running, list_pulled_models

SYSTEM = (
    "You are an expert Java developer writing Javadoc documentation. "
    "Generate complete, professional Javadoc. "
    "Include: one-sentence description, @param for EVERY parameter, @return. "
    "Return ONLY the Javadoc block starting with /** and ending with */. "
    "No explanation. No code. No markdown."
)

def build_prompt(base_prompt):
    return (f"{SYSTEM}\n\n{base_prompt}\n\n"
            "Return ONLY the Javadoc block (/** ... */). "
            "Include @param for every parameter and @return.")

def clean_javadoc(raw):
    raw = raw.strip()
    # Remove markdown fences
    raw = re.sub(r"```(?:java)?\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()

    # Extract /** ... */ block
    match = re.search(r"/\*\*.*?\*/", raw, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Wrap if needed
    if not raw.startswith("/**"):
        raw = "/**\n" + raw
    if not raw.endswith("*/"):
        raw = raw + "\n */"
    return raw.strip()

def run_doc_test(prompt, models, output_dir):
    if not is_ollama_running():
        print("ERROR: Ollama not running.")
        return {}

    pulled  = list_pulled_models()
    results = {}
    full_prompt = build_prompt(prompt)

    for model in models:
        print(f"\n  [{model}] Running documentation ...")
        short = model.split(":")[0]
        if not any(short in p for p in pulled):
            print(f"  SKIP — not pulled. Run: ollama pull {model}")
            results[model] = {"success": False, "error": "not pulled", "output_cleaned": ""}
            continue

        res = generate(model, full_prompt, temperature=0.1, max_tokens=512)

        if res["success"]:
            res["output_cleaned"] = clean_javadoc(res["response"])
            print(f"  Done in {res['duration_seconds']}s")
            preview = res['output_cleaned'][:300]
            print(f"  Preview:\n{preview}")
        else:
            res["output_cleaned"] = ""
            print(f"  FAILED: {res['error']}")

        results[model] = res

        # Save per-model outputs
        if output_dir:
            safe = model.replace(":", "_").replace("/", "_").replace(".", "_")
            mdir = os.path.join(output_dir, safe)
            os.makedirs(mdir, exist_ok=True)

            if res.get("output_cleaned"):
                with open(f"{mdir}/documentation.md", "w", encoding="utf-8") as f:
                    f.write(f"# Javadoc — {model}\n\n```java\n")
                    f.write(res["output_cleaned"])
                    f.write("\n```\n")

            with open(f"{mdir}/result_doc.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in res.items() if k != "response"},
                          f, indent=2, ensure_ascii=False)

    return results
