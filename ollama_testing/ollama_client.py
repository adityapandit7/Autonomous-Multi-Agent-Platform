"""
ollama_client.py — Ollama REST API wrapper
"""
import requests, json, time

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "codellama:7b",
    "deepseek-coder:6.7b",
    "gemma3:12b",
    "qwen2.5vl:7b",
]

def is_ollama_running():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except:
        return False

def list_pulled_models():
    try:
        return [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])]
    except:
        return []

def generate(model, prompt, temperature=0.1, max_tokens=1024, timeout=180):
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens,
                               "top_p": 0.9, "repeat_penalty": 1.1}},
            timeout=timeout,
        )
        data = r.json()
        return {
            "response":         data.get("response", "").strip(),
            "model":            model,
            "duration_seconds": round(time.time() - start, 2),
            "success":          True,
            "error":            None,
        }
    except requests.exceptions.Timeout:
        return {"response": "", "model": model,
                "duration_seconds": round(time.time()-start,2),
                "success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"response": "", "model": model,
                "duration_seconds": round(time.time()-start,2),
                "success": False, "error": str(e)}
