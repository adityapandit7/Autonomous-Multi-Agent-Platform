"""
generate_graphs.py
Autonomous Multi-Agent Platform — Graph Generator

Run from project root:
    python generate_graphs.py

Output: graphs/ folder with 4 PNG files
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# ── Output folder ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
GRAPHS_DIR = ROOT / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "figure.dpi":       140,
})
PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72", "#bc8cff", "#ffb77a",
]

# ── Load data ─────────────────────────────────────────────────────────────────
# Find the most recent pipeline run
runs = sorted((ROOT / "outputs").glob("run_*"), reverse=True)
if not runs:
    print("ERROR: No pipeline run found in outputs/")
    exit(1)
run_dir = runs[0]
print(f"Using pipeline run: {run_dir.name}")

# Load ollama summary — find most recent
ollama_summaries = sorted(
    (ROOT / "ollama_testing" / "ollama_outputs").glob("ollama_summary_*.json"),
    reverse=True
)
if not ollama_summaries:
    print("ERROR: No ollama_summary_*.json found in ollama_testing/ollama_outputs/")
    exit(1)
print(f"Using Ollama summary: {ollama_summaries[0].name}")

with open(run_dir / "EVALUATION_refactor.json", encoding="utf-8") as f:
    ref_eval = json.load(f)
with open(run_dir / "EVALUATION_doc.json", encoding="utf-8") as f:
    doc_eval = json.load(f)
with open(ollama_summaries[0], encoding="utf-8") as f:
    ollama = json.load(f)

# ── Build data arrays ─────────────────────────────────────────────────────────
models_tested = ollama["models_tested"]
models_short  = [
    "CodeT5+\n(fine-tuned)",
    "CodeLlama\n7B",
    "DeepSeek\n6.7B",
    "Gemma3\n12B",
    "Qwen2.5\n7B",
]
colors_models = [PALETTE[0], PALETTE[2], PALETTE[4], PALETTE[1], PALETTE[3]]

refactor_conf = [ref_eval["confidence"]["score"]] + [
    ollama["refactoring"][m]["confidence"] for m in models_tested
]
refactor_ast  = [ref_eval["ast_validity"]["valid"]] + [
    ollama["refactoring"][m]["ast_valid"] for m in models_tested
]
refactor_loc  = [ref_eval["improvement"]["loc_refactored"]] + [
    ollama["refactoring"][m]["loc_refactored"] for m in models_tested
]
loc_original  = ref_eval["improvement"]["loc_original"]

doc_conf   = [doc_eval["confidence"]["score"]] + [
    ollama["documentation"][m]["confidence"] for m in models_tested
]
doc_param  = [1/6] + [                          # CodeT5+: 1 of 6 params documented
    ollama["documentation"][m]["param_coverage"] for m in models_tested
]
doc_return = [1.0] + [
    1.0 if ollama["documentation"][m]["has_return"] else 0.0
    for m in models_tested
]

x = np.arange(len(models_short))


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Refactoring confidence
# ════════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    "Refactoring Confidence — CodeT5+ vs Ollama Models",
    color="#58a6ff", fontsize=14, fontweight="bold",
)
bars = ax.bar(x, refactor_conf, color=colors_models, alpha=0.9, width=0.55)
for b, v, ast in zip(bars, refactor_conf, refactor_ast):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", color="#c9d1d9", fontsize=11, fontweight="bold")
    label = "AST: OK" if ast else "AST: FAIL"
    color = "#3fb950"  if ast else "#f78166"
    ax.text(b.get_x() + b.get_width() / 2, 0.04, label,
            ha="center", color=color, fontsize=8)
ax.axhline(0.75, color="#ffa657", lw=1.5, ls="--", alpha=0.8,
           label="Accept threshold (0.75)")
ax.set_xticks(x)
ax.set_xticklabels(models_short, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Confidence Score")
ax.legend()
ax.grid(axis="y")
for s in ax.spines.values():
    s.set_edgecolor("#30363d")
plt.tight_layout()
out1 = GRAPHS_DIR / "graph1_refactor_confidence.png"
plt.savefig(out1, dpi=140, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — LOC reduction
# ════════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    "Lines of Code Before and After Refactoring",
    color="#58a6ff", fontsize=14, fontweight="bold",
)
w  = 0.35
b1 = ax.bar(x - w / 2, [loc_original] * 5, w,
            label=f"Original ({loc_original} LOC)", color="#30363d", alpha=0.9)
b2 = ax.bar(x + w / 2, refactor_loc, w,
            label="Refactored", color=colors_models, alpha=0.9)
for b, v in zip(b2, refactor_loc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4,
            f"{v}\n(-{loc_original - v})",
            ha="center", color="#c9d1d9", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models_short, fontsize=10)
ax.set_ylim(0, loc_original + 15)
ax.set_ylabel("Lines of Code")
ax.legend()
ax.grid(axis="y")
for s in ax.spines.values():
    s.set_edgecolor("#30363d")
plt.tight_layout()
out2 = GRAPHS_DIR / "graph2_loc_reduction.png"
plt.savefig(out2, dpi=140, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out2}")


# ════════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Documentation quality
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Documentation Quality — CodeT5+ vs Ollama Models",
    color="#58a6ff", fontsize=14, fontweight="bold",
)
w = 0.28
b1 = axes[0].bar(x - w / 2, doc_param,  w,
                 label="@param coverage", color=PALETTE[0], alpha=0.9)
b2 = axes[0].bar(x + w / 2, doc_return, w,
                 label="@return rate",   color=PALETTE[2], alpha=0.9)
for b, v in list(zip(b1, doc_param)) + list(zip(b2, doc_return)):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", color="#c9d1d9", fontsize=9, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_short, fontsize=9)
axes[0].set_ylim(0, 1.3)
axes[0].set_ylabel("Coverage Rate")
axes[0].set_title("@param and @return coverage")
axes[0].legend()
axes[0].grid(axis="y")

bars = axes[1].bar(x, doc_conf, color=colors_models, alpha=0.9, width=0.55)
for b, v in zip(bars, doc_conf):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", color="#c9d1d9", fontsize=11, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_short, fontsize=9)
axes[1].set_ylim(0, 1.2)
axes[1].set_ylabel("Confidence Score")
axes[1].set_title("Documentation confidence")
axes[1].grid(axis="y")

for ax2 in axes:
    for s in ax2.spines.values():
        s.set_edgecolor("#30363d")
plt.tight_layout()
out3 = GRAPHS_DIR / "graph3_doc_comparison.png"
plt.savefig(out3, dpi=140, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Saved: {out3}")

print(f"\nAll 3 graphs saved to: {GRAPHS_DIR}")
print("Files:")
for f in sorted(GRAPHS_DIR.glob("*.png")):
    print(f"  {f.name}")
