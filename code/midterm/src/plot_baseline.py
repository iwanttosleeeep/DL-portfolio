import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = Path("output/stage1")
metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))

ks = [int(k) for k in metrics["by_k"]]
f1 = [metrics["by_k"][str(k)]["macro_f1"] for k in ks]
recall = [metrics["by_k"][str(k)]["macro_recall"] for k in ks]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(ks, f1, marker="o", label="Macro F1")
axes[0].plot(ks, recall, marker="s", label="Macro recall")
axes[0].set_xlabel("Documents returned (k)")
axes[0].set_ylabel("Score")
axes[0].set_title("Fixed-k TF-IDF retrieval on validation")
axes[0].set_xticks(ks)
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.3)
axes[0].legend()

cats = list(metrics["top1_by_category"])
vals = [metrics["top1_by_category"][c]["f1"] for c in cats]
labels = [c.replace("_", " ") for c in cats]
axes[1].barh(labels, vals, color="#3b6ea5")
axes[1].set_xlim(0, 1.02)
axes[1].set_xlabel("Top-1 macro F1")
axes[1].set_title("Top-1 F1 by question category")
for i, v in enumerate(vals):
    axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
axes[1].grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(outdir / "baseline_metrics.png", dpi=180)
print(f"wrote {outdir / 'baseline_metrics.png'}")
