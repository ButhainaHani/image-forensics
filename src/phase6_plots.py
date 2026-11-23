import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
PHASE6_DIR = PROJECT_DIR / "outputs" / "phase6_experiments"
SUMMARY_CSV = PHASE6_DIR / "phase6_comparison_summary.csv"

# Read summary CSV using built-in csv module
rows_by_exp = {}
with SUMMARY_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows_by_exp[row["experiment"]] = {
            "mean_precision": float(row["mean_precision"]),
            "mean_recall": float(row["mean_recall"]),
            "mean_f1_score": float(row["mean_f1_score"]),
            "mean_iou": float(row["mean_iou"]),
        }

# Core 5 experiments required by Proposal Phase 6
core_order = [
    "ELA",
    "Noise",
    "Frequency",
    "ELA + Noise",
    "ELA + Noise + Frequency",
]

f1_scores = [rows_by_exp[exp]["mean_f1_score"] for exp in core_order]
iou_scores = [rows_by_exp[exp]["mean_iou"] for exp in core_order]
precisions = [rows_by_exp[exp]["mean_precision"] for exp in core_order]
recalls = [rows_by_exp[exp]["mean_recall"] for exp in core_order]

x = list(range(len(core_order)))

# 1. F1-Score and IoU comparison chart
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.36

ax.bar([i - width / 2 for i in x], f1_scores, width=width, label="Mean F1-Score")
ax.bar([i + width / 2 for i in x], iou_scores, width=width, label="Mean IoU")

ax.set_xticks(x)
ax.set_xticklabels(core_order, rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_title("Phase 6 Core Experiments: Mean F1-Score vs. Mean IoU")
ax.set_ylim(0, max(max(f1_scores), max(iou_scores)) * 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

plt.tight_layout()
out1 = PHASE6_DIR / "phase6_f1_iou_comparison.png"
plt.savefig(out1, dpi=200)
plt.close()

# 2. All four metrics comparison across core experiments
fig, ax = plt.subplots(figsize=(10, 5.5))
all_series = [
    (precisions, "Precision"),
    (recalls, "Recall"),
    (f1_scores, "F1-Score"),
    (iou_scores, "IoU"),
]
bar_w = 0.18

for idx, (values, lbl) in enumerate(all_series):
    offsets = [i + (idx - 1.5) * bar_w for i in x]
    ax.bar(offsets, values, width=bar_w, label=lbl)

ax.set_xticks(x)
ax.set_xticklabels(core_order, rotation=15, ha="right")
ax.set_ylabel("Mean Value")
ax.set_title("Phase 6 Core Experiments: Precision, Recall, F1-Score, and IoU")
ax.set_ylim(0, max(recalls) * 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()

plt.tight_layout()
out2 = PHASE6_DIR / "phase6_all_metrics.png"
plt.savefig(out2, dpi=200)
plt.close()

print("Saved plots:", out1, "and", out2)