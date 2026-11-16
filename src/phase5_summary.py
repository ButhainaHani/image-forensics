
import csv
from pathlib import Path
from collections import defaultdict

csv_path = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "phase5_experiments"
    / "experiment_comparison.csv"
)

metrics = ["precision", "recall", "f1_score", "iou"]
totals = defaultdict(lambda: {metric: 0.0 for metric in metrics})
counts = defaultdict(int)

with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        experiment = row["experiment"]

        for metric in metrics:
            value = row.get(metric, "")
            if value:
                totals[experiment][metric] += float(value)

        counts[experiment] += 1

summary = []

for experiment, values in totals.items():
    count = counts[experiment]
    averages = {
        metric: values[metric] / count
        for metric in metrics
    }
    summary.append((experiment, averages))

summary.sort(key=lambda item: item[1]["f1_score"], reverse=True)

print("\n===== PHASE 5 AVERAGE RESULTS =====")
print(f"{'Experiment':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")

for experiment, values in summary:
    print(
        f"{experiment:<28} "
        f"{values['precision']:>10.4f} "
        f"{values['recall']:>10.4f} "
        f"{values['f1_score']:>10.4f} "
        f"{values['iou']:>10.4f}"
    )

output_path = csv_path.parent / "phase5_average_summary.csv"

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["experiment"] + metrics)

    for experiment, values in summary:
        writer.writerow([experiment] + [values[m] for m in metrics])

print(f"\nSummary saved to: {output_path}")
