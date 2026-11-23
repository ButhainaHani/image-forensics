from pathlib import Path
import csv
import cv2
import numpy as np

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
PHASE6_DIR = OUTPUT_DIR / "phase6_experiments"

COMBINED_MAP_DIR = PHASE6_DIR / "ela_noise_frequency"
ELA_MAP_DIR = PHASE6_DIR / "ela"

DATASET_DIR = Path(
    r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\CASIA2"
)
TP_DIR = DATASET_DIR / "Tp"
GT_DIR = DATASET_DIR / "CASIA 2 Groundtruth"

ANALYSIS_DIR = PHASE6_DIR / "fp_fn_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TOP_K = 5


# ==========================================
# 2. HELPERS
# ==========================================

def load_masks(gt_folder):
    masks = {}
    for path in gt_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem.lower().endswith("_gt"):
                stem = path.stem[:-3]
                masks.setdefault(stem, path)
    return masks


def find_original_image(stem, tp_folder):
    for ext in IMAGE_EXTENSIONS:
        for candidate in (tp_folder / f"{stem}{ext}", tp_folder / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                img = cv2.imread(str(candidate))
                if img is not None:
                    return img
    return None


def compute_confusion_counts(map_u8, gt_binary):
    _, pred_u8 = cv2.threshold(map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred = pred_u8 > 0
    gt = gt_binary > 0

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    total = max(tp + fp + fn + tn, 1)

    eps = 1e-8
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float((2.0 * precision * recall) / (precision + recall + eps))
    iou = float(tp / (tp + fp + fn + eps))
    fp_ratio = float(fp / total)
    fn_ratio = float(fn / max(tp + fn, 1))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_ratio_of_image": fp_ratio,
        "fn_ratio_of_gt": fn_ratio,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
    }, pred_u8


def make_fp_fn_panel(orig_bgr, gt_binary, combined_u8, pred_u8, title_prefix):
    """Build 2x2 visual panel: Original | GT | Combined Map | FP (Red) & FN (Blue)."""
    h, w = gt_binary.shape[:2]
    size = (w, h)

    orig_r = cv2.resize(orig_bgr, size, interpolation=cv2.INTER_LINEAR)
    gt_r = cv2.resize(gt_binary, size, interpolation=cv2.INTER_NEAREST) > 0
    comb_r = cv2.resize(combined_u8, size, interpolation=cv2.INTER_LINEAR)
    pred_r = cv2.resize(pred_u8, size, interpolation=cv2.INTER_NEAREST) > 0

    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_r] = (255, 255, 255)

    comb_vis = cv2.applyColorMap(comb_r, cv2.COLORMAP_JET)

    # Error overlay on original: FP = Red (BGR: 0, 0, 255), FN = Cyan/Blue (BGR: 255, 180, 0)
    err_vis = orig_r.copy()
    fp_mask = np.logical_and(pred_r, ~gt_r)
    fn_mask = np.logical_and(~pred_r, gt_r)
    err_vis[fp_mask] = (0, 0, 255)
    err_vis[fn_mask] = (255, 180, 0)

    def tag(img, text):
        bar = np.zeros((34, w, 3), dtype=np.uint8)
        cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    top = np.hstack([tag(orig_r, f"{title_prefix} - Original"), tag(gt_vis, "Ground Truth (White)")])
    bot = np.hstack([tag(comb_vis, "Combined Map"), tag(err_vis, "FP=Red, FN=Blue")])
    return np.vstack([top, bot])


# ==========================================
# 3. MAIN FP / FN ANALYSIS
# ==========================================

def main():
    if not COMBINED_MAP_DIR.exists():
        raise FileNotFoundError(f"Combined map folder not found: {COMBINED_MAP_DIR}")

    gt_masks = load_masks(GT_DIR)
    combined_files = sorted(COMBINED_MAP_DIR.glob("*_map.png"))

    records = []
    cache_maps = {}

    for path in combined_files:
        stem = path.stem[:-4]  # strip "_map"
        if stem not in gt_masks:
            continue

        gt_img = cv2.imread(str(gt_masks[stem]), cv2.IMREAD_GRAYSCALE)
        comb_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gt_img is None or comb_img is None:
            continue

        h, w = gt_img.shape[:2]
        gt_binary = ((gt_img > 127).astype(np.uint8)) * 255
        comb_resized = cv2.resize(comb_img, (w, h), interpolation=cv2.INTER_LINEAR)

        counts, pred_u8 = compute_confusion_counts(comb_resized, gt_binary)
        counts["image"] = stem
        records.append(counts)
        cache_maps[stem] = (gt_binary, comb_resized, pred_u8)

    if not records:
        raise RuntimeError("No valid records found for FP/FN analysis.")

    # Save full FP/FN table
    csv_path = ANALYSIS_DIR / "phase6_fp_fn_metrics.csv"
    fieldnames = [
        "image",
        "tp",
        "fp",
        "fn",
        "tn",
        "fp_ratio_of_image",
        "fn_ratio_of_gt",
        "precision",
        "recall",
        "f1_score",
        "iou",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Aggregate dataset-level totals
    total_tp = sum(r["tp"] for r in records)
    total_fp = sum(r["fp"] for r in records)
    total_fn = sum(r["fn"] for r in records)
    total_tn = sum(r["tn"] for r in records)

    top_fp = sorted(records, key=lambda r: r["fp_ratio_of_image"], reverse=True)[:TOP_K]
    top_fn = sorted(records, key=lambda r: r["fn_ratio_of_gt"], reverse=True)[:TOP_K]

    print("\n========== OVERALL PIXEL CONFUSION (ELA + Noise + Frequency) ==========")
    print(f"Images analyzed: {len(records)}")
    print(f"Total TP: {total_tp:,} | Total FP: {total_fp:,} | Total FN: {total_fn:,} | Total TN: {total_tn:,}")

    print(f"\n========== TOP {TOP_K} FALSE POSITIVE CASES (BY FP RATIO) ==========")
    for rank, r in enumerate(top_fp, 1):
        print(
            f"{rank}. {r['image']} | FP={r['fp']:,} ({r['fp_ratio_of_image']:.4f}) | "
            f"F1={r['f1_score']:.4f} | IoU={r['iou']:.4f}"
        )

    print(f"\n========== TOP {TOP_K} FALSE NEGATIVE CASES (BY FN RATIO ON GT) ==========")
    for rank, r in enumerate(top_fn, 1):
        print(
            f"{rank}. {r['image']} | FN={r['fn']:,} ({r['fn_ratio_of_gt']:.4f}) | "
            f"F1={r['f1_score']:.4f} | IoU={r['iou']:.4f}"
        )

    # Save visual figures for top FP and top FN cases
    for tag_name, subset in [("HIGH_FP", top_fp), ("HIGH_FN", top_fn)]:
        for rank, r in enumerate(subset, 1):
            stem = r["image"]
            orig = find_original_image(stem, TP_DIR)
            if orig is None or stem not in cache_maps:
                continue
            gt_binary, comb_resized, pred_u8 = cache_maps[stem]
            panel = make_fp_fn_panel(orig, gt_binary, comb_resized, pred_u8, f"{tag_name} #{rank}")
            out_img = ANALYSIS_DIR / f"{tag_name.lower()}_{rank}_{stem}.png"
            cv2.imwrite(str(out_img), panel)

    print(f"\nFP/FN CSV saved to: {csv_path}")
    print(f"Visual error panels saved in: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()