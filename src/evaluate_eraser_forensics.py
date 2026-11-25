from pathlib import Path
import csv
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
ERASER_PREP_DIR = PROJECT_DIR / "outputs" / "eraser_prepared"
REMOVED_DIR = ERASER_PREP_DIR / "tampered_removed"
GT_DIR = ERASER_PREP_DIR / "groundtruth"

OUT_DIR = PROJECT_DIR / "outputs" / "eraser_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FORENSIC FEATURE MAPS
# ==========================================

def compute_ela_map(bgr_img, quality=90, scale=15):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", bgr_img, encode_param)
    if not ok:
        raise RuntimeError("Failed to JPEG-recompress image for ELA.")
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr_img, recompressed).astype(np.float32)
    ela = np.mean(diff, axis=2) * float(scale)
    return np.clip(ela, 0, 255).astype(np.uint8)


def compute_noise_map(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median_bg = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    residual = np.abs(gray - median_bg)
    local_mean = cv2.blur(residual, (7, 7))
    local_sq_mean = cv2.blur(residual ** 2, (7, 7))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))
    norm = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def compute_frequency_maps(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Block-wise DCT energy map (8x8 blocks)
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    dct_energy = np.zeros((ph, pw), dtype=np.float32)
    for y in range(0, ph, 8):
        for x in range(0, pw, 8):
            block = padded[y : y + 8, x : x + 8]
            dct_block = np.abs(cv2.dct(block))
            # High-frequency energy excluding DC component (0,0)
            dct_block[0, 0] = 0.0
            dct_energy[y : y + 8, x : x + 8] = float(np.mean(dct_block))

    dct_map = cv2.normalize(dct_energy[:h, :w], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Local FFT high-frequency magnitude map
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f_shift)).astype(np.float32)
    high_pass = cv2.absdiff(mag, cv2.GaussianBlur(mag, (15, 15), 0))
    fft_map = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    freq_combined = ((dct_map.astype(np.float32) + fft_map.astype(np.float32)) / 2.0).astype(np.uint8)
    return dct_map, fft_map, freq_combined


def fuse_maps(ela_u8, noise_u8, freq_u8):
    f_ela = ela_u8.astype(np.float32) / 255.0
    f_noise = noise_u8.astype(np.float32) / 255.0
    f_freq = freq_u8.astype(np.float32) / 255.0
    fused = (f_ela + f_noise + f_freq) / 3.0
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


# ==========================================
# 3. METRICS EVALUATION
# ==========================================

def evaluate_map(prob_u8, gt_bin):
    _, pred_u8 = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred = (pred_u8 > 0).astype(np.uint8)
    gt = (gt_bin > 0).astype(np.uint8)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)

    y_true_flat = gt.ravel()
    y_score_flat = (prob_u8.astype(np.float32) / 255.0).ravel()
    if len(np.unique(y_true_flat)) > 1:
        # Subsample for fast AUC on large images while preserving distribution
        step = max(1, len(y_true_flat) // 25000)
        auc = float(roc_auc_score(y_true_flat[::step], y_score_flat[::step]))
    else:
        auc = 0.5

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "iou": float(iou),
        "accuracy": float(acc),
        "auc": auc,
    }


# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    removed_files = sorted(REMOVED_DIR.glob("*_removed.png"))
    if not removed_files:
        raise FileNotFoundError(f"No *_removed.png files found in {REMOVED_DIR}")

    methods = ["ela", "noise", "dct", "fft", "frequency", "ela_noise_frequency"]
    agg = {m: [] for m in methods}
    per_image_rows = []

    for rem_path in removed_files:
        base_stem = rem_path.stem.replace("_removed", "")
        gt_path = GT_DIR / f"{base_stem}_gt.png"
        if not gt_path.is_file():
            continue

        bgr = cv2.imread(str(rem_path))
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue

        h, w = gt.shape[:2]
        bgr_r = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        gt_bin = ((gt > 127).astype(np.uint8)) * 255

        ela_m = compute_ela_map(bgr_r)
        noise_m = compute_noise_map(bgr_r)
        dct_m, fft_m, freq_m = compute_frequency_maps(bgr_r)
        comb_m = fuse_maps(ela_m, noise_m, freq_m)

        maps = {
            "ela": ela_m,
            "noise": noise_m,
            "dct": dct_m,
            "fft": fft_m,
            "frequency": freq_m,
            "ela_noise_frequency": comb_m,
        }

        row = {"image": base_stem}
        for m_name, m_arr in maps.items():
            scores = evaluate_map(m_arr, gt_bin)
            agg[m_name].append(scores)
            for k, v in scores.items():
                row[f"{m_name}_{k}"] = v
        per_image_rows.append(row)

    summary_rows = []
    print("\n========== ERASER (OBJECT REMOVAL) EVALUATION SUMMARY ==========")
    for m_name in methods:
        vals = agg[m_name]
        if not vals:
            continue
        mean_row = {"method": m_name, "samples_count": len(vals)}
        for k in ["precision", "recall", "f1_score", "iou", "accuracy", "auc"]:
            mean_row[f"mean_{k}"] = float(np.mean([x[k] for x in vals]))
        summary_rows.append(mean_row)
        print(
            f"{m_name:20s} | P={mean_row['mean_precision']:.4f} | "
            f"R={mean_row['mean_recall']:.4f} | F1={mean_row['mean_f1_score']:.4f} | "
            f"IoU={mean_row['mean_iou']:.4f} | Acc={mean_row['mean_accuracy']:.4f} | "
            f"AUC={mean_row['mean_auc']:.4f}"
        )

    summary_csv = OUT_DIR / "eraser_methods_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    details_csv = OUT_DIR / "eraser_per_image_metrics.csv"
    with details_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(f"\nSaved summary CSV : {summary_csv}")
    print(f"Saved details CSV : {details_csv}")


if __name__ == "__main__":
    main()