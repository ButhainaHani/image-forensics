from pathlib import Path
import csv
import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
PHASE6_DIR = OUTPUT_DIR / "phase6_experiments"
ML_OUT_DIR = OUTPUT_DIR / "ml_model"
ML_OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_FOLDERS = {
    "ela": PHASE6_DIR / "ela",
    "noise": PHASE6_DIR / "noise",
    "dct": PHASE6_DIR / "dct",
    "fft": PHASE6_DIR / "fft",
    "freq": PHASE6_DIR / "frequency",
    "comb": PHASE6_DIR / "ela_noise_frequency",
}

DATASET_DIR = Path(
    r"D:\Amira\Uni\level 3\semester 5\CSE281 Image Processing\CSE281 Project\Datasets\CASIA2"
)
GT_DIR = DATASET_DIR / "CASIA 2 Groundtruth"
AU_DIR = DATASET_DIR / "Au"  # Optional authentic folder if present in CASIA2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TAMPERED_AREA_THRESHOLD = 0.005


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


def find_map_file(folder, stem):
    if not folder.exists():
        return None
    for candidate in (folder / f"{stem}_map.png", folder / f"{stem}.png"):
        if candidate.is_file():
            return candidate
    return None


def extract_map_stats(arr_u8, prefix):
    norm = arr_u8.astype(np.float32) / 255.0
    return {
        f"{prefix}_mean": float(np.mean(norm)),
        f"{prefix}_std": float(np.std(norm)),
        f"{prefix}_max": float(np.max(norm)),
        f"{prefix}_p75": float(np.percentile(norm, 75)),
        f"{prefix}_p90": float(np.percentile(norm, 90)),
        f"{prefix}_p95": float(np.percentile(norm, 95)),
        f"{prefix}_gt05_ratio": float(np.mean(norm > 0.5)),
        f"{prefix}_gt07_ratio": float(np.mean(norm > 0.7)),
    }


def build_dataset(gt_masks):
    comb_dir = MAP_FOLDERS["comb"]
    if not comb_dir.exists():
        raise FileNotFoundError(f"Combined map folder not found: {comb_dir}")

    combined_files = sorted(comb_dir.glob("*_map.png"))
    X_list, y_list, stems = [], [], []
    feature_names = None

    for path in combined_files:
        stem = path.stem[:-4] if path.stem.endswith("_map") else path.stem
        if stem not in gt_masks:
            continue

        gt_img = cv2.imread(str(gt_masks[stem]), cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue

        h, w = gt_img.shape[:2]
        gt_bin = (gt_img > 127).astype(np.uint8)
        gt_ratio = float(np.mean(gt_bin))
        label = 1 if gt_ratio >= TAMPERED_AREA_THRESHOLD else 0

        feat = {"img_aspect": float(w / max(h, 1))}
        valid = True

        for key, folder in MAP_FOLDERS.items():
            m_path = find_map_file(folder, stem)
            if m_path is None:
                if key in ("ela", "comb"):
                    valid = False
                    break
                m_resized = np.zeros((h, w), dtype=np.uint8)
            else:
                m_img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                if m_img is None:
                    if key in ("ela", "comb"):
                        valid = False
                        break
                    m_resized = np.zeros((h, w), dtype=np.uint8)
                else:
                    m_resized = cv2.resize(m_img, (w, h), interpolation=cv2.INTER_LINEAR)

            feat.update(extract_map_stats(m_resized, key))

        if not valid:
            continue

        if feature_names is None:
            feature_names = list(feat.keys())

        X_list.append([feat[col] for col in feature_names])
        y_list.append(label)
        stems.append(stem)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, feature_names, stems


def evaluate_cv(name, clf, X, y, cv):
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict", n_jobs=-1)
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y, y_pred)
    prec_pos = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y, y_prob)

    print(f"\n================ {name} (5-Fold CV) ================")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision (1) : {prec_pos:.4f}")
    print(f"Recall (1)    : {rec_pos:.4f}")
    print(f"F1-Score (1)  : {f1_pos:.4f}")
    print(f"Macro F1      : {f1_macro:.4f}")
    print(f"Weighted F1   : {f1_weighted:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    return {
        "model": name,
        "accuracy": acc,
        "precision_pos": prec_pos,
        "recall_pos": rec_pos,
        "f1_pos": f1_pos,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ==========================================
# 3. MAIN
# ==========================================

def main():
    gt_masks = load_masks(GT_DIR)
    X, y, feature_names, _ = build_dataset(gt_masks)
    print(f"Dataset shape: X={X.shape}, y={y.shape} | Class 1={int(y.sum())}, Class 0={int((y == 0).sum())}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    gb_clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    results = [
        evaluate_cv("Random Forest", rf_clf, X, y, cv),
        evaluate_cv("HistGradientBoosting", gb_clf, X, y, cv),
    ]

    best_row = max(results, key=lambda r: (r["f1_macro"], r["roc_auc"]))
    best_clf = gb_clf if "HistGradientBoosting" in best_row["model"] else rf_clf
    best_clf.fit(X, y)

    joblib.dump(
        {
            "model": best_clf,
            "model_name": best_row["model"],
            "feature_names": feature_names,
            "threshold_area": TAMPERED_AREA_THRESHOLD,
        },
        ML_OUT_DIR / "best_forensics_model_cv.joblib",
    )

    out_csv = ML_OUT_DIR / "cv_models_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBest CV model ({best_row['model']}) saved. Summary CSV: {out_csv}")


if __name__ == "__main__":
    main()