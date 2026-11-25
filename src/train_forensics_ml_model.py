from pathlib import Path
import csv
import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TAMPERED_AREA_THRESHOLD = 0.005  # >= 0.5% tampered pixels in GT -> positive class (1)


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
    """Extract statistical features from a single 2D uint8 forensic map."""
    norm = arr_u8.astype(np.float32) / 255.0
    mean_val = float(np.mean(norm))
    std_val = float(np.std(norm))
    max_val = float(np.max(norm))
    p75 = float(np.percentile(norm, 75))
    p90 = float(np.percentile(norm, 90))
    p95 = float(np.percentile(norm, 95))
    high_ratio_05 = float(np.mean(norm > 0.5))
    high_ratio_07 = float(np.mean(norm > 0.7))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_max": max_val,
        f"{prefix}_p75": p75,
        f"{prefix}_p90": p90,
        f"{prefix}_p95": p95,
        f"{prefix}_gt05_ratio": high_ratio_05,
        f"{prefix}_gt07_ratio": high_ratio_07,
    }


def build_feature_dataset(gt_masks):
    comb_dir = MAP_FOLDERS["comb"]
    if not comb_dir.exists():
        raise FileNotFoundError(f"Combined map folder not found: {comb_dir}")

    combined_files = sorted(comb_dir.glob("*_map.png"))
    rows = []
    feature_dicts = []
    labels = []

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
        missing_required = False

        for key, folder in MAP_FOLDERS.items():
            m_path = find_map_file(folder, stem)
            if m_path is None:
                # Fallback to zeros if an optional sub-map is missing
                if key in ("ela", "comb"):
                    missing_required = True
                    break
                dummy = np.zeros((h, w), dtype=np.uint8)
                feat.update(extract_map_stats(dummy, key))
                continue

            m_img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m_img is None:
                if key in ("ela", "comb"):
                    missing_required = True
                    break
                dummy = np.zeros((h, w), dtype=np.uint8)
                feat.update(extract_map_stats(dummy, key))
                continue

            m_resized = cv2.resize(m_img, (w, h), interpolation=cv2.INTER_LINEAR)
            feat.update(extract_map_stats(m_resized, key))

        if missing_required:
            continue

        feature_dicts.append(feat)
        labels.append(label)
        rows.append({"image": stem, "label": label, **feat})

    if not feature_dicts:
        raise RuntimeError("No samples were built for ML training.")

    feature_names = list(feature_dicts[0].keys())
    X = np.array([[d[col] for col in feature_names] for d in feature_dicts], dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, feature_names, rows


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
    }


# ==========================================
# 3. MAIN TRAINING & COMPARISON
# ==========================================

def main():
    gt_masks = load_masks(GT_DIR)
    print(f"Loaded {len(gt_masks)} ground-truth masks.")

    X, y, feature_names, rows = build_feature_dataset(gt_masks)
    print(f"Built feature matrix: X={X.shape}, y={y.shape}, Positives={int(y.sum())}, Negatives={int((y == 0).sum())}")

    # Save extracted features CSV
    feat_csv = ML_OUT_DIR / "forensics_ml_features.csv"
    with feat_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", *feature_names])
        writer.writeheader()
        writer.writerows(rows)

    stratify_arg = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    gb_model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    gb_model.fit(X_train, y_train)

    results = [
        evaluate_model("Random Forest", rf_model, X_test, y_test),
        evaluate_model("Gradient Boosting (HistGB)", gb_model, X_test, y_test),
    ]

    best_res = max(results, key=lambda r: (r["f1_score"], r["roc_auc"]))
    best_model = gb_model if best_res["model"].startswith("Gradient") else rf_model

    model_bundle = {
        "model": best_model,
        "model_name": best_res["model"],
        "feature_names": feature_names,
    }
    model_path = ML_OUT_DIR / "best_forensics_model.joblib"
    joblib.dump(model_bundle, model_path)

    comp_csv = ML_OUT_DIR / "ml_models_comparison.csv"
    with comp_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBest model ({best_res['model']}) saved to: {model_path}")
    print(f"Features saved to: {feat_csv}")
    print(f"Comparison saved to: {comp_csv}")


if __name__ == "__main__":
    main()