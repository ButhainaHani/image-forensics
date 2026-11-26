import importlib
import inspect
import sys
import uuid
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from backend.config import RESULTS_DIR, SRC_DIR, UPLOAD_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _to_rel_posix(path: Path) -> str:
    return path.resolve().relative_to(RESULTS_DIR.parent.parent.resolve()).as_posix()


def _load_module(module_name: str):
    return importlib.import_module(module_name)


def _find_callable(module, candidates: list[str]):
    for name in candidates:
        attr = getattr(module, name, None)
        if callable(attr):
            return attr
    available = [
        k for k, v in vars(module).items()
        if callable(v) and not k.startswith("_")
    ]
    raise AttributeError(
        f"Could not find any of {candidates} in '{module.__name__}'. "
        f"Available callables: {available}"
    )


def _call_analysis_fn(fn, img_path: Path, out_dir: Path):
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs = {}

    if "output_dir" in params:
        kwargs["output_dir"] = str(out_dir)
    elif "out_dir" in params:
        kwargs["out_dir"] = str(out_dir)

    if "save_dir" in params:
        kwargs["save_dir"] = str(out_dir)

    try:
        return fn(str(img_path), **kwargs)
    except TypeError:
        try:
            return fn(img_path, **kwargs)
        except TypeError:
            bgr = cv2.imread(str(img_path))
            if bgr is not None:
                return fn(bgr, **kwargs)
            raise


def _save_pil_image(pil_img: Image.Image, out_path: Path) -> str:
    pil_img.convert("RGB").save(out_path)
    return _to_rel_posix(out_path)


def _save_array_as_image(arr: np.ndarray, out_path: Path) -> str:
    if arr.ndim == 2:
        norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(out_path), norm)
    elif arr.ndim == 3:
        if arr.dtype != np.uint8:
            norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            norm = arr
        cv2.imwrite(str(out_path), norm)
    else:
        raise ValueError("Unsupported output array shape for image saving.")
    return _to_rel_posix(out_path)


def _extract_output_from_result(result, out_dir: Path, default_filename: str) -> tuple[str | None, dict]:
    output_rel = None
    extra_info = {}

    if isinstance(result, Image.Image):
        output_rel = _save_pil_image(result, out_dir / default_filename)
    elif isinstance(result, np.ndarray):
        if result.ndim in (2, 3) and result.size >= 16:
            output_rel = _save_array_as_image(result, out_dir / default_filename)
        else:
            extra_info["features"] = result.tolist()
    elif isinstance(result, tuple):
        if len(result) >= 2 and isinstance(result[1], Image.Image):
            output_rel = _save_pil_image(result[1], out_dir / default_filename)
        if len(result) >= 3 and isinstance(result[2], (int, float)):
            extra_info["mean_error"] = float(result[2])

        for idx, item in enumerate(result):
            if isinstance(item, Image.Image) and output_rel is None:
                output_rel = _save_pil_image(item, out_dir / default_filename)
            elif isinstance(item, np.ndarray):
                if item.ndim in (2, 3) and item.size >= 16 and output_rel is None:
                    output_rel = _save_array_as_image(item, out_dir / default_filename)
                else:
                    extra_info[f"array_{idx}"] = (
                        item.tolist()
                        if item.size <= 64
                        else {"shape": list(item.shape), "mean": float(np.mean(item))}
                    )
            elif isinstance(item, (str, Path)) and Path(item).exists() and output_rel is None:
                output_rel = _to_rel_posix(Path(item))
            elif isinstance(item, dict):
                extra_info.update(item)
            elif isinstance(item, (int, float)) and "mean_error" not in extra_info:
                extra_info[f"metric_{idx}"] = float(item)
    elif isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, Image.Image) and output_rel is None:
                output_rel = _save_pil_image(v, out_dir / default_filename)
            elif isinstance(v, np.ndarray):
                if v.ndim in (2, 3) and v.size >= 16 and output_rel is None:
                    output_rel = _save_array_as_image(v, out_dir / default_filename)
                else:
                    extra_info[k] = (
                        v.tolist()
                        if v.size <= 64
                        else {"shape": list(v.shape), "mean": float(np.mean(v))}
                    )
            elif isinstance(v, (str, Path)) and Path(v).exists() and output_rel is None:
                output_rel = _to_rel_posix(Path(v))
            else:
                extra_info[k] = v
    elif isinstance(result, (str, Path)) and Path(result).exists():
        output_rel = _to_rel_posix(Path(result))

    return output_rel, extra_info


def _resolve_input_path(file_id: str) -> Path:
    img_path = UPLOAD_DIR / file_id
    if not img_path.exists():
        raise FileNotFoundError(f"Uploaded file '{file_id}' not found.")
    return img_path


def run_ela_analysis(file_id: str) -> dict:
    img_path = _resolve_input_path(file_id)
    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_module("ela_analysis")
    fn = _find_callable(mod, ["perform_ela", "compute_ela", "run_ela", "ela_analysis", "generate_ela"])
    result = _call_analysis_fn(fn, img_path, out_dir)
    output_rel, extra_info = _extract_output_from_result(result, out_dir, "ela.png")

    return {
        "module": "ela",
        "file_id": file_id,
        "job_id": job_id,
        "output_image": output_rel,
        "details": extra_info or None,
    }


def run_noise_analysis(file_id: str) -> dict:
    img_path = _resolve_input_path(file_id)
    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError("Could not read uploaded image for noise analysis.")

    mod = _load_module("noise_analysis")
    residual_fn = _find_callable(mod, ["extract_noise_residual"])
    map_fn = _find_callable(mod, ["compute_local_noise_map"])
    heatmap_fn = _find_callable(mod, ["generate_noise_heatmap"])

    residual = residual_fn(image)
    noise_map = map_fn(residual)
    heatmap = heatmap_fn(noise_map)

    noise_map_rel = _save_array_as_image(noise_map, out_dir / "noise_map.png")
    heatmap_rel = _save_array_as_image(heatmap, out_dir / "noise_heatmap.png")

    return {
        "module": "noise",
        "file_id": file_id,
        "job_id": job_id,
        "output_image": heatmap_rel,
        "details": {
            "noise_map": noise_map_rel,
            "mean_noise_map": float(np.mean(noise_map)),
        },
    }


def run_frequency_analysis(file_id: str) -> dict:
    img_path = _resolve_input_path(file_id)
    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError("Could not read uploaded image for frequency analysis.")

    mod = _load_module("frequency_analysis")
    dct_fn = _find_callable(mod, ["perform_dct_analysis"])
    fft_fn = _find_callable(mod, ["perform_fft_analysis"])
    feat_fn = _find_callable(mod, ["extract_frequency_features"])

    dct_map, dct_transform = dct_fn(image)
    fft_map, fft_transform = fft_fn(image)

    dct_map_rel = _save_array_as_image(dct_map, out_dir / "dct_map.png")
    fft_map_rel = _save_array_as_image(fft_map, out_dir / "fft_map.png")

    features = {}
    features.update(feat_fn(dct_transform, "dct"))
    features.update(feat_fn(fft_transform, "fft"))

    return {
        "module": "frequency",
        "file_id": file_id,
        "job_id": job_id,
        "output_image": fft_map_rel,
        "details": {
            "dct_map": dct_map_rel,
            "fft_map": fft_map_rel,
            "features": features,
        },
    }


def _load_gray_from_rel(rel_path: str | None) -> np.ndarray | None:
    if not rel_path:
        return None
    full_path = RESULTS_DIR.parent.parent / rel_path
    if not full_path.exists():
        return None
    return cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)


def run_combined_analysis(file_id: str) -> dict:
    img_path = _resolve_input_path(file_id)
    original = cv2.imread(str(img_path))
    if original is None:
        raise ValueError("Could not read uploaded image for combined analysis.")

    ela_res = run_ela_analysis(file_id)
    noise_res = run_noise_analysis(file_id)
    freq_res = run_frequency_analysis(file_id)

    job_id = uuid.uuid4().hex[:10]
    out_dir = RESULTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = original.shape[:2]
    size = (width, height)

    combine_mod = _load_module("feature_combination")
    normalize_fn = _find_callable(combine_mod, ["normalize_map"])
    spatial_freq_fn = _find_callable(combine_mod, ["create_spatial_frequency_map"])
    predict_fn = _find_callable(combine_mod, ["make_prediction"])

    maps_to_fuse = []

    ela_gray = _load_gray_from_rel(ela_res.get("output_image"))
    if ela_gray is not None:
        maps_to_fuse.append(normalize_fn(ela_gray, size))

    noise_details = noise_res.get("details") or {}
    noise_gray = _load_gray_from_rel(noise_details.get("noise_map") or noise_res.get("output_image"))
    if noise_gray is not None:
        maps_to_fuse.append(normalize_fn(noise_gray, size))

    freq_norm = spatial_freq_fn(original, size)
    maps_to_fuse.append(freq_norm)

    combined_map, binary_mask = predict_fn(maps_to_fuse)
    combined_rel = _save_array_as_image(combined_map, out_dir / "combined_heatmap.png")
    binary_rel = _save_array_as_image(binary_mask, out_dir / "combined_binary.png")

    suspicious_ratio = float(np.mean(binary_mask > 0))

    return {
        "file_id": file_id,
        "job_id": job_id,
        "ela": ela_res,
        "noise": noise_res,
        "frequency": freq_res,
        "combined_heatmap": combined_rel,
        "details": {
            "binary_mask": binary_rel,
            "suspicious_area_ratio": suspicious_ratio,
            "fused_modules_count": len(maps_to_fuse),
        },
    }