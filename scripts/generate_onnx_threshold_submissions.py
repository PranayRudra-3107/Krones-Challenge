import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_data_root() -> Path:
    candidates = [
        ROOT / "data",
        ROOT / "data" / "1st-krones-vision-ai-challenge",
        ROOT,
    ]
    for base in candidates:
        if (base / "sample_submission.csv").exists() and (base / "test_images").exists():
            return base
    raise FileNotFoundError("Could not find sample_submission.csv and test_images/.")


def load_metadata(model_dir: Path) -> dict:
    path = model_dir / "model_metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_roi_map(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    id_to_name = {int(image["id"]): image["file_name"] for image in payload.get("images", [])}
    roi_map = {}
    for ann in payload.get("annotations", []):
        name = id_to_name.get(int(ann["image_id"]))
        if name and "bbox" in ann:
            roi_map[name] = ann["bbox"]
            roi_map[Path(name).stem] = ann["bbox"]
    return roi_map


def canonical_bottle_type(value: str) -> int:
    text = str(value).lower()
    if "vichy" in text:
        return 0
    if "euro" in text:
        return 1
    if "nrw" in text:
        return 2
    return -1


def load_btype_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"].eq("test")].copy()
    df["btype_id"] = df["bottle_type"].map(canonical_bottle_type)
    out = {}
    for row in df.itertuples(index=False):
        image_id = str(row.image_id)
        out[image_id] = int(row.btype_id)
        out[Path(image_id).stem] = int(row.btype_id)
    return out


def crop_roi(image: np.ndarray, bbox, padding: int = 8) -> np.ndarray:
    if bbox is None:
        return image
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(iw, int(x + w) + padding)
    y2 = min(ih, int(y + h) + padding)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def read_image(test_img_dir: Path, image_id: str) -> np.ndarray:
    path = test_img_dir / image_id
    if not path.exists():
        stem = Path(image_id).stem
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = test_img_dir / f"{stem}{ext}"
            if candidate.exists():
                path = candidate
                break
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read test image: {image_id}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(
    test_img_dir: Path,
    image_id: str,
    roi_map: dict[str, list[float]],
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    image = read_image(test_img_dir, image_id)
    bbox = roi_map.get(image_id) or roi_map.get(Path(image_id).stem)
    image = crop_roi(image, bbox)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_inference(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict]:
    data_root = Path(args.data_root) if args.data_root else find_data_root()
    model_path = Path(args.model_path)
    model_dir = model_path.parent
    metadata = load_metadata(model_dir)

    img_size = int(metadata.get("img_size", args.img_size))
    batch_size = int(args.batch_size or metadata.get("batch_size", 32))
    mean = np.array(metadata.get("mean", DEFAULT_MEAN.tolist()), dtype=np.float32)
    std = np.array(metadata.get("std", DEFAULT_STD.tolist()), dtype=np.float32)
    output_activation = metadata.get("output_activation", "sigmoid")

    sample = pd.read_csv(data_root / "sample_submission.csv")
    roi_map = load_roi_map(data_root / "test_annotations_roi_only.json")
    btype_map = load_btype_map(data_root / "bottletypes.csv")
    test_img_dir = data_root / "test_images"

    providers = [
        provider
        for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in ort.get_available_providers()
    ] or ort.get_available_providers()
    session = ort.InferenceSession(str(model_path), providers=providers)
    session_inputs = session.get_inputs()
    image_input_name = session_inputs[0].name
    btype_input_name = session_inputs[1].name if len(session_inputs) > 1 else None
    output_name = session.get_outputs()[0].name

    image_ids = sample["image_id"].astype(str).tolist()
    probs = np.zeros(len(image_ids), dtype=np.float32)
    started = time.time()

    for start in range(0, len(image_ids), batch_size):
        end = min(start + batch_size, len(image_ids))
        ids = image_ids[start:end]
        batch = np.stack(
            [preprocess(test_img_dir, image_id, roi_map, img_size, mean, std) for image_id in ids],
            axis=0,
        )
        feed = {image_input_name: batch}
        if btype_input_name is not None:
            feed[btype_input_name] = np.array(
                [btype_map.get(image_id, btype_map.get(Path(image_id).stem, -1)) for image_id in ids],
                dtype=np.int64,
            )
        output = session.run([output_name], feed)[0]
        output = np.asarray(output).reshape(len(ids), -1)[:, 0]
        if output_activation == "sigmoid":
            output = sigmoid(output)
        probs[start:end] = output.astype(np.float32)
        if start == 0 or end == len(image_ids) or (start // batch_size) % 10 == 0:
            print(f"{end}/{len(image_ids)} images | elapsed {time.time() - started:.1f}s")

    return sample, probs, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thresholded submissions from an ONNX model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample, probs, metadata = run_inference(args)

    np.save(args.out_dir / "test_probs.npy", probs)
    prob_csv = sample[["image_id"]].copy()
    prob_csv["probability"] = probs
    prob_csv.to_csv(args.out_dir / "test_probs.csv", index=False)

    summary = {
        "model_path": str(args.model_path),
        "metadata_threshold": metadata.get("threshold"),
        "mean_probability": float(probs.mean()),
        "thresholds": {},
    }
    for threshold in args.thresholds:
        preds = (probs >= threshold).astype(int)
        sub = sample[["image_id"]].copy()
        sub["target"] = preds
        out_path = args.out_dir / f"submission_t{threshold:.3f}.csv"
        sub.to_csv(out_path, index=False)
        summary["thresholds"][f"{threshold:.3f}"] = {
            "positive_predictions": int(preds.sum()),
            "path": str(out_path),
        }
        print(f"threshold={threshold:.3f} positives={int(preds.sum())} wrote={out_path}")

    (args.out_dir / "threshold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
