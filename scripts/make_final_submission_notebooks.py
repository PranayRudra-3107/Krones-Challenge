import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "final_submission"
MODEL_DIR = FINAL_DIR / "model"


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


FINAL_EVALUATION_CELLS = [
    markdown_cell(
        """
# Krones Final Evaluation Notebook

This notebook is the final evaluation target. It is intentionally standalone:
it loads the attached ONNX model, reads the competition test data, runs ROI-based
inference, and writes `submission.csv`.

It should not depend on the local project scripts, `.venv`, training outputs, or
internet downloads.
"""
    ),
    markdown_cell(
        """
## Expected Kaggle Inputs

Attach these inputs to the Kaggle evaluation notebook:

- competition data containing `sample_submission.csv`, `test_images/`, and
  `test_annotations_roi_only.json`
- `bottletypes.csv` from the competition data, used as the second ONNX input
- model dataset containing `final_effnetv2_s.onnx`
- optional `model_metadata.json` next to the ONNX file

The metadata file stores preprocessing and threshold values. The notebook has
safe defaults, but the final trained model should provide final calibrated
values.
"""
    ),
    code_cell(
        """
from pathlib import Path
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "onnxruntime is required for the final ONNX evaluation notebook. "
        "Use the competition starter environment or attach an offline wheel if needed."
    ) from exc


IMG_SIZE = 256
BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.5
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_FILENAMES = ["final_effnetv2_s.onnx", "model.onnx"]
N_BOTTLE_TYPES = 3

print("onnxruntime:", ort.__version__)
print("available providers:", ort.get_available_providers())
"""
    ),
    code_cell(
        """
def find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/1st-krones-vision-ai-challenge"),
        Path("/kaggle/input"),
        Path("data"),
        Path.cwd(),
    ]
    for base in candidates:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            current_path = Path(current)
            if "sample_submission.csv" in files and (
                (current_path / "test_images").exists()
                or "test_annotations_roi_only.json" in files
            ):
                return current_path
    raise FileNotFoundError(
        "Could not find competition data root with sample_submission.csv."
    )


def find_model_path() -> Path:
    candidates = [
        Path("/kaggle/input"),
        Path("model"),
        Path("final_submission/model"),
        Path("outputs/final_effnetv2_s"),
        Path.cwd(),
    ]
    for base in candidates:
        if not base.exists():
            continue
        for name in MODEL_FILENAMES:
            direct = base / name
            if direct.exists():
                return direct
        for current, dirs, files in os.walk(base):
            for name in MODEL_FILENAMES:
                if name in files:
                    return Path(current) / name
    raise FileNotFoundError(
        "Could not find final_effnetv2_s.onnx. Attach the trained model as a Kaggle input."
    )


def find_bottletypes_path(data_root: Path) -> Path:
    direct = data_root / "bottletypes.csv"
    if direct.exists():
        return direct
    for base in [data_root, Path("/kaggle/input")]:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            if "bottletypes.csv" in files:
                return Path(current) / "bottletypes.csv"
    return direct


DATA_ROOT = find_data_root()
MODEL_PATH = find_model_path()
MODEL_DIR = MODEL_PATH.parent

SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
TEST_IMG_DIR = DATA_ROOT / "test_images"
ROI_PATH = DATA_ROOT / "test_annotations_roi_only.json"
BOTTLE_PATH = find_bottletypes_path(DATA_ROOT)

print("DATA_ROOT:", DATA_ROOT)
print("MODEL_PATH:", MODEL_PATH)
print("TEST_IMG_DIR:", TEST_IMG_DIR)
print("ROI_PATH:", ROI_PATH if ROI_PATH.exists() else "not found; full-image fallback")
print("BOTTLE_PATH:", BOTTLE_PATH if BOTTLE_PATH.exists() else "not found; unknown-type fallback")
"""
    ),
    code_cell(
        """
def load_metadata(model_dir: Path) -> dict:
    path = model_dir / "model_metadata.json"
    if path.exists():
        with path.open() as handle:
            return json.load(handle)
    return {}


metadata = load_metadata(MODEL_DIR)
IMG_SIZE = int(metadata.get("img_size", IMG_SIZE))
BATCH_SIZE = int(metadata.get("batch_size", BATCH_SIZE))
THRESHOLD = float(metadata.get("threshold", DEFAULT_THRESHOLD))
OUTPUT_ACTIVATION = metadata.get("output_activation", "sigmoid")
NORM_MEAN = np.array(metadata.get("mean", DEFAULT_MEAN.tolist()), dtype=np.float32)
NORM_STD = np.array(metadata.get("std", DEFAULT_STD.tolist()), dtype=np.float32)
INPUT_NAMES = metadata.get("input_names", ["image"])

print("metadata:", metadata or "defaults")
print("IMG_SIZE:", IMG_SIZE)
print("BATCH_SIZE:", BATCH_SIZE)
print("THRESHOLD:", THRESHOLD)
print("OUTPUT_ACTIVATION:", OUTPUT_ACTIVATION)
print("INPUT_NAMES:", INPUT_NAMES)
"""
    ),
    code_cell(
        """
def load_roi_map(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open() as handle:
        coco = json.load(handle)
    id_to_name = {
        int(image["id"]): image["file_name"]
        for image in coco.get("images", [])
    }
    roi_map = {}
    for ann in coco.get("annotations", []):
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


def read_image(image_id: str) -> np.ndarray:
    path = TEST_IMG_DIR / image_id
    if not path.exists():
        stem = Path(image_id).stem
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = TEST_IMG_DIR / f"{stem}{ext}"
            if candidate.exists():
                path = candidate
                break
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read test image: {image_id}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def preprocess(image_id: str, roi_map: dict[str, list[float]]) -> np.ndarray:
    image = read_image(image_id)
    cropped = crop_roi(image, roi_map.get(image_id) or roi_map.get(Path(image_id).stem))
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def btype_for(image_id: str, btype_map: dict[str, int]) -> int:
    return int(btype_map.get(image_id, btype_map.get(Path(image_id).stem, -1)))


roi_map = load_roi_map(ROI_PATH)
btype_map = load_btype_map(BOTTLE_PATH)
sample = pd.read_csv(SAMPLE_PATH)

print("test rows:", len(sample))
print("ROI entries:", len(roi_map))
print("Bottle type entries:", len(btype_map))
print(sample.head())
"""
    ),
    code_cell(
        """
providers = [
    provider
    for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider in ort.get_available_providers()
]
if not providers:
    providers = ort.get_available_providers()

session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
session_inputs = session.get_inputs()
image_input_name = session_inputs[0].name
btype_input_name = session_inputs[1].name if len(session_inputs) > 1 else None
output_name = session.get_outputs()[0].name

print("using providers:", session.get_providers())
for item in session_inputs:
    print("input:", item.name, item.shape, item.type)
print("output:", output_name, session.get_outputs()[0].shape)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_batch(batch_ids: list[str]) -> np.ndarray:
    batch = np.stack([preprocess(image_id, roi_map) for image_id in batch_ids], axis=0)
    feed = {image_input_name: batch}
    if btype_input_name is not None:
        feed[btype_input_name] = np.array(
            [btype_for(image_id, btype_map) for image_id in batch_ids],
            dtype=np.int64,
        )
    output = session.run([output_name], feed)[0]
    output = np.asarray(output).reshape(len(batch_ids), -1)[:, 0]
    if OUTPUT_ACTIVATION == "sigmoid":
        output = sigmoid(output)
    elif OUTPUT_ACTIVATION in {"identity", "probability", "none"}:
        output = output.astype(np.float32)
    else:
        raise ValueError(f"Unknown output_activation: {OUTPUT_ACTIVATION}")
    return output.astype(np.float32)
"""
    ),
    code_cell(
        """
started = time.time()
image_ids = sample["image_id"].astype(str).tolist()
probs = np.zeros(len(image_ids), dtype=np.float32)

for start in range(0, len(image_ids), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(image_ids))
    probs[start:end] = run_batch(image_ids[start:end])
    if start == 0 or end == len(image_ids) or (start // BATCH_SIZE) % 10 == 0:
        elapsed = time.time() - started
        print(f"{end}/{len(image_ids)} images | elapsed {elapsed:.1f}s")

preds = (probs >= THRESHOLD).astype(int)
submission = sample[["image_id"]].copy()
submission["target"] = preds
submission.to_csv("submission.csv", index=False)

elapsed = time.time() - started
print("finished inference")
print("elapsed_seconds:", round(elapsed, 2))
print("mean_probability:", float(probs.mean()))
print("positive_predictions:", int(preds.sum()), "/", len(preds))
print("wrote: submission.csv")
submission.head()
"""
    ),
]


FINAL_TRAINING_CELLS = [
    markdown_cell(
        """
# Krones Final Training Notebook

Run this after the ConvNeXt Small and MaxViT Tiny teacher notebooks have exported
their OOF/test predictions. It builds the final EfficientNetV2-S student from
COCO-derived labels, ROI crops, bottle-type metadata, robust noisy-label
weights, and teacher soft labels, then exports `final_effnetv2_s.onnx`.
"""
    ),
    markdown_cell(
        """
## Final Strategy

- Use official COCO annotations as the source of truth for training labels.
- Train two diverse teacher models: ConvNeXt Small and MaxViT Tiny.
- Blend teacher OOF probabilities into soft supervision.
- Downweight samples where teachers strongly disagree with the COCO label.
- Train one EfficientNetV2-S student with ROI crop plus bottle-type input.
- Export only the EfficientNetV2-S student to ONNX for final evaluation.

This balances the final score components: hidden-test F1, inference efficiency,
and technical insight.
"""
    ),
    code_cell(
        """
from pathlib import Path
import json

STRATEGY_PATH = Path("../configs/final_strategy.json")
if not STRATEGY_PATH.exists():
    STRATEGY_PATH = Path("configs/final_strategy.json")

if STRATEGY_PATH.exists():
    strategy = json.loads(STRATEGY_PATH.read_text())
    print(json.dumps(strategy, indent=2))
else:
    print("Final strategy file not found. Include the strategy summary here before sharing.")
"""
    ),
    code_cell(
        """
import subprocess
import sys
from pathlib import Path


def find_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "scripts").exists() and (candidate / "data").exists():
            return candidate
    return start


ROOT = find_root()
print("ROOT:", ROOT)
print("Python:", sys.executable)
"""
    ),
    code_cell(
        """
# Step 1: COCO preprocessing.
# This writes outputs/preprocessing/final_train.csv where target is derived from
# train_annotations.json, not blindly copied from train.csv.
subprocess.check_call([
    sys.executable,
    str(ROOT / "scripts" / "preprocess_coco.py"),
])
"""
    ),
    code_cell(
        """
# Step 2: merge ConvNeXt Small + MaxViT Tiny OOF predictions into soft labels.
# If teacher OOF files are not present yet, this still creates the table with
# COCO hard labels and teacher_available=False. For the real final model, rerun
# after both teachers have finished P2 export.
import json

REQUIRE_TEACHERS = True
subprocess.check_call([
    sys.executable,
    str(ROOT / "scripts" / "make_teacher_soft_labels.py"),
])
soft_summary_path = ROOT / "outputs" / "final_effnetv2_s" / "teacher_soft_summary.json"
soft_summary = json.loads(soft_summary_path.read_text())
print(json.dumps(soft_summary, indent=2))
if REQUIRE_TEACHERS and soft_summary["teacher_coverage"] == 0:
    raise RuntimeError(
        "Teacher OOF predictions are missing. Train/export ConvNeXt Small and "
        "MaxViT Tiny first, or set REQUIRE_TEACHERS=False for a hard-label smoke run."
    )
"""
    ),
    code_cell(
        """
# Step 3: train final EfficientNetV2-S student and export ONNX.
# On Kaggle GPU, start with EPOCHS=8. On the Mac, use MAX_TRAIN_ROWS for a quick
# smoke run first, then remove it for the real run.
EPOCHS = 8
BATCH_SIZE = 16
MAX_TRAIN_ROWS = None  # set to 512 for a fast local smoke test

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "train_final_student.py"),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--num-workers", "2",
]
if MAX_TRAIN_ROWS is not None:
    cmd += ["--max-train-rows", str(MAX_TRAIN_ROWS)]

print(" ".join(cmd))
subprocess.check_call(cmd)
"""
    ),
    code_cell(
        """
# Step 4: final artifacts expected by final_evaluation.ipynb.
expected = [
    ROOT / "outputs" / "final_effnetv2_s" / "final_effnetv2_s.onnx",
    ROOT / "outputs" / "final_effnetv2_s" / "model_metadata.json",
    ROOT / "outputs" / "final_effnetv2_s" / "teacher_soft_train.csv",
    ROOT / "outputs" / "final_effnetv2_s" / "final_student_summary.json",
]
for path in expected:
    print(("OK  " if path.exists() else "MISS"), path)

summary_path = ROOT / "outputs" / "final_effnetv2_s" / "final_student_summary.json"
if summary_path.exists():
    print(summary_path.read_text())
"""
    ),
]


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (FINAL_DIR / "final_evaluation.ipynb").write_text(
        json.dumps(notebook(FINAL_EVALUATION_CELLS), indent=1) + "\n"
    )
    (FINAL_DIR / "final_training.ipynb").write_text(
        json.dumps(notebook(FINAL_TRAINING_CELLS), indent=1) + "\n"
    )

    metadata = {
        "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
        "model_file": "final_effnetv2_s.onnx",
        "img_size": 256,
        "batch_size": 64,
        "threshold": 0.5,
        "output_activation": "sigmoid",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_names": ["image", "btype_id"],
        "onnx_opset": 18,
        "uses_roi_crop": True,
        "uses_bottle_type_metadata": True,
        "training_label_source": "COCO-derived target from train_annotations.json",
        "notes": "Replace threshold after final validation calibration. Attach final_effnetv2_s.onnx next to this metadata.",
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    readme = """# Final Submission Package

This folder is the clean final-submission target.

Files:
- `final_evaluation.ipynb`: standalone ONNX inference notebook.
- `final_training.ipynb`: final student training controller.
- `model/model_metadata.json`: preprocessing and threshold metadata.

Before final sharing, attach `final_effnetv2_s.onnx` as a Kaggle dataset/input
next to `model_metadata.json`, then run `final_evaluation.ipynb` end to end and
confirm it writes `submission.csv`.

The final ONNX expects two inputs: image tensor and `btype_id`. The evaluation
notebook loads `bottletypes.csv` and passes that metadata automatically.
"""
    (FINAL_DIR / "README.md").write_text(readme)

    print(f"Wrote {FINAL_DIR / 'final_evaluation.ipynb'}")
    print(f"Wrote {FINAL_DIR / 'final_training.ipynb'}")
    print(f"Wrote {MODEL_DIR / 'model_metadata.json'}")
    print(f"Wrote {FINAL_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
