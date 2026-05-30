import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = ROOT / "notebooks" / "v26_maxvit_mac.ipynb"

TEACHER_MODELS = [
    {
        "tag": "v26_convnext_small",
        "model_name": "convnext_small.fb_in22k_ft_in1k",
        "notebook": "v26_convnext_small_mac.ipynb",
        "label": "ConvNeXt Small",
    },
    {
        "tag": "v26_maxvit",
        "model_name": "maxvit_rmlp_tiny_rw_256.sw_in1k",
        "notebook": "v26_maxvit_mac.ipynb",
        "label": "MaxViT Tiny",
    },
]

FINAL_STUDENT = {
    "tag": "final_effnetv2_s",
    "model_name": "tf_efficientnetv2_s.in21k_ft_in1k",
    "label": "EfficientNetV2-S",
    "role": "final_student",
    "target_export": "outputs/final_effnetv2_s/final_effnetv2_s.onnx",
}

FINAL_STRATEGY = {
    "goal": "final_submission",
    "teachers": [
        {
            "tag": model["tag"],
            "model_name": model["model_name"],
            "notebook": model["notebook"],
            "label": model["label"],
            "role": "teacher",
        }
        for model in TEACHER_MODELS
    ],
    "student": FINAL_STUDENT,
    "preprocessing": {
        "label_source": "COCO annotations from train_annotations.json",
        "canonical_train_file": "outputs/preprocessing/final_train.csv",
        "roi_source": "COCO ROI annotations",
        "bottle_type_metadata": "data/bottletypes.csv",
    },
    "student_training": {
        "teacher_soft_label_file": "outputs/final_effnetv2_s/teacher_soft_train.csv",
        "robust_noisy_label_weighting": True,
        "roi_cropping": True,
        "bottle_type_input": True,
        "onnx_inputs": ["image", "btype_id"],
    },
    "summary": (
        "Use COCO-derived labels, train ConvNeXt Small + MaxViT Tiny as teachers, "
        "distill their soft predictions into one ROI/bottle-aware EfficientNetV2-S ONNX model."
    ),
}

def set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.splitlines()]


def patch_notebook(nb: dict, model: dict) -> dict:
    replacements = {
        "VERSION_TAG = 'v26_maxvit'": f"VERSION_TAG = '{model['tag']}'",
        "MODEL_NAME  = 'maxvit_rmlp_tiny_rw_256.sw_in1k'   # 256px native, timm-native weights, fast on T4": (
            f"MODEL_NAME  = '{model['model_name']}'"
        ),
        "print(f'\\nMaxViT config: {VERSION_TAG}')": "print(f'\\nModel config: {VERSION_TAG}')",
        "print('MaxViT TEACHER EXPORT \u2014 P2 test predictions for B4 ensemble')": (
            "print('ENSEMBLE MEMBER EXPORT - P2 test predictions')"
        ),
        "MaxViT TEACHER EXPORT \u2014 P2 test predictions for B4 ensemble": (
            "ENSEMBLE MEMBER EXPORT - P2 test predictions"
        ),
        "MaxViT P2 OOF F1": f"{model['label']} P2 OOF F1",
        "MaxViT holdout F1": f"{model['label']} holdout F1",
        "MaxViT is a strong teacher": f"{model['label']} is a strong teacher",
        "This notebook does NOT export ONNX. MaxViT is teacher-only.": (
            "This notebook does NOT export ONNX. It exports fold and mean probabilities for ensembling."
        ),
    }

    for cell in nb["cells"]:
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        text = "".join(cell.get("source", []))
        for old, new in replacements.items():
            text = text.replace(old, new)
        set_cell_source(cell, text.rstrip())

    note = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Ensemble Member: {model['label']}\n",
            "\n",
            f"- `VERSION_TAG`: `{model['tag']}`\n",
            f"- `MODEL_NAME`: `{model['model_name']}`\n",
            f"- Outputs: `outputs/{model['tag']}/`\n",
        ],
    }
    nb["cells"].insert(2, note)
    return nb


def main() -> int:
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_mac_notebook.py")])

    base = json.loads(BASE_NOTEBOOK.read_text())
    for model in TEACHER_MODELS:
        nb = json.loads(json.dumps(base))
        patched = patch_notebook(nb, model)
        out_path = ROOT / "notebooks" / model["notebook"]
        out_path.write_text(json.dumps(patched, indent=1))
        print(f"Wrote {out_path}")

    manifest = {
        "models": TEACHER_MODELS,
        "prediction_files": [
            f"outputs/{model['tag']}/test_{model['tag']}_p2_mean.npy"
            for model in TEACHER_MODELS
        ],
        "final_student": FINAL_STUDENT,
    }
    manifest_path = ROOT / "outputs" / "ensemble_models.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")

    config_path = ROOT / "configs" / "final_strategy.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(FINAL_STRATEGY, indent=2) + "\n")
    print(f"Wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
