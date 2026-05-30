import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LEADERBOARD_MODELS = [
    {
        "source": ROOT / "notebooks" / "v26_convnext_small_mac.ipynb",
        "out": ROOT / "notebooks" / "lb_convnext_small_mac.ipynb",
        "old_tag": "v26_convnext_small",
        "tag": "lb_convnext_small",
        "model_name": "convnext_small.fb_in22k_ft_in1k",
        "label": "ConvNeXt Small",
    },
    {
        "source": ROOT / "notebooks" / "v26_maxvit_mac.ipynb",
        "out": ROOT / "notebooks" / "lb_maxvit_mac.ipynb",
        "old_tag": "v26_maxvit",
        "tag": "lb_maxvit",
        "model_name": "maxvit_rmlp_tiny_rw_256.sw_in1k",
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


LEADERBOARD_SPLIT_CELL = """# =========================================================
# CELL 9: LEADERBOARD SPLIT - use all labeled data for CV
# =========================================================
# Leaderboard mode disables the extra 10% holdout so every labeled image can
# contribute to the 5-fold CV ensemble. OOF predictions from the folds still
# provide threshold/weight tuning, but there is no separate untouched guardrail.

train_idx_all = np.arange(len(labels_corrected))
holdout_idx = np.array([], dtype=int)

_hold_p  = SAVE / f'holdout_indices_{VERSION_TAG}.npy'
_train_p = SAVE / f'train_indices_{VERSION_TAG}.npy'
np.save(_hold_p, holdout_idx)
np.save(_train_p, train_idx_all)

labels_train_active = labels_corrected.reset_index(drop=True).copy()
labels_holdout = labels_corrected.iloc[[]].reset_index(drop=True).copy()
y_holdout = np.array([], dtype=np.int64)

labels_train_active.to_csv(SAVE / f'labels_train_active_{VERSION_TAG}.csv', index=False)
labels_holdout.to_csv(SAVE / f'labels_holdout_{VERSION_TAG}.csv', index=False)

print('Leaderboard mode: using all labeled data for 5-fold training.')
print(f'  Train active: {len(labels_train_active)} (100%)')
print(f'  Holdout:      {len(labels_holdout)} (0%)')
print(f'  Train class1 ratio: {labels_train_active.target.mean():.4f}')

print('\\nPer-type training coverage:')
for tname, tid in BOTTLE_TYPE_CANONICAL.items():
    t_n = (labels_train_active.btype_id == tid).sum()
    print(f'  {tname:7s}: train={t_n:>5d}')

print('\\nOOF fold validation replaces the separate holdout for leaderboard mode.')
"""


def set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.splitlines()]


def patch_notebook(model: dict) -> None:
    nb = json.loads(model["source"].read_text())

    for cell in nb["cells"]:
        if cell.get("cell_type") not in {"code", "markdown"}:
            continue
        text = "".join(cell.get("source", []))
        text = text.replace(model["old_tag"], model["tag"])
        text = text.replace("HOLDOUT_FRACTION = 0.10", "HOLDOUT_FRACTION = 0.0")
        text = text.replace(
            f"# Ensemble Member: {model['label']}",
            f"# Leaderboard Ensemble Member: {model['label']}",
        )
        text = text.replace(
            "This copy was generated for Apple Silicon local runs.",
            "This leaderboard copy was generated for Apple Silicon local runs.",
        )
        set_cell_source(cell, text.rstrip())

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and "# CELL 9:" in "".join(cell.get("source", [])):
            set_cell_source(cell, LEADERBOARD_SPLIT_CELL)
            break
    else:
        raise RuntimeError(f"Could not find Cell 9 in {model['source']}")

    note = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Leaderboard Mode: {model['label']}\n",
            "\n",
            "This notebook disables the extra 10% holdout and uses all labeled images in 5-fold CV training.\n",
            f"- `VERSION_TAG`: `{model['tag']}`\n",
            f"- `MODEL_NAME`: `{model['model_name']}`\n",
            f"- Outputs: `outputs/{model['tag']}/`\n",
        ],
    }
    nb["cells"].insert(3, note)
    model["out"].write_text(json.dumps(nb, indent=1))
    print(f"Wrote {model['out']}")


def main() -> int:
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_ensemble_notebooks.py")])

    for model in LEADERBOARD_MODELS:
        patch_notebook(model)

    manifest = {
        "models": [
            {
                "tag": model["tag"],
                "model_name": model["model_name"],
                "notebook": model["out"].name,
                "label": model["label"],
                "weight": 1.0,
            }
            for model in LEADERBOARD_MODELS
        ],
        "final_student": FINAL_STUDENT,
        "prediction_files": [
            f"outputs/{model['tag']}/test_{model['tag']}_p2_mean.npy"
            for model in LEADERBOARD_MODELS
        ],
    }
    manifest_path = ROOT / "outputs" / "leaderboard_models.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
