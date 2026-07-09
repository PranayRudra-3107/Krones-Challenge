# Krones Challenge

Computer vision pipeline for the Krones Vision AI Challenge. The task is binary
defect classification for bottle inspection images: predict whether each bottle
image contains an actionable defect.

This repository contains the local Apple Silicon workflow, Kaggle GPU helper
notebooks/scripts, COCO-label preprocessing, teacher-student distillation, ONNX
export, and final submission tooling.

## Project Summary

The final approach is a single EfficientNetV2-S student model trained with:

- labels derived from the official COCO annotations, not blindly from
  `train.csv`;
- ROI cropping from COCO bottle regions;
- bottle-type metadata as an additional model input;
- teacher soft labels from ConvNeXt Small and DINOv3 ViT-L predictions;
- robust sample weighting when teacher predictions disagree with COCO-derived
  labels;
- ONNX export for final Kaggle evaluation.

The final ONNX model expects two inputs:

```text
image
btype_id
```

The current exported metadata records:

```text
student model: EfficientNetV2-S
image size:    256
ONNX opset:    18
validation F1: 0.96616
best epoch:    8
```

## Repository Layout

```text
.
|-- configs/
|   |-- final_strategy.json
|   `-- kaggle_teacher_jobs.json
|-- final_submission/
|   |-- final_evaluation.ipynb
|   |-- final_training.ipynb
|   |-- model/
|   |   |-- final_effnetv2_s.onnx
|   |   |-- final_effnetv2_s.onnx.data
|   |   `-- model_metadata.json
|   |-- README.md
|   `-- submission.csv
|-- notebooks/
|   |-- group22_eda_label_correction.ipynb
|   |-- lb_convnext_small_feature_export.ipynb
|   |-- lb_convnext_small_mac.ipynb
|   |-- lb_maxvit_feature_export.ipynb
|   |-- lb_maxvit_mac.ipynb
|   |-- v26_convnext_small_mac.ipynb
|   `-- v26_maxvit_mac.ipynb
|-- scripts/
|   |-- apply_student_oof_thresholds.py
|   |-- check_setup.py
|   |-- ensemble_student_onnx.py
|   |-- ensemble_submit.py
|   |-- generate_onnx_threshold_submissions.py
|   |-- kaggle_monitor.py
|   |-- make_teacher_soft_labels.py
|   |-- preprocess_coco.py
|   |-- train_final_student.py
|   `-- ...
|-- requirements-mac.txt
`-- README.md
```

Large local folders such as `data/`, `outputs/`, `.venv/`, checkpoints, and NumPy
prediction arrays are intentionally ignored by git.

## Environment Setup

From the project root:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
```

Run the setup check:

```bash
python scripts/check_setup.py
```

On Apple Silicon, a healthy setup should report PyTorch MPS availability and
find the expected competition files under `data/`.

## Data

Expected local data layout:

```text
data/
|-- bottletypes.csv
|-- sample_submission.csv
|-- test_annotations_roi_only.json
|-- test_images/
|-- train.csv
|-- train_annotations.json
`-- train_images/
```

The dataset is not committed to this repository. Download it through Kaggle or
place it manually under `data/`.

## COCO Preprocessing

Run preprocessing before training:

```bash
source .venv/bin/activate
python scripts/preprocess_coco.py
```

This parses `train_annotations.json` and `test_annotations_roi_only.json`,
applies the official category rules, joins bottle-type metadata, and writes:

```text
outputs/preprocessing/final_train.csv
outputs/preprocessing/final_test.csv
outputs/preprocessing/label_coco_audit.csv
outputs/preprocessing/label_mismatches.csv
outputs/preprocessing/train_roi.csv
outputs/preprocessing/test_roi.csv
outputs/preprocessing/preprocessing_summary.json
```

Current preprocessing audit:

```text
train rows:        35342
test rows:         4418
label mismatches:  0
train ROI missing: 0
```

Downstream training should use `outputs/preprocessing/final_train.csv`, where
`target` is the COCO-derived label.

## Teacher Models

Teacher predictions are used to build soft labels for the final student.

Current teacher strategy:

```text
ConvNeXt Small: convnext_small.fb_in22k_ft_in1k
DINOv3 ViT-L:   precomputed OOF/test probabilities
Student:        tf_efficientnetv2_s.in21k_ft_in1k
```

Generate teacher soft-label tables:

```bash
source .venv/bin/activate
python scripts/make_teacher_soft_labels.py
```

Expected outputs:

```text
outputs/final_effnetv2_s/teacher_soft_train.csv
outputs/final_effnetv2_s/teacher_soft_test.csv
outputs/final_effnetv2_s/teacher_soft_summary.json
```

The soft-label builder keeps the COCO-derived target and adds teacher probability
columns plus sample weights for noisy-label robustness.

## Student Training

Train and export the final EfficientNetV2-S student:

```bash
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_final_student.py \
  --device mps \
  --train-csv outputs/final_effnetv2_s/teacher_soft_train.csv \
  --out-dir outputs/final_effnetv2_s \
  --onnx-path outputs/final_effnetv2_s/final_effnetv2_s.onnx \
  --epochs 12 \
  --batch-size 8 \
  --infer-batch-size 32 \
  --num-workers 0 \
  --img-size 256 \
  --lr 1e-4 \
  --distill-alpha 0.75 \
  --btype-weight 0.10
```

Main outputs:

```text
outputs/final_effnetv2_s/final_effnetv2_s_best.pth
outputs/final_effnetv2_s/final_effnetv2_s.onnx
outputs/final_effnetv2_s/final_effnetv2_s.onnx.data
outputs/final_effnetv2_s/model_metadata.json
outputs/final_effnetv2_s/final_student_summary.json
```

Important: the ONNX export uses external data. Keep
`final_effnetv2_s.onnx.data` in the same folder as `final_effnetv2_s.onnx`.

## Submission Generation

Run the clean final evaluation notebook:

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace \
  final_submission/final_evaluation.ipynb \
  --ExecutePreprocessor.timeout=0
```

Validate the submission:

```bash
python - <<'PY'
import pandas as pd

sub = pd.read_csv("final_submission/submission.csv")
sample = pd.read_csv("data/sample_submission.csv")

print("rows:", len(sub), "expected:", len(sample))
print("columns:", list(sub.columns))
print("target values:", sorted(sub.target.unique()))
print("positive predictions:", int(sub.target.sum()))
PY
```

The expected submission format is:

```text
image_id,target
```

## Threshold and Ensemble Utilities

Generate threshold-sweep submissions from an exported ONNX model:

```bash
source .venv/bin/activate
python scripts/generate_onnx_threshold_submissions.py \
  --onnx outputs/final_effnetv2_s/final_effnetv2_s.onnx \
  --metadata outputs/final_effnetv2_s/model_metadata.json \
  --out-dir outputs/final_effnetv2_s/threshold_sweep
```

Average several student ONNX exports:

```bash
source .venv/bin/activate
python scripts/ensemble_student_onnx.py \
  --models path/to/fold0.onnx path/to/fold1.onnx \
  --metadata path/to/model_metadata.json \
  --out-dir outputs/student_ensemble
```

Blend teacher prediction files:

```bash
source .venv/bin/activate
python scripts/ensemble_submit.py \
  --config outputs/leaderboard_models.json \
  --out-dir outputs/leaderboard_ensemble \
  --optimize-weights \
  --per-bottle-thresholds
```

Use public leaderboard submissions sparingly. Threshold tuning can quickly burn
the daily Kaggle submission allowance.

## Kaggle GPU Workflow

Kaggle GPU is useful for teacher training and heavier experiments. Generate and
push chunked Kaggle notebooks rather than relying on one long interactive run.

Prepare a chunk:

```bash
source .venv/bin/activate
scripts/kaggle_push_chunk.sh \
  --model lb_convnext_small \
  --phase p1 \
  --folds 0 \
  --no-export
```

Monitor jobs:

```bash
source .venv/bin/activate
python scripts/kaggle_monitor.py --watch --interval 300 --log-lines 80
```

Download completed outputs:

```bash
python scripts/kaggle_monitor.py --download-completed
```

Kaggle credentials should live outside the repository, for example in
`~/.kaggle/access_token`. Do not commit API tokens.

## Final Submission Package

The final Kaggle-facing package is:

```text
final_submission/final_evaluation.ipynb
final_submission/model/final_effnetv2_s.onnx
final_submission/model/final_effnetv2_s.onnx.data
final_submission/model/model_metadata.json
final_submission/submission.csv
```

`final_evaluation.ipynb` is the clean inference notebook. It loads the ONNX
model, reads bottle-type metadata, applies ROI cropping and ImageNet
normalization, runs ONNX Runtime inference, thresholds probabilities, and writes
`submission.csv`.

## Reproducibility Notes

- COCO annotations are the canonical label source.
- `train.csv` labels are kept only for auditing.
- ROI boxes come from COCO annotation files.
- Bottle-type metadata comes from `data/bottletypes.csv`.
- Model weights and large prediction arrays are not committed.
- The project has both local Mac notebooks and Kaggle GPU helpers; the final
  evaluation notebook is the clean artifact for submission.

## Useful Commands

```bash
# Check local setup
python scripts/check_setup.py

# Rebuild preprocessing tables
python scripts/preprocess_coco.py

# Build teacher soft labels
python scripts/make_teacher_soft_labels.py

# Train final student
python scripts/train_final_student.py --epochs 12 --batch-size 8

# Execute final evaluation notebook
jupyter nbconvert --execute --to notebook --inplace final_submission/final_evaluation.ipynb
```

## Safety

Do not commit:

- Kaggle API tokens;
- the full competition dataset;
- `.venv/`;
- large experimental `outputs/` directories;
- temporary notebooks created only for one-off Kaggle runs.
