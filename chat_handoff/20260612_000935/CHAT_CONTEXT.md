# Krones Challenge Handoff

Generated: Fri Jun 12 00:09:35 CEST 2026

Project root:

```text
/Users/pranayrudra/Projects/Krones Challenge
```

## Current Goal

The current goal shifted from public leaderboard chasing to preparing the final
submission package. The chosen final strategy is:

- preprocess labels from COCO annotations, not directly from `train.csv`;
- use ROI cropping from COCO boxes;
- include bottle-type metadata as a second ONNX input;
- use ConvNeXt Small + MaxViT Tiny as teacher models;
- train EfficientNetV2-S as the final single student model;
- apply robust noisy-label weighting when teacher ensemble strongly disagrees
  with the COCO-derived label;
- export EfficientNetV2-S to ONNX for final evaluation.

## Final Model State

The final EfficientNetV2-S student has already been trained locally and exported.

```text
best validation F1: 0.9636056881176187
best epoch: 5
threshold: 0.4050000000000002
train rows: 31807
valid rows: 3535
```

Important detail: PyTorch exported the ONNX model using external data. The file
`final_effnetv2_s.onnx.data` is required and must stay in the same folder as
`final_effnetv2_s.onnx`.

## Final Submission State

The final evaluation notebook was executed successfully and produced a valid CSV.

```text
submission rows: 4418
columns: image_id,target
target values: [0, 1]
positive predictions: 2558
```

Primary final files:

```text
final_submission/final_evaluation.ipynb
final_submission/model/final_effnetv2_s.onnx
final_submission/model/final_effnetv2_s.onnx.data
final_submission/model/model_metadata.json
final_submission/submission.csv
```

## Verification Commands

Check ONNX loads:

```bash
source .venv/bin/activate
python - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession("final_submission/model/final_effnetv2_s.onnx", providers=["CPUExecutionProvider"])
print([i.name for i in s.get_inputs()], [o.name for o in s.get_outputs()])
PY
```

Run final evaluation notebook:

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace final_submission/final_evaluation.ipynb --ExecutePreprocessor.timeout=0
```

Validate submission:

```bash
source .venv/bin/activate
python - <<'PY'
import pandas as pd
sub = pd.read_csv("final_submission/submission.csv")
sample = pd.read_csv("data/sample_submission.csv")
print(len(sub), len(sample), list(sub.columns), sorted(sub.target.unique()))
PY
```

## Safety Notes

- Do not store or repeat Kaggle tokens in project files.
- Do not include the full 21 GB dataset in handoff packets.
- Keep `.onnx` and `.onnx.data` together.
- The final evaluation notebook is the clean final target; development notebooks
  and scripts are supporting material only.
