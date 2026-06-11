# Krones Challenge Local Setup

This workspace is prepared to run the `v26_maxvit` notebook locally on Apple
Silicon using PyTorch MPS.

The original notebook in Downloads is left untouched. The local, Mac-adjusted
copy is:

```text
notebooks/v26_maxvit_mac.ipynb
```

## Current Status

Verified on this machine:

```text
Python: 3.11 in .venv
PyTorch: installed
Device: Apple Silicon MPS available
Kaggle CLI: installed and authenticated
Dataset: present under data/
```

Dataset counts:

```text
train rows: 35342
train images: 35342
test images: 4418
sample submission rows: 4418
```

## Project Layout

```text
.
|-- data/
|   |-- bottletypes.csv
|   |-- sample_submission.csv
|   |-- test_annotations_roi_only.json
|   |-- test_images/
|   |-- train.csv
|   |-- train_annotations.json
|   `-- train_images/
|-- configs/
|   `-- final_strategy.json
|-- final_submission/
|   |-- final_evaluation.ipynb
|   |-- final_training.ipynb
|   |-- model/
|   |   |-- final_effnetv2_s.onnx
|   |   |-- final_effnetv2_s.onnx.data
|   |   `-- model_metadata.json
|   |-- submission.csv
|   `-- README.md
|-- notebooks/
|   |-- lb_convnext_small_mac.ipynb
|   |-- lb_maxvit_mac.ipynb
|   |-- v26_convnext_small_mac.ipynb
|   `-- v26_maxvit_mac.ipynb
|-- outputs/
|-- scripts/
|   |-- check_setup.py
|   |-- ensemble_submit.py
|   |-- kaggle_download.sh
|   |-- kaggle_submit.sh
|   |-- make_ensemble_notebooks.py
|   |-- make_final_submission_notebooks.py
|   |-- make_kaggle_chunk_notebook.py
|   |-- make_leaderboard_notebooks.py
|   |-- make_mac_notebook.py
|   |-- make_teacher_soft_labels.py
|   |-- preprocess_coco.py
|   |-- smoke_ensemble_models.py
|   `-- train_final_student.py
|-- requirements-mac.txt
`-- README.md
```

`data/`, `outputs/`, `.venv/`, model weights, and NumPy output files are ignored
by git.

## Environment

Activate the local environment:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate
```

Install or refresh dependencies:

```bash
uv pip install -r requirements-mac.txt
```

Run the setup check:

```bash
python scripts/check_setup.py
```

A healthy setup should show all dataset paths as `OK`, `mps available: True`,
and `Setup looks ready.`

## Running Jupyter

Start Jupyter Lab:

```bash
source .venv/bin/activate
jupyter lab
```

Then open:

```text
notebooks/v26_maxvit_mac.ipynb
```

Use the kernel:

```text
Krones MaxViT (.venv)
```

The notebook is configured to use:

```text
DATA = /Users/pranayrudra/Projects/Krones Challenge/data
SAVE = /Users/pranayrudra/Projects/Krones Challenge/outputs/v26_maxvit
DEVICE = mps
```

## Notebook Changes

The Mac notebook keeps the competition logic from the Kaggle notebook, but
changes local runtime concerns:

- Uses project-relative `data/` and `outputs/v26_maxvit/` paths
- Detects Apple Silicon `mps`
- Enables MPS CPU fallback for unsupported operations
- Guards CUDA-only AMP/cache calls
- Uses `BATCH_SIZE = 4` and `GRAD_ACCUM_STEPS = 8`
- Uses `NUM_WORKERS = 0` for safer local Jupyter execution
- Disables `pin_memory` unless CUDA is active
- Uses the official COCO label rules from the competition brief for rule-based
  label auditing

Do not casually run the entire notebook unless you are ready for a long training
run. Start with the setup, config, audit, and ROI-analysis cells first.

## Final Strategy

The selected final-submission plan is teacher-student distillation:

```text
Teachers:
  ConvNeXt Small: convnext_small.fb_in22k_ft_in1k
  MaxViT Tiny:    maxvit_rmlp_tiny_rw_256.sw_in1k

Final student:
  EfficientNetV2-S: tf_efficientnetv2_s.in21k_ft_in1k
```

The teachers are used to produce diverse soft predictions. EfficientNetV2-S is
reserved for the final single-model student and ONNX export, so it is not also
trained as a teacher.

The strategy is recorded in:

```text
configs/final_strategy.json
```

Generate or refresh the two teacher notebooks:

```bash
source .venv/bin/activate
python scripts/make_ensemble_notebooks.py
```

This creates:

```text
notebooks/v26_convnext_small_mac.ipynb
notebooks/v26_maxvit_mac.ipynb
```

Smoke-test the two teacher backbones plus the EfficientNetV2-S final student:

```bash
source .venv/bin/activate
python scripts/smoke_ensemble_models.py
```

To train the teachers, run each teacher notebook separately through its P2 export
cell. Each teacher writes to its own output folder:

```text
outputs/v26_convnext_small/
outputs/v26_maxvit/
```

After both teacher notebooks have produced these files:

```text
outputs/v26_convnext_small/oof_v26_convnext_small_p2_fold*.npy
outputs/v26_maxvit/oof_v26_maxvit_p2_fold*.npy
outputs/v26_convnext_small/test_v26_convnext_small_p2_mean.npy
outputs/v26_maxvit/test_v26_maxvit_p2_mean.npy
```

Build the final student training table:

```bash
source .venv/bin/activate
python scripts/preprocess_coco.py
python scripts/make_teacher_soft_labels.py
```

This creates:

```text
outputs/preprocessing/final_train.csv
outputs/preprocessing/final_test.csv
outputs/final_effnetv2_s/teacher_soft_train.csv
outputs/final_effnetv2_s/teacher_soft_test.csv
```

`final_train.csv` uses `target` derived from COCO annotations. The original
`train.csv` target is preserved only as `target_original` for audit.
`final_submission/final_training.ipynb` has `REQUIRE_TEACHERS = True`, so a
Run All will stop before student training if the teacher OOF predictions are
not present yet.

Train the final EfficientNetV2-S student and export ONNX:

```bash
source .venv/bin/activate
python scripts/train_final_student.py --epochs 8 --batch-size 16 --num-workers 2
```

The student uses:

```text
ROI crop from COCO annotations
bottle-type metadata as ONNX input btype_id
ConvNeXt + MaxViT teacher soft labels when available
robust noisy-label weights when teachers strongly disagree with the COCO target
```

The final artifacts are:

```text
outputs/final_effnetv2_s/final_effnetv2_s.onnx
outputs/final_effnetv2_s/final_effnetv2_s.onnx.data
outputs/final_effnetv2_s/model_metadata.json
outputs/final_effnetv2_s/final_student_summary.json
```

Current final student validation:

```text
best validation F1: 0.96361
best epoch: 5
threshold: 0.4050
train rows: 31807
valid rows: 3535
```

Full teacher training on a 16 GB Mac can take a long time. Kaggle GPU chunk
training is the practical path for teacher checkpoints; the local Mac setup is
still useful for preprocessing, smoke tests, debugging, soft-label generation, and the final
ONNX/evaluation notebook checks.

## Final Submission Notebooks

The final evaluation submission should not use the local project control
notebook or call scripts from this repository. Use the clean final package:

```text
final_submission/final_evaluation.ipynb
final_submission/final_training.ipynb
final_submission/model/final_effnetv2_s.onnx
final_submission/model/final_effnetv2_s.onnx.data
final_submission/model/model_metadata.json
```

Regenerate the scaffolds:

```bash
source .venv/bin/activate
python scripts/make_final_submission_notebooks.py
```

`final_evaluation.ipynb` is standalone. It finds the attached Kaggle competition
data, loads `final_effnetv2_s.onnx`, applies ROI crop + ImageNet normalization,
runs ONNX inference, thresholds probabilities, and writes `submission.csv`.
The final ONNX has two inputs: `image` and `btype_id`; the notebook reads
`bottletypes.csv` and feeds the bottle-type metadata automatically.

Before final sharing, attach a Kaggle dataset/input containing:

```text
final_effnetv2_s.onnx
final_effnetv2_s.onnx.data
model_metadata.json
```

The `.onnx.data` file is required because PyTorch exported the ONNX weights as
external data next to the graph file. Keep it in the same folder as
`final_effnetv2_s.onnx`.

Local verification command:

```bash
source .venv/bin/activate
jupyter nbconvert --execute --to notebook --inplace final_submission/final_evaluation.ipynb --ExecutePreprocessor.timeout=0
```

Verified local output:

```text
submission.csv rows: 4418
target values: [0, 1]
positive predictions: 2558
```

Do not share local development controller notebooks as the final evaluation
notebook. The final evaluation target is `final_submission/final_evaluation.ipynb`.

## Leaderboard Push

The current top public leaderboard score checked locally is:

```text
0.97324
```

To chase the public leaderboard first, use the `lb_*` notebooks. They disable
the extra 10% holdout and use all labeled images in 5-fold CV training. This is
less conservative than the `v26_*` notebooks, but it gives the model more data
and is the right mode for a leaderboard push.

Generate leaderboard notebooks:

```bash
source .venv/bin/activate
python scripts/make_leaderboard_notebooks.py
```

Train these teacher notebooks separately:

```text
notebooks/lb_convnext_small_mac.ipynb
notebooks/lb_maxvit_mac.ipynb
```

After both teachers have exported their P2 mean predictions:

```text
outputs/lb_convnext_small/test_lb_convnext_small_p2_mean.npy
outputs/lb_maxvit/test_lb_maxvit_p2_mean.npy
```

Create the teacher blend:

```bash
source .venv/bin/activate
python scripts/ensemble_submit.py \
  --config outputs/leaderboard_models.json \
  --out-dir outputs/leaderboard_ensemble \
  --optimize-weights \
  --per-bottle-thresholds
```

Submit it only if you explicitly want a public leaderboard check:

```bash
scripts/kaggle_submit.sh outputs/leaderboard_ensemble/submission_ensemble.csv "convnext maxvit teacher ensemble"
```

This mode optimizes teacher weights and bottle-type-specific thresholds from OOF
predictions. The final submission target is still one EfficientNetV2-S student
ONNX model, not the slow teacher ensemble.

## Kaggle GPU Chunk Training

For Kaggle free GPU, avoid one huge interactive run. Generate a small Kaggle
chunk notebook, push it as a private GPU kernel, then save/download the output
checkpoint before starting the next chunk.

If this Kaggle account has no GPU quota left, do not use a second personal
account to bypass the limit. The safe handoff path is either:

- a real teammate Kaggle account that has joined the Krones Challenge and the
  same Kaggle team;
- an external GPU machine where competition rules allow external compute.

Create a portable handoff package for a joined teammate account:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate

python scripts/make_gpu_handoff_package.py \
  --model v26_convnext_small \
  --phase p1 \
  --folds 0 \
  --kernel-owner TEAMMATE_KAGGLE_USERNAME \
  --accelerator NvidiaTeslaT4 \
  --require-gpu-name T4 \
  --no-export
```

This writes:

```text
handoff/krones-v26-convnext-small-p1-f0/
handoff/krones-v26-convnext-small-p1-f0.zip
```

The teammate runs this from inside the unzipped handoff folder:

```bash
pip install kaggle
kaggle kernels push -p kaggle_gpu_chunk --accelerator NvidiaTeslaT4
kaggle kernels status TEAMMATE_KAGGLE_USERNAME/krones-v26-convnext-small-p1-f0
```

After completion, they download outputs:

```bash
mkdir -p downloaded_outputs
kaggle kernels output TEAMMATE_KAGGLE_USERNAME/krones-v26-convnext-small-p1-f0 \
  -p downloaded_outputs
```

Put the returned files under this Mac project, then merge the model artifacts:

```bash
mkdir -p outputs/kaggle_runs/krones-v26-convnext-small-p1-f0
# place returned files in outputs/kaggle_runs/krones-v26-convnext-small-p1-f0/
rsync -av \
  outputs/kaggle_runs/krones-v26-convnext-small-p1-f0/outputs/v26_convnext_small/ \
  outputs/v26_convnext_small/
```

Use the same handoff command for MaxViT by changing the model:

```bash
python scripts/make_gpu_handoff_package.py \
  --model v26_maxvit \
  --phase p1 \
  --folds 1 \
  --kernel-owner TEAMMATE_KAGGLE_USERNAME \
  --accelerator NvidiaTeslaT4 \
  --require-gpu-name T4 \
  --no-export
```

Start ConvNeXt Small P1 fold 0 on Kaggle GPU:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate

scripts/kaggle_push_chunk.sh \
  --model v26_convnext_small \
  --phase p1 \
  --folds 0 \
  --no-export \
  --kernel-owner dronakp \
  --accelerator NvidiaTeslaT4 \
  --require-gpu-name T4

.venv/bin/kaggle kernels push -p kaggle/gpu_chunk --accelerator NvidiaTeslaT4
```

If the active Kaggle account uses the private `Krones_Challenges` dataset
instead of direct competition input access, add the dataset source and skip the
competition source:

```bash
scripts/kaggle_push_chunk.sh \
  --model v26_convnext_small \
  --phase p1 \
  --folds 0 \
  --no-export \
  --kernel-owner pranayrudra3107 \
  --accelerator NvidiaTeslaT4 \
  --require-gpu-name T4 \
  --dataset-source pranayrudra3107/krones-challenges \
  --dataset-source pranayrudra3107/krones-metadata-files \
  --no-competition-source

.venv/bin/kaggle kernels push -p kaggle/gpu_chunk --accelerator NvidiaTeslaT4
```

Expected ConvNeXt kernel ref:

```text
pranayrudra3107/krones-v26-convnext-small-p1-f0
```

Start MaxViT Tiny P1 fold 0 on Kaggle GPU:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate

scripts/kaggle_push_chunk.sh \
  --model v26_maxvit \
  --phase p1 \
  --folds 0 \
  --no-export \
  --kernel-owner dronakp \
  --accelerator NvidiaTeslaT4 \
  --require-gpu-name T4

.venv/bin/kaggle kernels push -p kaggle/gpu_chunk --accelerator NvidiaTeslaT4
```

Expected MaxViT kernel ref:

```text
dronakp/krones-v26-maxvit-p1-f0
```

To run all P1 folds automatically in one T4 job, change `--folds 0` to:

```text
--folds 0,1,2,3,4
```

The helper writes the prepared Kaggle package here before each push:

```text
kaggle/gpu_chunk/krones_gpu_chunk.ipynb
kaggle/gpu_chunk/kernel-metadata.json
```

Track both teacher jobs from one terminal:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate
python scripts/kaggle_monitor.py --watch --interval 300
```

For a live terminal progress bar with current fold, epoch, and last F1:

```bash
cd "/Users/pranayrudra/Projects/Krones Challenge"
source .venv/bin/activate
python scripts/kaggle_monitor.py --watch --interval 120 --log-lines 80
```

Add `--show-logs` if you also want the raw Kaggle log tail under the progress
table.

For a one-time check:

```bash
python scripts/kaggle_monitor.py
```

Download outputs after it finishes:

```bash
mkdir -p outputs/kaggle_runs/convnext_p1_f0
.venv/bin/kaggle kernels output dronakp/krones-v26-convnext-small-p1-f0 \
  -p outputs/kaggle_runs/convnext_p1_f0
```

For the next Kaggle run, attach the previous checkpoint output as a Kaggle
kernel/dataset input and pass it to the generator:

```bash
python scripts/make_kaggle_chunk_notebook.py \
  --model v26_convnext_small \
  --phase p2 \
  --folds 0 \
  --kernel-source dronakp/krones-v26-convnext-small-p1-f0
```

Supported model tags:

```text
v26_convnext_small
v26_maxvit
lb_convnext_small
lb_maxvit
```

Supported phases:

```text
p1      train selected P1 folds
p2      resume from P1 checkpoint and train selected P2 folds
both    run P1 then P2 for selected folds in one Kaggle version
export  export predictions from available P2 checkpoints
```

There is also a helper that prepares the package without pushing:

```bash
scripts/kaggle_push_chunk.sh --model v26_convnext_small --phase p1 --folds 0 --no-export
```

To make that helper push immediately, set `KAGGLE_PUSH=1`.

Track all teacher GPU jobs from one terminal:

```bash
source .venv/bin/activate
python scripts/kaggle_monitor.py --watch --interval 300
```

For a compact one-time check:

```bash
source .venv/bin/activate
python scripts/kaggle_monitor.py
```

The monitor reads:

```text
configs/kaggle_teacher_jobs.json
```

and writes:

```text
outputs/kaggle_monitor/status.json
outputs/kaggle_monitor/history.jsonl
outputs/kaggle_monitor/*.log
```

When a job is complete, download completed outputs automatically:

```bash
python scripts/kaggle_monitor.py --download-completed
```

## Feature Distillation Continuation

After the leaderboard P2 teacher jobs are downloaded, generate the continuation
notebooks:

```bash
source .venv/bin/activate
python scripts/make_feature_distillation_notebooks.py
```

This writes:

```text
notebooks/lb_convnext_small_feature_export.ipynb
notebooks/lb_maxvit_feature_export.ipynb
```

Open each notebook and run all cells. They do not train the teachers again.
They load the existing P2 fold checkpoints from:

```text
outputs/lb_convnext_small/
outputs/lb_maxvit/
```

and export feature-distillation assets under:

```text
outputs/feature_distillation/lb_convnext_small/
outputs/feature_distillation/lb_maxvit/
```

Main exported files per teacher:

```text
features_<tag>_p2_oof.npy
features_<tag>_p2_test_mean.npy
binary_logits_<tag>_p2_oof.npy
binary_probs_<tag>_p2_oof.npy
aux27_multitarget_<tag>.npy
feature_distill_manifest_<tag>.json
```

Important detail: the current ConvNeXt and MaxViT checkpoints are binary
teachers. They cannot produce true 27-way teacher logits without training a
separate auxiliary teacher head. These notebooks therefore export real teacher
penultimate features plus binary logits/probabilities, and COCO-derived
auxiliary subclass/proxy targets for the 27-signal student workaround.

Train an EfficientNetV2-S student from the stronger leaderboard teacher soft
labels:

```bash
source .venv/bin/activate
python scripts/make_teacher_soft_labels.py \
  --teachers lb_convnext_small lb_maxvit \
  --out-dir outputs/final_effnetv2_s_lb \
  --phases p2 p1

PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu python -u scripts/train_final_student.py \
  --device mps \
  --train-csv outputs/final_effnetv2_s_lb/teacher_soft_train.csv \
  --out-dir outputs/final_effnetv2_s_lb_mps \
  --onnx-path outputs/final_effnetv2_s_lb_mps/final_effnetv2_s.onnx \
  --epochs 12 \
  --batch-size 8 \
  --infer-batch-size 32 \
  --num-workers 0 \
  --img-size 256 \
  --lr 1e-4 \
  --distill-alpha 0.75 \
  --btype-weight 0.10 \
  2>&1 | tee outputs/final_effnetv2_s_lb_mps/train_mps.log
```

After both feature-export notebooks have written their `.npy` files, run the
feature-KD version:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu python -u scripts/train_final_student.py \
  --device mps \
  --train-csv outputs/final_effnetv2_s_lb/teacher_soft_train.csv \
  --out-dir outputs/final_effnetv2_s_feature_kd_mps \
  --onnx-path outputs/final_effnetv2_s_feature_kd_mps/final_effnetv2_s.onnx \
  --feature-distill-dirs \
    outputs/feature_distillation/lb_convnext_small \
    outputs/feature_distillation/lb_maxvit \
  --feature-weight 0.75 \
  --aux27-weight 0.20 \
  --rkd-weight 0.05 \
  --epochs 10 \
  --batch-size 8 \
  --infer-batch-size 32 \
  --num-workers 0 \
  --img-size 256 \
  --lr 8e-5 \
  --distill-alpha 0.75 \
  --btype-weight 0.10 \
  2>&1 | tee outputs/final_effnetv2_s_feature_kd_mps/train_mps.log
```

For a stronger CSV-only attempt, train five warm-start EfficientNetV2-S folds
and average their ONNX predictions:

```bash
source .venv/bin/activate
scripts/train_effnetv2_lb_folds_mps.sh
```

The script writes fold checkpoints under:

```text
outputs/final_effnetv2_s_lb_5fold_mps/fold0/
outputs/final_effnetv2_s_lb_5fold_mps/fold1/
outputs/final_effnetv2_s_lb_5fold_mps/fold2/
outputs/final_effnetv2_s_lb_5fold_mps/fold3/
outputs/final_effnetv2_s_lb_5fold_mps/fold4/
```

and ensemble submissions under:

```text
outputs/final_effnetv2_s_lb_5fold_mps/ensemble/
```

This is a pragmatic public-score attempt: it warm-starts each fold from the best
LB-teacher student checkpoint, then uses the fold diversity for test-time
averaging. Because the warm-start checkpoint has already seen a broad train
split, do not treat the fold validation scores as perfectly clean OOF estimates;
use the public leaderboard and threshold sweep to judge the ensemble.

## COCO Preprocessing

Run the COCO preprocessing/audit script before training:

```bash
source .venv/bin/activate
python scripts/preprocess_coco.py
```

It parses `train_annotations.json` and `test_annotations_roi_only.json`, applies
the official category rules, joins bottle-type metadata, and writes:

```text
outputs/preprocessing/train_preprocessed.csv
outputs/preprocessing/final_train.csv
outputs/preprocessing/final_test.csv
outputs/preprocessing/labels_corrected.csv
outputs/preprocessing/label_coco_audit.csv
outputs/preprocessing/label_mismatches.csv
outputs/preprocessing/train_roi.csv
outputs/preprocessing/test_roi.csv
outputs/preprocessing/preprocessing_summary.json
```

Current audit result:

```text
Label mismatches: 0
Train ROI missing: 0
Unknown categories: []
```

So the current `train.csv` labels are consistent with the official COCO
category and area-threshold rules. The preprocessing artifacts are still useful
because they make ROI, bottle type, category presence, and label provenance
explicit for analysis and training. From now on, downstream final training
should read `outputs/preprocessing/final_train.csv`, where `target` is the
COCO-derived label.

## Kaggle

The Kaggle access token is stored outside this project:

```text
~/.kaggle/access_token
```

Do not commit tokens or paste them into project files.

Download or refresh the competition data:

```bash
scripts/kaggle_download.sh
```

Submit a CSV:

```bash
scripts/kaggle_submit.sh outputs/leaderboard_ensemble/submission_ensemble.csv "convnext maxvit teacher ensemble"
```

The Kaggle MCP submission tools are not exposed in the current Codex session, so
CLI submission is the ready fallback.

## Outputs

Training artifacts are written under:

```text
outputs/v26_maxvit/
outputs/v26_convnext_small/
outputs/lb_convnext_small/
outputs/lb_maxvit/
outputs/final_effnetv2_s/
outputs/ensemble/
outputs/leaderboard_ensemble/
```

Expected artifacts include model checkpoints, OOF predictions, teacher
predictions, and submission CSVs depending on which notebook cells are run.

## Useful Commands

Check setup:

```bash
source .venv/bin/activate
python scripts/check_setup.py
```

Regenerate the Mac notebook from the original Downloads notebook:

```bash
source .venv/bin/activate
python scripts/make_mac_notebook.py
```

Regenerate all ensemble notebooks:

```bash
source .venv/bin/activate
python scripts/make_ensemble_notebooks.py
```

Regenerate leaderboard notebooks:

```bash
source .venv/bin/activate
python scripts/make_leaderboard_notebooks.py
```

Smoke-test ensemble backbones:

```bash
source .venv/bin/activate
python scripts/smoke_ensemble_models.py
```

Blend trained ensemble predictions:

```bash
source .venv/bin/activate
python scripts/ensemble_submit.py
```

Blend leaderboard predictions with OOF weight and per-bottle threshold tuning:

```bash
source .venv/bin/activate
python scripts/ensemble_submit.py \
  --config outputs/leaderboard_models.json \
  --out-dir outputs/leaderboard_ensemble \
  --optimize-weights \
  --per-bottle-thresholds
```

Regenerate preprocessing artifacts:

```bash
source .venv/bin/activate
python scripts/preprocess_coco.py
```

Build teacher soft labels for the final student:

```bash
source .venv/bin/activate
python scripts/make_teacher_soft_labels.py
```

Train and export the final EfficientNetV2-S ONNX:

```bash
source .venv/bin/activate
python scripts/train_final_student.py --epochs 8 --batch-size 16 --num-workers 2
```

Monitor Kaggle teacher jobs:

```bash
source .venv/bin/activate
python scripts/kaggle_monitor.py --watch --interval 300
```

List Kaggle competition files:

```bash
source .venv/bin/activate
kaggle competitions files -c 1st-krones-vision-ai-challenge
```
