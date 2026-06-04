#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/final_effnetv2_s_lb_5fold_mps}"
TRAIN_CSV="${TRAIN_CSV:-outputs/final_effnetv2_s_lb/teacher_soft_train.csv}"
if [[ "${WARM_START+x}" != "x" ]]; then
  WARM_START="outputs/final_effnetv2_s_lb_mps/final_effnetv2_s_best.pth"
fi
EPOCHS="${EPOCHS:-5}"
IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
LR="${LR:-5e-5}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.75}"
BTYPE_WEIGHT="${BTYPE_WEIGHT:-0.10}"
N_FOLDS="${N_FOLDS:-5}"

mkdir -p "$OUT_ROOT"

HAVE_CAFFEINATE=0
if command -v caffeinate >/dev/null 2>&1; then
  HAVE_CAFFEINATE=1
fi

run_keep_awake() {
  if [[ "$HAVE_CAFFEINATE" == "1" ]]; then
    PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -dimsu "$@"
  else
    PYTORCH_ENABLE_MPS_FALLBACK=1 "$@"
  fi
}

for fold in $(seq 0 $((N_FOLDS - 1))); do
  fold_dir="$OUT_ROOT/fold${fold}"
  mkdir -p "$fold_dir"
  echo
  echo "===== EfficientNetV2-S LB fold ${fold}/${N_FOLDS} ====="
  if [[ -n "$WARM_START" ]]; then
    run_keep_awake "$PYTHON" -u scripts/train_final_student.py \
      --device mps \
      --train-csv "$TRAIN_CSV" \
      --warm-start-checkpoint "$WARM_START" \
      --out-dir "$fold_dir" \
      --onnx-path "$fold_dir/final_effnetv2_s.onnx" \
      --fold "$fold" \
      --n-folds "$N_FOLDS" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --infer-batch-size "$INFER_BATCH_SIZE" \
      --num-workers 0 \
      --img-size "$IMG_SIZE" \
      --lr "$LR" \
      --distill-alpha "$DISTILL_ALPHA" \
      --btype-weight "$BTYPE_WEIGHT" \
      2>&1 | tee "$fold_dir/train_mps.log"
  else
    run_keep_awake "$PYTHON" -u scripts/train_final_student.py \
      --device mps \
      --train-csv "$TRAIN_CSV" \
      --out-dir "$fold_dir" \
      --onnx-path "$fold_dir/final_effnetv2_s.onnx" \
      --fold "$fold" \
      --n-folds "$N_FOLDS" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --infer-batch-size "$INFER_BATCH_SIZE" \
      --num-workers 0 \
      --img-size "$IMG_SIZE" \
      --lr "$LR" \
      --distill-alpha "$DISTILL_ALPHA" \
      --btype-weight "$BTYPE_WEIGHT" \
      2>&1 | tee "$fold_dir/train_mps.log"
  fi
done

echo
echo "===== Averaging fold ONNX predictions ====="
"$PYTHON" scripts/ensemble_student_onnx.py \
  --model-paths \
    "$OUT_ROOT/fold0/final_effnetv2_s.onnx" \
    "$OUT_ROOT/fold1/final_effnetv2_s.onnx" \
    "$OUT_ROOT/fold2/final_effnetv2_s.onnx" \
    "$OUT_ROOT/fold3/final_effnetv2_s.onnx" \
    "$OUT_ROOT/fold4/final_effnetv2_s.onnx" \
  --out-dir "$OUT_ROOT/ensemble" \
  --thresholds 0.300 0.320 0.340 0.350 0.360 0.370 0.380 0.390 0.405 0.430 0.450 \
  --batch-size "$INFER_BATCH_SIZE" \
  --img-size "$IMG_SIZE"

echo
echo "Done. Ensemble submissions are in: $OUT_ROOT/ensemble"
