#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
KAGGLE_BIN="$ROOT/.venv/bin/kaggle"
KAGGLE_ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python at $PYTHON" >&2
  exit 1
fi

if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Missing Kaggle CLI at $KAGGLE_BIN" >&2
  exit 1
fi

"$PYTHON" "$ROOT/scripts/make_kaggle_chunk_notebook.py" "$@"

echo
echo "Prepared Kaggle chunk package:"
echo "  $ROOT/kaggle/gpu_chunk"
echo
echo "This script does not start a GPU job by default."
echo "To start the Kaggle GPU run:"
echo "  $KAGGLE_BIN kernels push -p \"$ROOT/kaggle/gpu_chunk\" --accelerator $KAGGLE_ACCELERATOR"
echo
echo "To push immediately, rerun with:"
echo "  KAGGLE_PUSH=1 $0 $*"

if [[ "${KAGGLE_PUSH:-0}" == "1" ]]; then
  "$KAGGLE_BIN" kernels push -p "$ROOT/kaggle/gpu_chunk" --accelerator "$KAGGLE_ACCELERATOR"
fi
