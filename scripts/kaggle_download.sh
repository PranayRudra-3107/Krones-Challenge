#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPETITION="${KAGGLE_COMPETITION:-1st-krones-vision-ai-challenge}"
DATA_DIR="${KAGGLE_DATA_DIR:-$ROOT/data}"
KAGGLE_BIN="$ROOT/.venv/bin/kaggle"

if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Kaggle CLI not found at $KAGGLE_BIN. Run: uv pip install -r requirements-mac.txt" >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
"$KAGGLE_BIN" competitions download -c "$COMPETITION" -p "$DATA_DIR"

shopt -s nullglob
for zipfile in "$DATA_DIR"/*.zip; do
  unzip -n "$zipfile" -d "$DATA_DIR"
done

echo "Dataset ready at: $DATA_DIR"
