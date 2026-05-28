#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPETITION="${KAGGLE_COMPETITION:-1st-krones-vision-ai-challenge}"
SUBMISSION_FILE="${1:-}"
MESSAGE="${2:-local submission}"
KAGGLE_BIN="$ROOT/.venv/bin/kaggle"

if [[ -z "$SUBMISSION_FILE" ]]; then
  echo "Usage: scripts/kaggle_submit.sh path/to/submission.csv \"message\"" >&2
  exit 1
fi

if [[ ! -f "$SUBMISSION_FILE" ]]; then
  echo "Submission file not found: $SUBMISSION_FILE" >&2
  exit 1
fi

if [[ ! -x "$KAGGLE_BIN" ]]; then
  echo "Kaggle CLI not found at $KAGGLE_BIN. Run: uv pip install -r requirements-mac.txt" >&2
  exit 1
fi

"$KAGGLE_BIN" competitions submit -c "$COMPETITION" -f "$SUBMISSION_FILE" -m "$MESSAGE"

