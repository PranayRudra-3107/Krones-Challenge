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
|-- notebooks/
|   `-- v26_maxvit_mac.ipynb
|-- outputs/
|-- scripts/
|   |-- check_setup.py
|   |-- kaggle_download.sh
|   |-- kaggle_submit.sh
|   `-- make_mac_notebook.py
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

Do not casually run the entire notebook unless you are ready for a long training
run. Start with the setup, config, audit, and ROI-analysis cells first.

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
scripts/kaggle_submit.sh outputs/v26_maxvit/submission.csv "local MaxViT submission"
```

The Kaggle MCP submission tools are not exposed in the current Codex session, so
CLI submission is the ready fallback.

## Outputs

Training artifacts are written under:

```text
outputs/v26_maxvit/
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

List Kaggle competition files:

```bash
source .venv/bin/activate
kaggle competitions files -c 1st-krones-vision-ai-challenge
```
