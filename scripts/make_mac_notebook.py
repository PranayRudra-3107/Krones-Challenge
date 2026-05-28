import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/Users/pranayrudra/Downloads/v26_maxvit.ipynb")
OUT = ROOT / "notebooks" / "v26_maxvit_mac.ipynb"


def set_cell_source(cell, source: str) -> None:
    cell["source"] = [line + "\n" for line in source.splitlines()]


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected notebook text was not found:\n{old[:200]}")
    return text.replace(old, new)


def main() -> int:
    if not SRC.exists():
        raise FileNotFoundError(f"Source notebook not found: {SRC}")

    nb = json.loads(SRC.read_text())

    # Add a local-note markdown cell after the original title cell.
    nb["cells"].insert(
        1,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Local Mac Runtime Notes\n",
                "\n",
                "This copy was generated for Apple Silicon local runs. It keeps the competition logic intact while using project-relative paths, `mps` device detection, and Mac-friendly memory defaults.\n",
            ],
        },
    )

    # Cell numbers shifted by the inserted markdown cell.
    install_cell = nb["cells"][2]
    imports_cell = nb["cells"][3]
    config_cell = nb["cells"][4]

    set_cell_source(
        install_cell,
        """# =========================================================
# CELL 1: Local dependency check
# =========================================================
# Install packages from the terminal before running this notebook:
#   source .venv/bin/activate
#   uv pip install -r requirements-mac.txt

import importlib.util

required_modules = [
    'torch', 'torchvision', 'timm', 'albumentations', 'cv2',
    'numpy', 'pandas', 'sklearn', 'scipy', 'tqdm', 'cleanlab'
]

missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    raise ImportError(
        'Missing packages: ' + ', '.join(missing) +
        '. From the project root, run: uv pip install -r requirements-mac.txt'
    )

print('Local dependency check done.')""",
    )

    text = "".join(imports_cell["source"])
    text = replace_required(
        text,
        "import os, gc, json, time, random, warnings, math, copy\n",
        """import os
# Set before importing torch. MPS fallback lets unsupported ops run on CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

import gc, json, time, random, warnings, math, copy
""",
    )
    text = replace_required(
        text,
        """def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRIMARY_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch {torch.__version__} | timm {timm.__version__} | device {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
""",
        """def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

PRIMARY_DEVICE = DEVICE
AMP_ENABLED = DEVICE.type == 'cuda'
AMP_DEVICE_TYPE = 'cuda' if AMP_ENABLED else 'cpu'
PIN_MEMORY = DEVICE.type == 'cuda'

def amp_autocast(enabled=None):
    return torch.amp.autocast(
        device_type=AMP_DEVICE_TYPE,
        enabled=AMP_ENABLED if enabled is None else enabled,
    )

def make_grad_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

print(f'PyTorch {torch.__version__} | timm {timm.__version__} | device {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
elif DEVICE.type == 'mps':
    print('Apple Silicon MPS backend active')
else:
    print('CPU backend active')
""",
    )
    set_cell_source(imports_cell, text.rstrip())

    text = "".join(config_cell["source"])
    text = replace_required(
        text,
        """# ── Paths (auto-detected) ─────────────────────────────────────
DATA = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.csv' in files and 'train_annotations.json' in files:
        DATA = Path(root); break
if DATA is None:
    DATA = Path('/kaggle/input/1st-krones-vision-ai-challenge')
print(f'DATA: {DATA}')
SAVE = Path('/kaggle/working'); SAVE.mkdir(exist_ok=True)
""",
        """# ── Local paths ───────────────────────────────────────────────
def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for cand in candidates:
        if (cand / 'requirements-mac.txt').exists() and (cand / 'notebooks').exists():
            return cand
    return start

PROJECT_ROOT = find_project_root()
DATA_CANDIDATES = [
    PROJECT_ROOT / 'data' / '1st-krones-vision-ai-challenge',
    PROJECT_ROOT / 'data',
    PROJECT_ROOT,
    Path.home() / 'Downloads' / '1st-krones-vision-ai-challenge',
]
DATA = None
for cand in DATA_CANDIDATES:
    if (cand / 'train.csv').exists() and (cand / 'train_annotations.json').exists():
        DATA = cand
        break
if DATA is None:
    raise FileNotFoundError(
        'Dataset not found. Extract it to '
        f\"{PROJECT_ROOT / 'data' / '1st-krones-vision-ai-challenge'}\"
    )
print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'DATA: {DATA}')
SAVE = PROJECT_ROOT / 'outputs' / VERSION_TAG
SAVE.mkdir(parents=True, exist_ok=True)
""",
    )
    text = replace_required(
        text,
        """# ── T4 memory budget ──────────────────────────────────────────
# MaxViT-Tiny @ 384 is attention-heavy → BS=12 with grad accum, grad checkpoint ON.
BATCH_SIZE          = 16     # 256px MaxViT-Tiny fits BS=16 on T4 w/ grad checkpoint
GRAD_ACCUM_STEPS    = 2      # effective batch = 16×2 = 32 (matches B4)
USE_GRAD_CHECKPOINT = True   # essential for MaxViT on T4
""",
        """# ── Local Mac memory budget ───────────────────────────────────
# Start conservatively on 16 GB unified memory. Raise BATCH_SIZE if stable.
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 8      # effective batch = 32 (matches original)
USE_GRAD_CHECKPOINT = True
""",
    )
    text = text.replace("NUM_WORKERS    = 4", "NUM_WORKERS    = 0          # safer inside local Jupyter on macOS")
    set_cell_source(config_cell, text.rstrip())

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"])
        text = text.replace(
            "for cand in [Path('/kaggle/input/bottletypes/bottletypes.csv')]:",
            "for cand in [PROJECT_ROOT / 'data' / 'bottletypes' / 'bottletypes.csv']:",
        )
        text = text.replace("os.walk('/kaggle/input')", "os.walk(PROJECT_ROOT / 'outputs')")
        text = text.replace("torch.cuda.amp.autocast(enabled=True)", "amp_autocast()")
        text = text.replace("torch.cuda.amp.autocast(enabled=False)", "amp_autocast(enabled=False)")
        text = text.replace("torch.cuda.amp.GradScaler()", "make_grad_scaler()")
        text = text.replace("torch.cuda.empty_cache()", "empty_device_cache()")
        text = text.replace("pin_memory=True", "pin_memory=PIN_MEMORY")
        set_cell_source(cell, text.rstrip())

    # Patch SWA averaging so it never tries float64 math on MPS tensors.
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"])
        if "def average_state_dicts(states):" not in text:
            continue
        old = """def average_state_dicts(states):
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if not torch.is_floating_point(avg[k]): continue
        acc = torch.zeros_like(avg[k], dtype=torch.float64)
        for s in states: acc += s[k].double()
        avg[k] = (acc / len(states)).to(avg[k].dtype)
    return avg
"""
        new = """def average_state_dicts(states):
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        if not torch.is_floating_point(avg[k]):
            continue
        target_device = avg[k].device
        target_dtype = avg[k].dtype
        acc = torch.zeros_like(avg[k].detach().cpu(), dtype=torch.float64)
        for s in states:
            acc += s[k].detach().cpu().double()
        avg[k] = (acc / len(states)).to(dtype=target_dtype, device=target_device)
    return avg
"""
        text = replace_required(text, old, new)
        set_cell_source(cell, text.rstrip())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
