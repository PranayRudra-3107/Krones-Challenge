import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "kaggle" / "gpu_chunk"
COMPETITION = "1st-krones-vision-ai-challenge"

MODELS = {
    "v26_convnext_small": {
        "notebook": ROOT / "notebooks" / "v26_convnext_small_mac.ipynb",
        "label": "ConvNeXt Small",
        "batch_size": 12,
        "grad_accum": 3,
    },
    "v26_maxvit": {
        "notebook": ROOT / "notebooks" / "v26_maxvit_mac.ipynb",
        "label": "MaxViT Tiny",
        "batch_size": 12,
        "grad_accum": 3,
    },
    "lb_convnext_small": {
        "notebook": ROOT / "notebooks" / "lb_convnext_small_mac.ipynb",
        "label": "ConvNeXt Small",
        "batch_size": 12,
        "grad_accum": 3,
    },
    "lb_maxvit": {
        "notebook": ROOT / "notebooks" / "lb_maxvit_mac.ipynb",
        "label": "MaxViT Tiny",
        "batch_size": 12,
        "grad_accum": 3,
    },
}


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.rstrip().splitlines()]


def set_cell_source(cell: dict, text: str) -> None:
    cell["source"] = source_lines(text)
    cell["execution_count"] = None
    cell["outputs"] = []


def patch_text_cell(cell: dict, replacements: dict[str, str]) -> None:
    text = "".join(cell.get("source", []))
    for old, new in replacements.items():
        text = text.replace(old, new)
    set_cell_source(cell, text)


def replace_cell(nb: dict, marker: str, text: str) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code" and marker in "".join(cell.get("source", [])):
            set_cell_source(cell, text)
            return
    raise RuntimeError(f"Could not find cell marker: {marker}")


def dependency_cell(required_gpu_name: str | None) -> str:
    text = """
# =========================================================
# CELL 1: Kaggle dependency check
# =========================================================
import importlib.util
import os
import subprocess
import sys

REQUIRED_GPU_NAME = __REQUIRED_GPU_NAME__

def kaggle_gpu_compute_capability():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=name,compute_cap',
                '--format=csv,noheader',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(',')]
    if len(parts) < 2:
        return None
    try:
        return parts[0], float(parts[1])
    except ValueError:
        return parts[0], None


gpu_info = kaggle_gpu_compute_capability()
if gpu_info:
    gpu_name, compute_cap = gpu_info
    print(f'Kaggle GPU: {gpu_name} | compute capability: {compute_cap}')
    if REQUIRED_GPU_NAME and REQUIRED_GPU_NAME.lower() not in gpu_name.lower():
        raise RuntimeError(
            f'Requested GPU containing {REQUIRED_GPU_NAME!r}, but Kaggle assigned {gpu_name!r}. '
            'Stop early and rerun to avoid wasting a long training job on the wrong accelerator.'
        )
    # Kaggle can assign P100/Pascal. CUDA 12.8 PyTorch wheels do not support
    # Pascal, so install the CUDA 12.6 wheel before torch is imported.
    if compute_cap is not None and compute_cap < 7.0:
        print('Installing pinned CUDA 12.6 PyTorch wheel for Pascal/P100 compatibility.')
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
            'torch==2.8.0+cu126', 'torchvision==0.23.0+cu126',
            '--index-url', 'https://download.pytorch.org/whl/cu126',
        ])
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
            '--force-reinstall', '--no-deps', 'Pillow==11.3.0',
        ])

module_to_package = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'timm': 'timm',
    'albumentations': 'albumentations',
    'cv2': 'opencv-python-headless',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'tqdm': 'tqdm',
    'cleanlab': 'cleanlab',
}

missing_packages = [
    package for module, package in module_to_package.items()
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    print('Installing missing packages:', missing_packages)
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', '--no-input',
        *missing_packages,
    ])

missing_after = [
    module for module in module_to_package
    if importlib.util.find_spec(module) is None
]
if missing_after:
    raise ImportError('Missing packages after install: ' + ', '.join(missing_after))

print('Kaggle dependency check done.')
"""
    return text.replace("__REQUIRED_GPU_NAME__", repr(required_gpu_name))


def kernel_slug(model_tag: str, phase: str, folds: list[int]) -> str:
    folds_part = "-".join(str(fold) for fold in folds) if folds else "all"
    return f"krones-{model_tag.replace('_', '-')}-{phase}-f{folds_part}"


def kaggle_data_patch() -> dict[str, str]:
    return {
        "PROJECT_ROOT = find_project_root()": (
            "if Path('/kaggle/working').exists():\n"
            "    PROJECT_ROOT = Path('/kaggle/working')\n"
            "else:\n"
            "    PROJECT_ROOT = find_project_root()"
        ),
        "DATA_CANDIDATES = [\n    PROJECT_ROOT / 'data' / '1st-krones-vision-ai-challenge',": (
            "DATA_CANDIDATES = [\n"
            "    Path('/kaggle/input/1st-krones-vision-ai-challenge'),\n"
            "    PROJECT_ROOT / 'data' / '1st-krones-vision-ai-challenge',"
        ),
        """if DATA is None:
    raise FileNotFoundError(
        'Dataset not found. Extract it to '
        f"{PROJECT_ROOT / 'data' / '1st-krones-vision-ai-challenge'}"
    )""": """if DATA is None and Path('/kaggle/input').exists():
    for root, dirs, files in os.walk('/kaggle/input'):
        cand = Path(root)
        if 'train.csv' in files and 'train_annotations.json' in files:
            DATA = cand
            break
if DATA is None:
    raise FileNotFoundError(
        'Dataset not found. Could not find train.csv and train_annotations.json under /kaggle/input.'
    )""",
        "BATCH_SIZE          = 4": "BATCH_SIZE          = {batch_size}",
        "GRAD_ACCUM_STEPS    = 8": "GRAD_ACCUM_STEPS    = {grad_accum}",
        "NUM_WORKERS    = 0": "NUM_WORKERS    = 2",
        """BTYPES_PATH = DATA / 'bottletypes.csv'
if not BTYPES_PATH.exists():
    for cand in [PROJECT_ROOT / 'data' / 'bottletypes' / 'bottletypes.csv']:
        if cand.exists(): BTYPES_PATH = cand; break
assert BTYPES_PATH.exists(), 'bottletypes.csv not found'""": """BTYPES_PATH = DATA / 'bottletypes.csv'
if not BTYPES_PATH.exists() and Path('/kaggle/input').exists():
    for cand in Path('/kaggle/input').rglob('bottletypes.csv'):
        if cand.exists():
            BTYPES_PATH = cand
            break
if not BTYPES_PATH.exists():
    for cand in [PROJECT_ROOT / 'data' / 'bottletypes' / 'bottletypes.csv']:
        if cand.exists(): BTYPES_PATH = cand; break
assert BTYPES_PATH.exists(), 'bottletypes.csv not found'""",
        """test_ann_path = None
for cand in [DATA/'test_annotations_roi_only.json', DATA/'test_annotations.json']:
    if cand.exists(): test_ann_path = cand; break""": """test_ann_path = None
for cand in [DATA/'test_annotations_roi_only.json', DATA/'test_annotations.json']:
    if cand.exists(): test_ann_path = cand; break
if test_ann_path is None and Path('/kaggle/input').exists():
    for cand in Path('/kaggle/input').rglob('test_annotations_roi_only.json'):
        if cand.exists():
            test_ann_path = cand
            break""",
    }


def controls_cell(args: argparse.Namespace, dataset_sources: list[str]) -> str:
    previous_inputs = list(args.previous_input or [])
    for source in args.kernel_source or []:
        slug = source.split("/")[-1]
        path = f"/kaggle/input/{slug}"
        if path not in previous_inputs:
            previous_inputs.append(path)

    run_export = args.export
    if run_export is None:
        run_export = args.phase in {"p2", "export", "both"}

    return f"""
# =========================================================
# KAGGLE CHUNK CONTROLS - generated locally
# =========================================================
RUN_PHASE = {args.phase!r}
RUN_FOLDS = {args.folds!r}
RUN_EXPORT = {bool(run_export)!r}
PREVIOUS_OUTPUT_INPUTS = {previous_inputs!r}

print("Kaggle chunk controls")
print("  VERSION_TAG:", VERSION_TAG)
print("  RUN_PHASE:", RUN_PHASE)
print("  RUN_FOLDS:", RUN_FOLDS)
print("  RUN_EXPORT:", RUN_EXPORT)
print("  SAVE:", SAVE)
print("  Previous checkpoint inputs:", PREVIOUS_OUTPUT_INPUTS or "none")

def copy_previous_outputs():
    import shutil

    patterns = [
        f"model_{{VERSION_TAG}}_*.pth",
        f"oof_{{VERSION_TAG}}_*.npy",
        f"oof_idx_{{VERSION_TAG}}_*.npy",
        f"oof_bt_{{VERSION_TAG}}_*.npy",
        f"test_{{VERSION_TAG}}_*.npy",
        f"labels_*_{{VERSION_TAG}}.csv",
        f"*indices_{{VERSION_TAG}}.npy",
    ]
    candidate_bases = []
    copied = 0
    for raw_base in PREVIOUS_OUTPUT_INPUTS:
        base = Path(raw_base)
        if not base.exists():
            print(f"  resume input missing: {{base}}")
            continue
        candidate_bases.append(base)
    if Path('/kaggle/input').exists():
        print('  /kaggle/input roots:')
        for child in sorted(Path('/kaggle/input').iterdir()):
            print(f'    {{child}}')
        candidate_bases.append(Path('/kaggle/input'))
    seen_sources = set()
    for base in candidate_bases:
        for pattern in patterns:
            for src in base.rglob(pattern):
                if not src.is_file():
                    continue
                if src in seen_sources:
                    continue
                seen_sources.add(src)
                dst = SAVE / src.name
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
                shutil.copy2(src, dst)
                copied += 1
                print(f"  copied {{src}} -> {{dst}}")
    print(f"Resume copy complete: {{copied}} files copied")

copy_previous_outputs()
"""


def p1_cell() -> str:
    return """
# =========================================================
# CELL 17: PHASE 1 - selected Kaggle chunk folds
# =========================================================
if RUN_PHASE in {"p1", "both"}:
    print(f"Running P1 folds: {RUN_FOLDS}")
    for fold in RUN_FOLDS:
        if fold < 0 or fold >= N_FOLDS:
            raise ValueError(f"Invalid fold {fold}; expected 0..{N_FOLDS - 1}")
        if fold_completed('p1', fold):
            print(f'p1 fold {fold} done - skip'); continue
        tr_idx, va_idx = SKF_SPLITS[fold]
        df_tr = labels_train_active.iloc[tr_idx].reset_index(drop=True)
        df_va = labels_train_active.iloc[va_idx].copy(); df_va.index = va_idx
        res = train_fold('p1', fold, df_tr, df_va,
                         P1_EPOCHS, LR_P1,
                         init_state=None, use_swa=True, use_recal_bn=True,
                         use_mix=True, curriculum='bucket_a_only',
                         drop_path_rate=DROP_PATH_RATE_P1)
        del res; gc.collect(); empty_device_cache()
    print('Kaggle chunk P1 complete')
else:
    print(f'Skipping P1 because RUN_PHASE={RUN_PHASE}')
"""


def p1_backup_cell() -> str:
    return """
# =========================================================
# CELL 18: P1 backup submission - skipped in chunk mode
# =========================================================
print('P1 backup submission skipped in Kaggle chunk mode to save GPU time.')
"""


def noise_cleaning_cell() -> str:
    return """
# =========================================================
# CELL 19: Model-based noise cleaning at 0.95 confidence
# =========================================================
if RUN_PHASE in {"p2", "both", "export"}:
    p1_oof = np.full(len(labels_train_active), np.nan, dtype=np.float32)
    missing_oof = []
    for fold in range(N_FOLDS):
        p = SAVE / f'oof_{VERSION_TAG}_p1_fold{fold}.npy'
        i = SAVE / f'oof_idx_{VERSION_TAG}_p1_fold{fold}.npy'
        if p.exists() and i.exists():
            idx = np.load(i); p1_oof[idx] = np.load(p)
        else:
            missing_oof.append(fold)
    if missing_oof:
        print(f'Warning: missing P1 OOF folds for noise cleaning: {missing_oof}')

    valid = ~np.isnan(p1_oof)
    y_curr = labels_train_active['target'].values
    noisy  = np.zeros(len(labels_train_active), dtype=bool)
    noisy[(y_curr == 0) & (p1_oof >= NOISE_CONF_THRESH) & valid] = True
    noisy[(y_curr == 1) & (p1_oof <= 1-NOISE_CONF_THRESH) & valid] = True
    n_noisy = noisy.sum()
    print(f'Noise cleaning at conf>={NOISE_CONF_THRESH}: {n_noisy} samples removed')

    print('Per-type noise:')
    for tname, tid in BOTTLE_TYPE_CANONICAL.items():
        tm = labels_train_active['btype_id'].values == tid
        nb = (noisy & tm).sum()
        print(f'  {tname}: {nb}/{tm.sum()}')

    labels_clean = labels_train_active.loc[~noisy].reset_index(drop=True)
    labels_clean.to_csv(SAVE / f'labels_clean_{VERSION_TAG}.csv', index=False)
    clean_stems_set = set(labels_clean['stem'])
    print(f'labels_clean: {len(labels_clean)} (dropped {n_noisy})')
else:
    labels_clean = labels_train_active.copy().reset_index(drop=True)
    clean_stems_set = set(labels_clean['stem'])
    print(f'Skipping noise cleaning because RUN_PHASE={RUN_PHASE}')
"""


def p2_cell() -> str:
    return """
# =========================================================
# CELL 20: PHASE 2 - selected Kaggle chunk folds
# =========================================================
if RUN_PHASE in {"p2", "both"}:
    print(f"Running P2 folds: {RUN_FOLDS}")
    for fold in RUN_FOLDS:
        if fold < 0 or fold >= N_FOLDS:
            raise ValueError(f"Invalid fold {fold}; expected 0..{N_FOLDS - 1}")
        if fold_completed('p2', fold):
            print(f'p2 fold {fold} done - skip'); continue
        p1_path = SAVE / f'model_{VERSION_TAG}_p1_fold{fold}.pth'
        if not p1_path.exists():
            raise FileNotFoundError(f'Missing P1 checkpoint for P2 fold {fold}: {p1_path}')
        tr_idx, va_idx = SKF_SPLITS[fold]
        df_va = labels_train_active.iloc[va_idx].copy(); df_va.index = va_idx
        df_tr_full = labels_train_active.iloc[tr_idx].reset_index(drop=True)
        df_tr = df_tr_full[df_tr_full.stem.isin(clean_stems_set)].reset_index(drop=True)

        init = torch.load(p1_path, map_location=PRIMARY_DEVICE, weights_only=True)
        res = train_fold('p2', fold, df_tr, df_va,
                         P2_EPOCHS, LR_P2,
                         init_state=init, use_swa=True, use_recal_bn=True,
                         use_mix=True, curriculum='bucket_ab',
                         drop_path_rate=DROP_PATH_RATE_P1)
        del res, init; gc.collect(); empty_device_cache()
    print('Kaggle chunk P2 complete')
else:
    print(f'Skipping P2 because RUN_PHASE={RUN_PHASE}')
"""


def export_cell() -> str:
    return """
# =========================================================
# CELL 21: TEACHER EXPORT - selected/available P2 predictions
# =========================================================
if RUN_EXPORT:
    print('='*60)
    print('ENSEMBLE MEMBER EXPORT - P2 test predictions')
    print('='*60)

    test_df = pd.DataFrame({'image_id': test_ids})
    export_folds = RUN_FOLDS if RUN_PHASE in {"p2", "both"} else list(range(N_FOLDS))

    for fold in export_folds:
        p = SAVE / f'model_{VERSION_TAG}_p2_fold{fold}.pth'
        if not p.exists():
            print(f'  fold {fold}: P2 weights missing - skip'); continue
        out_p = SAVE / f'test_{VERSION_TAG}_p2_fold{fold}.npy'
        if out_p.exists():
            print(f'  fold {fold}: prediction already exists - skip')
            continue
        m = BottleClsMT(pretrained=False).to(PRIMARY_DEVICE)
        m.load_state_dict(torch.load(p, map_location=PRIMARY_DEVICE, weights_only=True), strict=False)
        m.eval()
        probs, bts = predict_test_tta(m, test_df, TEST_IMG, TEST_ROI)
        np.save(out_p, probs)
        np.save(SAVE / f'test_{VERSION_TAG}_btype.npy', bts)
        print(f'  fold {fold}: saved {out_p.name} (mean prob={probs.mean():.4f})')
        del m; gc.collect(); empty_device_cache()

    fold_test_preds = []
    for fold in range(N_FOLDS):
        fold_p = SAVE / f'test_{VERSION_TAG}_p2_fold{fold}.npy'
        if fold_p.exists():
            fold_test_preds.append(np.load(fold_p))

    if fold_test_preds:
        mean_pred = np.mean(np.stack(fold_test_preds), axis=0)
        np.save(SAVE / f'test_{VERSION_TAG}_p2_mean.npy', mean_pred)
        print(f'Saved fold-averaged test_{VERSION_TAG}_p2_mean.npy '
              f'(mean prob={mean_pred.mean():.4f}, n_folds={len(fold_test_preds)})')

    p2_oof = np.full(len(labels_train_active), np.nan)
    for fold in range(N_FOLDS):
        op = SAVE / f'oof_{VERSION_TAG}_p2_fold{fold}.npy'
        oi = SAVE / f'oof_idx_{VERSION_TAG}_p2_fold{fold}.npy'
        if op.exists() and oi.exists():
            idx = np.load(oi); p2_oof[idx] = np.load(op)
    valid = ~np.isnan(p2_oof)
    if valid.sum() > 0:
        y_active = labels_train_active['target'].values
        t_oof, f_oof = best_threshold(y_active[valid], p2_oof[valid])
        print(f'P2 OOF F1: {f_oof:.5f} @ t={t_oof:.3f} '
              f'({valid.sum()}/{len(labels_train_active)} samples)')
else:
    print('Teacher export skipped because RUN_EXPORT=False')

manifest = {
    'version_tag': VERSION_TAG,
    'run_phase': RUN_PHASE,
    'run_folds': RUN_FOLDS,
    'run_export': RUN_EXPORT,
    'artifacts': sorted(p.name for p in SAVE.glob('*') if p.is_file()),
}
(SAVE / f'kaggle_chunk_manifest_{VERSION_TAG}.json').write_text(json.dumps(manifest, indent=2))
print(f'Wrote manifest: {SAVE / f"kaggle_chunk_manifest_{VERSION_TAG}.json"}')
"""


def build_notebook(args: argparse.Namespace, model: dict) -> dict:
    nb = json.loads(model["notebook"].read_text())

    replacements = kaggle_data_patch()
    replacements = {
        key: value.format(batch_size=args.batch_size, grad_accum=args.grad_accum)
        for key, value in replacements.items()
    }

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            patch_text_cell(cell, replacements)
        else:
            cell["metadata"] = cell.get("metadata", {})

    config_index = None
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and "# CELL 3:" in "".join(cell.get("source", [])):
            config_index = index
            break
    if config_index is None:
        raise RuntimeError("Could not find config cell")

    dataset_sources = ([] if args.no_competition_source else [COMPETITION]) + list(args.dataset_source or [])
    nb["cells"].insert(
        config_index + 1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines(controls_cell(args, dataset_sources)),
        },
    )

    replace_cell(nb, "# CELL 17:", p1_cell())
    replace_cell(nb, "# CELL 1:", dependency_cell(args.require_gpu_name))
    replace_cell(nb, "# CELL 18:", p1_backup_cell())
    replace_cell(nb, "# CELL 19:", noise_cleaning_cell())
    replace_cell(nb, "# CELL 20:", p2_cell())
    replace_cell(nb, "# CELL 21:", export_cell())

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nb.setdefault("metadata", {})
    nb["metadata"]["kaggle_chunk"] = {
        "model": args.model,
        "phase": args.phase,
        "folds": args.folds,
    }
    return nb


def write_metadata(args: argparse.Namespace, model: dict, notebook_name: str) -> dict:
    owner = args.kernel_owner or os.environ.get("KAGGLE_KERNEL_OWNER", "dronakp")
    slug = args.kernel_slug or kernel_slug(args.model, args.phase, args.folds)
    dataset_sources = args.dataset_source or []
    competition_sources = [] if args.no_competition_source else [COMPETITION]
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": args.accelerator,
        "dataset_sources": dataset_sources,
        "competition_sources": competition_sources,
        "kernel_sources": args.kernel_source or [],
        "model_sources": [],
    }
    (PACKAGE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_folds(value: str) -> list[int]:
    folds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise argparse.ArgumentTypeError("At least one fold is required")
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Kaggle GPU chunk notebook.")
    parser.add_argument("--model", choices=sorted(MODELS), default="lb_convnext_small")
    parser.add_argument("--phase", choices=["p1", "p2", "both", "export"], default="p1")
    parser.add_argument("--folds", type=parse_folds, default=[0], help="Comma-separated folds, e.g. 0 or 0,1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--previous-input", action="append", help="Kaggle input path with previous checkpoints.")
    parser.add_argument("--dataset-source", action="append", help="Extra Kaggle dataset source slug owner/dataset.")
    parser.add_argument(
        "--no-competition-source",
        action="store_true",
        help="Do not attach the original competition input; use dataset sources only.",
    )
    parser.add_argument("--kernel-source", action="append", help="Previous Kaggle kernel output source slug owner/kernel.")
    parser.add_argument("--accelerator", default=os.environ.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4"))
    parser.add_argument("--require-gpu-name", default=None, help="Exit early unless assigned GPU name contains this text, e.g. T4.")
    parser.add_argument("--export", action="store_true", default=None, help="Export test predictions after training.")
    parser.add_argument("--no-export", action="store_false", dest="export", help="Do not export test predictions.")
    parser.add_argument("--kernel-owner", help="Kaggle username/owner for kernel metadata.")
    parser.add_argument("--kernel-slug", help="Override Kaggle kernel slug.")
    args = parser.parse_args()

    model = MODELS[args.model]
    if args.batch_size is None:
        args.batch_size = model["batch_size"]
    if args.grad_accum is None:
        args.grad_accum = model["grad_accum"]

    if args.phase == "export":
        args.folds = list(range(5))

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    notebook_name = "krones_gpu_chunk.ipynb"
    nb = build_notebook(args, model)
    (PACKAGE_DIR / notebook_name).write_text(json.dumps(nb, indent=1) + "\n")
    metadata = write_metadata(args, model, notebook_name)

    print(f"Wrote {PACKAGE_DIR / notebook_name}")
    print(f"Wrote {PACKAGE_DIR / 'kernel-metadata.json'}")
    print(f"Kernel id: {metadata['id']}")
    print("Push with:")
    print(f"  .venv/bin/kaggle kernels push -p {PACKAGE_DIR} --accelerator {args.accelerator}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
