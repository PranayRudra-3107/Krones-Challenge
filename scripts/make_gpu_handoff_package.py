#!/usr/bin/env python3
"""Create a safe GPU run handoff package.

The package contains a generated Kaggle GPU chunk notebook plus instructions for
a teammate account that has joined the competition/team, or for a separate GPU
machine. It intentionally does not copy credentials, tokens, data, or outputs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_PACKAGE_DIR = ROOT / "kaggle" / "gpu_chunk"
HANDOFF_ROOT = ROOT / "handoff"


def parse_folds(value: str) -> list[int]:
    folds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            folds.append(int(part))
    if not folds:
        raise argparse.ArgumentTypeError("At least one fold is required")
    return folds


def slug_for(model: str, phase: str, folds: list[int], explicit_slug: str | None) -> str:
    if explicit_slug:
        return explicit_slug
    folds_part = "-".join(str(fold) for fold in folds)
    return f"krones-{model.replace('_', '-')}-{phase}-f{folds_part}"


def build_chunk(args: argparse.Namespace, slug: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_kaggle_chunk_notebook.py"),
        "--model",
        args.model,
        "--phase",
        args.phase,
        "--folds",
        ",".join(str(fold) for fold in args.folds),
        "--accelerator",
        args.accelerator,
        "--kernel-owner",
        args.kernel_owner,
        "--kernel-slug",
        slug,
    ]
    if args.require_gpu_name:
        cmd.extend(["--require-gpu-name", args.require_gpu_name])
    if args.no_competition_source:
        cmd.append("--no-competition-source")
    if args.no_export:
        cmd.append("--no-export")
    if args.export:
        cmd.append("--export")
    for source in args.kernel_source or []:
        cmd.extend(["--kernel-source", source])
    for source in args.dataset_source or []:
        cmd.extend(["--dataset-source", source])
    subprocess.check_call(cmd, cwd=ROOT)


def write_readme(
    out_dir: Path,
    *,
    model: str,
    phase: str,
    folds: list[int],
    kernel_owner: str,
    slug: str,
    accelerator: str,
) -> None:
    ref = f"{kernel_owner}/{slug}"
    folds_text = ",".join(str(fold) for fold in folds)
    readme = f"""# GPU Handoff Package

This package was generated from the Krones Challenge project for:

```text
model: {model}
phase: {phase}
folds: {folds_text}
kernel ref: {ref}
accelerator: {accelerator}
```

## Important Rule Boundary

Use this package only with one of these safe paths:

- a real teammate Kaggle account that has joined the Krones Challenge and the same Kaggle team;
- an external GPU machine where competition rules allow external compute.

Do not use a second personal Kaggle account to bypass quota or competition access.

This package is Kaggle-kernel-specific. For a non-Kaggle external GPU machine,
use the full project repository plus the local `data/` folder instead.

## Teammate Kaggle Run

On the teammate machine/account:

```bash
pip install kaggle
kaggle kernels push -p kaggle_gpu_chunk --accelerator {accelerator}
kaggle kernels status {ref}
```

After the run finishes:

```bash
mkdir -p downloaded_outputs
kaggle kernels output {ref} -p downloaded_outputs
```

Send back the downloaded output folder or archive.

## Pull Outputs Back Into This Mac Project

From this Mac project root:

```bash
cd "{ROOT}"
mkdir -p outputs/kaggle_runs/{slug}
# Put the returned files under outputs/kaggle_runs/{slug}/ first.
rsync -av outputs/kaggle_runs/{slug}/outputs/{model}/ outputs/{model}/
```

Then rebuild teacher soft labels when enough teacher OOF/test predictions exist:

```bash
source .venv/bin/activate
python scripts/make_teacher_soft_labels.py
```

## Package Contents

```text
kaggle_gpu_chunk/
  krones_gpu_chunk.ipynb
  kernel-metadata.json
```
"""
    (out_dir / "README_GPU_HANDOFF.md").write_text(readme)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a portable GPU handoff package.")
    parser.add_argument("--model", required=True, choices=[
        "v26_convnext_small",
        "v26_maxvit",
        "lb_convnext_small",
        "lb_maxvit",
    ])
    parser.add_argument("--phase", default="p1", choices=["p1", "p2", "both", "export"])
    parser.add_argument("--folds", type=parse_folds, default=[0])
    parser.add_argument("--kernel-owner", required=True, help="Teammate/external Kaggle username for metadata.")
    parser.add_argument("--kernel-slug", help="Override kernel slug.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--require-gpu-name", default="T4")
    parser.add_argument("--kernel-source", action="append")
    parser.add_argument("--dataset-source", action="append")
    parser.add_argument("--no-competition-source", action="store_true")
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument("--no-export", action="store_true", default=False)
    parser.add_argument("--out-root", type=Path, default=HANDOFF_ROOT)
    args = parser.parse_args()

    slug = slug_for(args.model, args.phase, args.folds, args.kernel_slug)
    build_chunk(args, slug)

    out_dir = args.out_root / slug
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    package_out = out_dir / "kaggle_gpu_chunk"
    shutil.copytree(KAGGLE_PACKAGE_DIR, package_out)
    write_readme(
        out_dir,
        model=args.model,
        phase=args.phase,
        folds=args.folds,
        kernel_owner=args.kernel_owner,
        slug=slug,
        accelerator=args.accelerator,
    )

    archive_base = args.out_root / slug
    archive_path = shutil.make_archive(str(archive_base), "zip", out_dir)

    print(f"Wrote handoff package: {out_dir}")
    print(f"Wrote archive: {archive_path}")
    print(f"Kernel ref: {args.kernel_owner}/{slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
