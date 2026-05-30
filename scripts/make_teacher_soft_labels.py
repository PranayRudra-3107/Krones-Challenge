import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_PREPROCESS = ROOT / "outputs" / "preprocessing"
DEFAULT_OUT = ROOT / "outputs" / "final_effnetv2_s"
DEFAULT_CONFIG = ROOT / "configs" / "final_strategy.json"
DEFAULT_TEACHERS = ["v26_convnext_small", "v26_maxvit"]


def load_teacher_tags(config_path: Path) -> list[str]:
    if not config_path.exists():
        return DEFAULT_TEACHERS
    with config_path.open() as handle:
        payload = json.load(handle)
    return [
        item["tag"]
        for item in payload.get("teachers", [])
        if item.get("tag")
    ] or DEFAULT_TEACHERS


def ensure_final_train(preprocess_dir: Path) -> Path:
    final_train_path = preprocess_dir / "final_train.csv"
    if final_train_path.exists():
        return final_train_path

    from preprocess_coco import preprocess

    preprocess(DATA, preprocess_dir)
    if not final_train_path.exists():
        raise FileNotFoundError(f"Expected preprocessing output: {final_train_path}")
    return final_train_path


def load_oof_prediction(tag: str, phases: list[str]) -> tuple[pd.DataFrame | None, np.ndarray | None, str | None]:
    out_dir = ROOT / "outputs" / tag
    active_path = out_dir / f"labels_train_active_{tag}.csv"
    if not active_path.exists():
        return None, None, None

    active = pd.read_csv(active_path).reset_index(drop=True)
    for phase in phases:
        oof = np.full(len(active), np.nan, dtype=np.float64)
        found = False
        for fold in range(5):
            prob_path = out_dir / f"oof_{tag}_{phase}_fold{fold}.npy"
            idx_path = out_dir / f"oof_idx_{tag}_{phase}_fold{fold}.npy"
            if not (prob_path.exists() and idx_path.exists()):
                continue
            idx = np.load(idx_path).astype(int)
            probs = np.load(prob_path).astype(np.float64).reshape(-1)
            if len(idx) != len(probs):
                print(f"{tag} {phase} fold {fold}: length mismatch; skipped.")
                continue
            valid_idx = (idx >= 0) & (idx < len(oof))
            oof[idx[valid_idx]] = probs[valid_idx]
            found = True
        if found and not np.isnan(oof).all():
            return active, oof, phase
    return None, None, None


def load_test_prediction(tag: str, phases: list[str]) -> tuple[np.ndarray | None, str | None]:
    out_dir = ROOT / "outputs" / tag
    candidates = []
    for phase in phases:
        candidates.append((out_dir / f"test_{tag}_{phase}_mean.npy", phase))
        candidates.append((out_dir / f"test_{tag}_{phase}_ensemble.npy", phase))
    candidates.append((out_dir / f"test_{tag}_p1_ensemble.npy", "p1"))

    for path, phase in candidates:
        if path.exists():
            return np.load(path).astype(np.float64).reshape(-1), phase
    return None, None


def blend_columns(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = df[columns].to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    sums = np.where(valid, values, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    out = np.full(len(df), np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def add_noise_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    target = out["target"].astype(int).to_numpy()
    teacher = out["teacher_prob"].to_numpy(dtype=np.float64)
    available = np.isfinite(teacher)

    bucket = np.full(len(out), "teacher_missing", dtype=object)
    sample_weight = np.ones(len(out), dtype=np.float32)
    distill_weight = np.zeros(len(out), dtype=np.float32)

    clean = available & (
        ((target == 1) & (teacher >= 0.50))
        | ((target == 0) & (teacher < 0.50))
    )
    weak = available & ~clean & (
        ((target == 1) & (teacher > 0.30))
        | ((target == 0) & (teacher < 0.70))
    )
    moderate = available & ~clean & (
        ((target == 1) & (teacher > 0.10) & (teacher <= 0.30))
        | ((target == 0) & (teacher >= 0.70) & (teacher < 0.90))
    )
    hard = available & ~clean & (
        ((target == 1) & (teacher <= 0.10))
        | ((target == 0) & (teacher >= 0.90))
    )

    bucket[clean] = "teacher_agrees"
    bucket[weak] = "weak_disagree"
    bucket[moderate] = "moderate_disagree"
    bucket[hard] = "hard_disagree"

    sample_weight[weak] = 0.80
    sample_weight[moderate] = 0.50
    sample_weight[hard] = 0.25

    confidence = np.clip(np.abs(teacher - 0.5) * 2.0, 0.0, 1.0)
    distill_weight[available] = (0.25 + 0.35 * confidence[available]).astype(np.float32)
    distill_weight[hard] = 0.70

    target_soft = target.astype(np.float64)
    target_soft[available] = 0.65 * target[available] + 0.35 * teacher[available]

    out["teacher_available"] = available
    out["noise_bucket"] = bucket
    out["sample_weight"] = sample_weight
    out["distill_weight"] = distill_weight
    out["target_soft"] = np.clip(target_soft, 0.02, 0.98)
    return out


def build_soft_labels(
    teacher_tags: list[str],
    preprocess_dir: Path,
    out_dir: Path,
    phases: list[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_train = pd.read_csv(ensure_final_train(preprocess_dir)).reset_index(drop=True)
    if "target" not in final_train.columns:
        raise ValueError("final_train.csv must contain a COCO-derived target column.")

    train_soft = final_train.copy()
    train_soft["stem"] = train_soft["stem"].astype(str)
    teacher_summaries = []
    teacher_cols = []

    for tag in teacher_tags:
        active, oof, phase = load_oof_prediction(tag, phases)
        col = f"teacher_{tag}_prob"
        train_soft[col] = np.nan
        if active is None or oof is None or phase is None:
            teacher_summaries.append({"tag": tag, "oof_status": "missing"})
            continue

        active = active.copy()
        active["stem"] = active["image_id"].map(lambda value: Path(str(value)).stem)
        prob_by_stem = dict(zip(active["stem"].astype(str), oof))
        train_soft[col] = train_soft["stem"].map(prob_by_stem).astype(float)
        coverage = int(train_soft[col].notna().sum())
        teacher_cols.append(col)
        teacher_summaries.append({
            "tag": tag,
            "oof_status": "loaded",
            "phase": phase,
            "coverage": coverage,
            "rows": int(len(train_soft)),
        })

    if teacher_cols:
        train_soft["teacher_prob"] = blend_columns(train_soft, teacher_cols)
    else:
        train_soft["teacher_prob"] = np.nan

    train_soft = add_noise_weights(train_soft)
    train_path = out_dir / "teacher_soft_train.csv"
    train_soft.to_csv(train_path, index=False)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    test_soft = sample[["image_id"]].copy()
    test_teacher_cols = []
    for tag in teacher_tags:
        probs, phase = load_test_prediction(tag, phases)
        col = f"teacher_{tag}_prob"
        test_soft[col] = np.nan
        if probs is None:
            continue
        if len(probs) != len(test_soft):
            print(f"{tag}: test prediction length {len(probs)} != {len(test_soft)}; skipped.")
            continue
        test_soft[col] = probs
        test_teacher_cols.append(col)
        for item in teacher_summaries:
            if item["tag"] == tag:
                item["test_status"] = "loaded"
                item["test_phase"] = phase
                break

    if test_teacher_cols:
        test_soft["teacher_prob"] = blend_columns(test_soft, test_teacher_cols)
    else:
        test_soft["teacher_prob"] = np.nan
    test_path = out_dir / "teacher_soft_test.csv"
    test_soft.to_csv(test_path, index=False)

    bucket_counts = {
        str(key): int(value)
        for key, value in train_soft["noise_bucket"].value_counts().items()
    }
    summary = {
        "teacher_tags": teacher_tags,
        "phases": phases,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "teacher_columns": teacher_cols,
        "teacher_summaries": teacher_summaries,
        "teacher_coverage": int(train_soft["teacher_available"].sum()),
        "train_rows": int(len(train_soft)),
        "noise_bucket_counts": bucket_counts,
        "mean_sample_weight": float(train_soft["sample_weight"].mean()),
        "label_source": "COCO-derived target from outputs/preprocessing/final_train.csv",
    }
    (out_dir / "teacher_soft_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge ConvNeXt/MaxViT teacher OOF predictions into EfficientNetV2-S soft-label tables."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preprocess-dir", type=Path, default=DEFAULT_PREPROCESS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teachers", nargs="*", default=None)
    parser.add_argument("--phases", nargs="+", default=["p2", "p1"])
    args = parser.parse_args()

    teachers = args.teachers or load_teacher_tags(args.config)
    summary = build_soft_labels(teachers, args.preprocess_dir, args.out_dir, args.phases)
    print(f"Teachers: {', '.join(summary['teacher_tags'])}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Teacher coverage: {summary['teacher_coverage']}/{summary['train_rows']}")
    print(f"Noise buckets: {summary['noise_bucket_counts']}")
    print(f"Wrote: {summary['train_path']}")
    print(f"Wrote: {summary['test_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
