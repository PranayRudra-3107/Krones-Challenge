import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]


def best_threshold(y_true: np.ndarray, probs: np.ndarray, lo: float, hi: float, step: float) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    thresholds = np.arange(lo, hi + step / 2, step)
    for threshold in thresholds:
        score = f1_score(y_true, (probs >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t, best_f1


def canonical_bottle_type(value: str) -> int:
    text = str(value).lower()
    if "vichy" in text:
        return 0
    if "euro" in text:
        return 1
    if "nrw" in text:
        return 2
    return -1


def load_test_btypes(data_root: Path, sample: pd.DataFrame) -> np.ndarray:
    path = data_root / "bottletypes.csv"
    if not path.exists():
        return np.full(len(sample), -1, dtype=np.int64)
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"].eq("test")].copy()
    df["btype_id"] = df["bottle_type"].map(canonical_bottle_type).astype(int)
    mapping = {}
    for row in df.itertuples(index=False):
        image_id = str(row.image_id)
        mapping[image_id] = int(row.btype_id)
        mapping[Path(image_id).stem] = int(row.btype_id)
    return np.array(
        [mapping.get(str(image_id), mapping.get(Path(str(image_id)).stem, -1)) for image_id in sample["image_id"]],
        dtype=np.int64,
    )


def load_oof(out_root: Path, train_csv: Path, n_folds: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(train_csv).reset_index(drop=True)
    oof = np.full(len(train_df), np.nan, dtype=np.float32)
    for fold in range(n_folds):
        fold_dir = out_root / f"fold{fold}"
        idx_path = fold_dir / "valid_row_idx.npy"
        prob_path = fold_dir / "valid_probs_final_effnetv2_s.npy"
        if not idx_path.exists() or not prob_path.exists():
            raise FileNotFoundError(f"Missing fold validation files under {fold_dir}")
        idx = np.load(idx_path).astype(np.int64)
        probs = np.load(prob_path).astype(np.float32).reshape(-1)
        if len(idx) != len(probs):
            raise ValueError(f"fold{fold}: {len(idx)} valid indices but {len(probs)} probs")
        oof[idx] = probs
    valid = np.isfinite(oof)
    if not valid.all():
        raise RuntimeError(f"OOF incomplete: {int((~valid).sum())} missing rows")
    targets = train_df["target"].astype(int).to_numpy()
    return train_df, targets, oof


def apply_thresholds(
    sample: pd.DataFrame,
    probs: np.ndarray,
    test_btypes: np.ndarray,
    global_threshold: float,
    btype_thresholds: dict[int, float] | None,
) -> np.ndarray:
    preds = np.zeros(len(sample), dtype=np.int64)
    if not btype_thresholds:
        return (probs >= global_threshold).astype(np.int64)
    for btype in np.unique(test_btypes):
        mask = test_btypes == btype
        threshold = float(btype_thresholds.get(int(btype), global_threshold))
        preds[mask] = (probs[mask] >= threshold).astype(np.int64)
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description="Use student fold OOF predictions to threshold ensemble test probabilities.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ensemble-probs", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "outputs/final_effnetv2_s_lb/teacher_soft_train.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--lo", type=float, default=0.20)
    parser.add_argument("--hi", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--per-bottle", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df, targets, oof = load_oof(args.out_root, args.train_csv, args.n_folds)
    sample = pd.read_csv(args.data_root / "sample_submission.csv")
    test_probs = np.load(args.ensemble_probs).astype(np.float32).reshape(-1)
    if len(test_probs) != len(sample):
        raise ValueError(f"test prob length {len(test_probs)} != sample rows {len(sample)}")

    global_t, global_f1 = best_threshold(targets, oof, args.lo, args.hi, args.step)
    btype_thresholds = None
    btype_info = {}
    if args.per_bottle:
        btype_thresholds = {}
        btypes = train_df["btype_id"].fillna(-1).astype(int).to_numpy()
        for btype in sorted(set(btypes.tolist())):
            mask = btypes == btype
            if mask.sum() < 500:
                btype_thresholds[int(btype)] = global_t
                btype_info[str(int(btype))] = {
                    "threshold": global_t,
                    "f1": None,
                    "n": int(mask.sum()),
                    "source": "global_fallback",
                }
                continue
            threshold, score = best_threshold(targets[mask], oof[mask], args.lo, args.hi, args.step)
            btype_thresholds[int(btype)] = threshold
            btype_info[str(int(btype))] = {
                "threshold": threshold,
                "f1": score,
                "n": int(mask.sum()),
                "source": "per_bottle_oof",
            }

    test_btypes = load_test_btypes(args.data_root, sample)
    preds_global = apply_thresholds(sample, test_probs, test_btypes, global_t, None)
    sub_global = sample[["image_id"]].copy()
    sub_global["target"] = preds_global
    global_path = args.out_dir / f"submission_oof_global_t{global_t:.3f}.csv"
    sub_global.to_csv(global_path, index=False)

    per_bottle_path = None
    if btype_thresholds:
        preds_btype = apply_thresholds(sample, test_probs, test_btypes, global_t, btype_thresholds)
        sub_btype = sample[["image_id"]].copy()
        sub_btype["target"] = preds_btype
        per_bottle_path = args.out_dir / "submission_oof_per_bottle.csv"
        sub_btype.to_csv(per_bottle_path, index=False)

    summary = {
        "out_root": str(args.out_root),
        "ensemble_probs": str(args.ensemble_probs),
        "train_csv": str(args.train_csv),
        "oof_rows": int(len(oof)),
        "global_threshold": global_t,
        "global_oof_f1": global_f1,
        "global_positive_predictions": int(preds_global.sum()),
        "global_submission": str(global_path),
        "per_bottle": bool(args.per_bottle),
        "btype_thresholds": btype_info,
        "per_bottle_submission": str(per_bottle_path) if per_bottle_path else None,
    }
    (args.out_dir / "oof_threshold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
