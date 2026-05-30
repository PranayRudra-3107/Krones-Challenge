import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "outputs" / "ensemble"

DEFAULT_MODELS = [
    {"tag": "v26_convnext_small", "weight": 1.0},
    {"tag": "v26_maxvit", "weight": 1.0},
]


def best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.20, 0.81, 0.005):
        f1 = f1_score(y_true, (probs >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(threshold), float(f1)
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


def load_model_config(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_MODELS
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "models" in payload:
        return [
            {"tag": item["tag"], "weight": float(item.get("weight", 1.0))}
            for item in payload["models"]
        ]
    return payload


def load_test_prediction(tag: str) -> np.ndarray:
    out_dir = ROOT / "outputs" / tag
    candidates = [
        out_dir / f"test_{tag}_p2_mean.npy",
        out_dir / f"test_{tag}_p1_ensemble.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float64)
    raise FileNotFoundError(
        f"No prediction file found for {tag}. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_oof_prediction(tag: str) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    out_dir = ROOT / "outputs" / tag
    active_path = out_dir / f"labels_train_active_{tag}.csv"
    if not active_path.exists():
        return None, None

    active = pd.read_csv(active_path)
    oof = np.full(len(active), np.nan, dtype=np.float64)
    for fold in range(5):
        prob_path = out_dir / f"oof_{tag}_p2_fold{fold}.npy"
        idx_path = out_dir / f"oof_idx_{tag}_p2_fold{fold}.npy"
        if prob_path.exists() and idx_path.exists():
            idx = np.load(idx_path).astype(int)
            oof[idx] = np.load(prob_path).astype(np.float64)

    if np.isnan(oof).all():
        return None, None
    return active, oof


def load_oof_members(models: list[dict]) -> tuple[pd.DataFrame | None, list[dict]]:
    active_ref = None
    members = []

    for model in models:
        tag = model["tag"]
        weight = float(model.get("weight", 1.0))
        active, oof = load_oof_prediction(tag)
        if active is None or oof is None:
            continue
        if active_ref is None:
            active_ref = active
        elif not active_ref["image_id"].equals(active["image_id"]):
            print(f"OOF alignment differs for {tag}; skipping its OOF for thresholding.")
            continue
        members.append({"tag": tag, "weight": weight, "oof": oof})

    return active_ref, members


def blend_arrays(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive")
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for array, weight in zip(arrays, weights):
        out += array.astype(np.float64) * float(weight)
    return out / total_weight


def optimize_weights(active: pd.DataFrame, members: list[dict]) -> tuple[dict[str, float], str]:
    if len(members) < 2:
        return {member["tag"]: member["weight"] for member in members}, "not enough OOF members"

    y_true = active["target"].to_numpy().astype(int)
    oofs = [member["oof"] for member in members]
    valid = np.ones(len(active), dtype=bool)
    for oof in oofs:
        valid &= ~np.isnan(oof)

    if valid.sum() <= 100:
        return {member["tag"]: member["weight"] for member in members}, "insufficient aligned OOF"

    grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best = None
    for weights in itertools.product(grid, repeat=len(members)):
        blended = blend_arrays([oof[valid] for oof in oofs], list(weights))
        threshold, f1 = best_threshold(y_true[valid], blended)
        if best is None or f1 > best["f1"]:
            best = {"weights": weights, "threshold": threshold, "f1": f1}

    weight_map = {
        member["tag"]: float(weight)
        for member, weight in zip(members, best["weights"])
    }
    source = (
        f"OOF weight search F1={best['f1']:.5f} "
        f"threshold={best['threshold']:.4f} samples={int(valid.sum())}"
    )
    return weight_map, source


def derive_global_threshold(active: pd.DataFrame | None, members: list[dict], blended_test: np.ndarray) -> tuple[float, str]:
    if active is not None and members:
        arrays = [member["oof"] for member in members]
        weights = [float(member["weight"]) for member in members]
        blended_oof = blend_arrays(arrays, weights)
        valid = ~np.isnan(blended_oof)
        if valid.sum() > 100:
            threshold, f1 = best_threshold(
                active.loc[valid, "target"].to_numpy().astype(int),
                blended_oof[valid],
            )
            return threshold, f"OOF ensemble F1={f1:.5f} on {int(valid.sum())} samples"

    final_train = ROOT / "outputs" / "preprocessing" / "final_train.csv"
    train_path = final_train if final_train.exists() else DATA / "train.csv"
    train = pd.read_csv(train_path)
    positive_rate = float(train["target"].mean())
    source = "COCO final_train" if train_path == final_train else "train.csv"
    threshold = float(np.quantile(blended_test, 1 - positive_rate))
    return threshold, f"{source} positive-rate quantile fallback ({positive_rate:.4f})"


def derive_btype_thresholds(active: pd.DataFrame, members: list[dict]) -> tuple[dict[int, float], str]:
    arrays = [member["oof"] for member in members]
    weights = [float(member["weight"]) for member in members]
    blended_oof = blend_arrays(arrays, weights)
    valid = ~np.isnan(blended_oof)
    global_threshold, global_f1 = best_threshold(
        active.loc[valid, "target"].to_numpy().astype(int),
        blended_oof[valid],
    )

    thresholds = {}
    parts = [f"global={global_threshold:.4f}/F1={global_f1:.5f}"]
    for btype in sorted(active["btype_id"].dropna().astype(int).unique()):
        mask = valid & active["btype_id"].astype(int).eq(btype).to_numpy()
        if mask.sum() < 100:
            thresholds[int(btype)] = global_threshold
            parts.append(f"b{btype}=global(n={int(mask.sum())})")
            continue
        threshold, f1 = best_threshold(
            active.loc[mask, "target"].to_numpy().astype(int),
            blended_oof[mask],
        )
        thresholds[int(btype)] = threshold
        parts.append(f"b{btype}={threshold:.4f}/F1={f1:.5f}/n={int(mask.sum())}")
    return thresholds, "; ".join(parts)


def test_btypes(sample: pd.DataFrame) -> np.ndarray:
    bottletypes = pd.read_csv(DATA / "bottletypes.csv")
    bottletypes = bottletypes[bottletypes["split"].eq("test")].copy()
    bottletypes["btype_id"] = bottletypes["bottle_type"].map(canonical_bottle_type)
    bmap = dict(zip(bottletypes["image_id"], bottletypes["btype_id"]))
    return sample["image_id"].map(bmap).fillna(-1).astype(int).to_numpy()


def apply_thresholds(probs: np.ndarray, sample: pd.DataFrame, global_threshold: float, btype_thresholds: dict[int, float] | None) -> np.ndarray:
    if not btype_thresholds:
        return (probs >= global_threshold).astype(int)

    btypes = test_btypes(sample)
    pred = np.zeros(len(sample), dtype=int)
    for btype in np.unique(btypes):
        threshold = float(btype_thresholds.get(int(btype), global_threshold))
        mask = btypes == btype
        pred[mask] = (probs[mask] >= threshold).astype(int)
    return pred


def main() -> int:
    parser = argparse.ArgumentParser(description="Blend ensemble model predictions into a Kaggle submission.")
    parser.add_argument("--config", type=Path, default=ROOT / "outputs" / "ensemble_models.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--optimize-weights", action="store_true")
    parser.add_argument("--per-bottle-thresholds", action="store_true")
    args = parser.parse_args()

    models = load_model_config(args.config if args.config.exists() else None)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    active, oof_members = load_oof_members(models)

    weight_source = "config/default weights"
    if args.optimize_weights and active is not None and oof_members:
        weight_map, weight_source = optimize_weights(active, oof_members)
        for model in models:
            if model["tag"] in weight_map:
                model["weight"] = weight_map[model["tag"]]
        for member in oof_members:
            if member["tag"] in weight_map:
                member["weight"] = weight_map[member["tag"]]

    blended = np.zeros(len(sample), dtype=np.float64)
    total_weight = 0.0
    used = []
    for model in models:
        tag = model["tag"]
        weight = float(model.get("weight", 1.0))
        probs = load_test_prediction(tag)
        if len(probs) != len(sample):
            raise ValueError(f"{tag} prediction length {len(probs)} != sample rows {len(sample)}")
        blended += probs * weight
        total_weight += weight
        used.append(tag)

    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive")
    blended /= total_weight

    threshold, threshold_source = (
        (args.threshold, "manual") if args.threshold is not None else derive_global_threshold(active, oof_members, blended)
    )

    btype_thresholds = None
    btype_threshold_source = None
    if args.per_bottle_thresholds and active is not None and oof_members:
        btype_thresholds, btype_threshold_source = derive_btype_thresholds(active, oof_members)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "ensemble_probs.npy", blended)
    submission = sample[["image_id"]].copy()
    submission["target"] = apply_thresholds(blended, sample, threshold, btype_thresholds)
    submission_path = args.out_dir / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    summary = {
        "models": used,
        "weights": {model["tag"]: float(model.get("weight", 1.0)) for model in models},
        "weight_source": weight_source,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "btype_thresholds": {str(k): float(v) for k, v in (btype_thresholds or {}).items()},
        "btype_threshold_source": btype_threshold_source,
        "mean_probability": float(blended.mean()),
        "positive_predictions": int(submission["target"].sum()),
        "rows": int(len(submission)),
        "submission_path": str(submission_path),
    }
    (args.out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Models: {', '.join(used)}")
    print(f"Weights: {summary['weights']} ({weight_source})")
    print(f"Threshold: {threshold:.4f} ({threshold_source})")
    if btype_thresholds:
        print(f"Per-bottle thresholds: {summary['btype_thresholds']}")
    print(f"Positive predictions: {summary['positive_predictions']}/{summary['rows']}")
    print(f"Wrote: {submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
