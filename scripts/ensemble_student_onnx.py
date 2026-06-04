import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from generate_onnx_threshold_submissions import run_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Average multiple EfficientNetV2-S ONNX student models.")
    parser.add_argument("--model-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    prob_stack = []
    model_summaries = []

    for i, model_path in enumerate(args.model_paths):
        infer_args = SimpleNamespace(
            model_path=Path(model_path),
            data_root=args.data_root,
            out_dir=args.out_dir,
            thresholds=[],
            batch_size=args.batch_size,
            img_size=args.img_size,
        )
        print(f"\n=== model {i + 1}/{len(args.model_paths)}: {model_path} ===")
        sample, probs, metadata = run_inference(infer_args)
        samples.append(sample)
        prob_stack.append(probs.astype(np.float32))
        np.save(args.out_dir / f"test_probs_model{i}.npy", probs.astype(np.float32))
        model_summaries.append({
            "model_path": str(model_path),
            "metadata_threshold": metadata.get("threshold"),
            "validation_f1": metadata.get("validation_f1"),
            "validation_epoch": metadata.get("validation_epoch"),
            "mean_probability": float(probs.mean()),
        })

    base_sample = samples[0]
    for sample in samples[1:]:
        if not base_sample["image_id"].equals(sample["image_id"]):
            raise RuntimeError("Model inference sample order mismatch.")

    stacked = np.stack(prob_stack, axis=0)
    probs_mean = stacked.mean(axis=0).astype(np.float32)
    np.save(args.out_dir / "ensemble_probs.npy", probs_mean)
    prob_csv = base_sample[["image_id"]].copy()
    prob_csv["probability"] = probs_mean
    prob_csv.to_csv(args.out_dir / "ensemble_probs.csv", index=False)

    summary = {
        "model_paths": [str(path) for path in args.model_paths],
        "models": model_summaries,
        "mean_probability": float(probs_mean.mean()),
        "thresholds": {},
    }
    for threshold in args.thresholds:
        preds = (probs_mean >= threshold).astype(int)
        sub = base_sample[["image_id"]].copy()
        sub["target"] = preds
        out_path = args.out_dir / f"submission_t{threshold:.3f}.csv"
        sub.to_csv(out_path, index=False)
        summary["thresholds"][f"{threshold:.3f}"] = {
            "positive_predictions": int(preds.sum()),
            "path": str(out_path),
        }
        print(f"threshold={threshold:.3f} positives={int(preds.sum())} wrote={out_path}")

    (args.out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote: {args.out_dir / 'ensemble_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
