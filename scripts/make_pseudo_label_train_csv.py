import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_probs(path: Path, expected: int) -> np.ndarray:
    probs = np.load(path).astype(np.float64).reshape(-1)
    if len(probs) != expected:
        raise ValueError(f"{path} has {len(probs)} rows, expected {expected}.")
    return probs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a cautious pseudo-label training CSV from high-confidence test predictions.")
    parser.add_argument("--train-csv", type=Path, default=ROOT / "outputs/final_effnetv2_s_lb/teacher_soft_train.csv")
    parser.add_argument("--final-test-csv", type=Path, default=ROOT / "outputs/preprocessing/final_test.csv")
    parser.add_argument("--sample-csv", type=Path, default=ROOT / "data/sample_submission.csv")
    parser.add_argument("--student-probs", type=Path, default=ROOT / "outputs/final_effnetv2_s_lb_mps/threshold_sweep/test_probs.npy")
    parser.add_argument("--teacher-probs", type=Path, default=ROOT / "outputs/leaderboard_ensemble/ensemble_probs.npy")
    parser.add_argument("--student-weight", type=float, default=0.70)
    parser.add_argument("--pos-threshold", type=float, default=0.985)
    parser.add_argument("--neg-threshold", type=float, default=0.015)
    parser.add_argument("--max-disagreement", type=float, default=0.12)
    parser.add_argument("--pseudo-sample-weight", type=float, default=0.30)
    parser.add_argument("--pseudo-distill-weight", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/final_effnetv2_s_lb_pseudo")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv).reset_index(drop=True)
    sample = pd.read_csv(args.sample_csv).reset_index(drop=True)
    final_test = pd.read_csv(args.final_test_csv).reset_index(drop=True)
    if not sample["image_id"].equals(final_test["image_id"]):
        final_test = sample[["image_id"]].merge(final_test, on="image_id", how="left")

    student = load_probs(args.student_probs, len(sample))
    teacher = load_probs(args.teacher_probs, len(sample))
    blend = args.student_weight * student + (1.0 - args.student_weight) * teacher
    disagreement = np.abs(student - teacher)

    pos_mask = (blend >= args.pos_threshold) & (disagreement <= args.max_disagreement)
    neg_mask = (blend <= args.neg_threshold) & (disagreement <= args.max_disagreement)
    selected = pos_mask | neg_mask

    pseudo = final_test.loc[selected].copy()
    pseudo["target"] = pos_mask[selected].astype(int)
    pseudo["target_coco"] = pseudo["target"]
    pseudo["target_corrected"] = pseudo["target"]
    pseudo["target_original"] = pseudo["target"]
    pseudo["target_source"] = "pseudo_label"
    pseudo["target_note"] = "High-confidence pseudo label from LB student + teacher ensemble agreement."
    pseudo["label_mismatch"] = False
    pseudo["teacher_prob"] = blend[selected]
    pseudo["pseudo_student_prob"] = student[selected]
    pseudo["pseudo_teacher_prob"] = teacher[selected]
    pseudo["pseudo_disagreement"] = disagreement[selected]
    pseudo["teacher_available"] = True
    pseudo["noise_bucket"] = "pseudo_high_confidence"
    pseudo["sample_weight"] = float(args.pseudo_sample_weight)
    pseudo["distill_weight"] = float(args.pseudo_distill_weight)
    pseudo["target_soft"] = np.clip(blend[selected], 0.02, 0.98)
    pseudo["is_pseudo"] = True
    pseudo["image_source"] = "test"

    train_out = train.copy()
    train_out["is_pseudo"] = False
    train_out["image_source"] = "train"

    for col in train_out.columns:
        if col not in pseudo.columns:
            pseudo[col] = np.nan
    for col in pseudo.columns:
        if col not in train_out.columns:
            train_out[col] = np.nan
    pseudo = pseudo[train_out.columns]
    combined = pd.concat([train_out, pseudo], ignore_index=True)

    out_path = args.out_dir / "teacher_soft_train_pseudo.csv"
    pseudo_path = args.out_dir / "pseudo_rows.csv"
    combined.to_csv(out_path, index=False)
    pseudo.to_csv(pseudo_path, index=False)

    summary = {
        "train_csv": str(args.train_csv),
        "out_path": str(out_path),
        "pseudo_path": str(pseudo_path),
        "train_rows": int(len(train_out)),
        "pseudo_rows": int(len(pseudo)),
        "combined_rows": int(len(combined)),
        "pseudo_positive": int(pseudo["target"].sum()) if len(pseudo) else 0,
        "pseudo_negative": int(len(pseudo) - pseudo["target"].sum()) if len(pseudo) else 0,
        "student_probs": str(args.student_probs),
        "teacher_probs": str(args.teacher_probs),
        "student_weight": args.student_weight,
        "pos_threshold": args.pos_threshold,
        "neg_threshold": args.neg_threshold,
        "max_disagreement": args.max_disagreement,
        "pseudo_sample_weight": args.pseudo_sample_weight,
        "pseudo_distill_weight": args.pseudo_distill_weight,
    }
    (args.out_dir / "pseudo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
