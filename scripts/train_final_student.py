import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_TRAIN_CSV = ROOT / "outputs" / "final_effnetv2_s" / "teacher_soft_train.csv"
DEFAULT_OUT = ROOT / "outputs" / "final_effnetv2_s"
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
N_BOTTLE_TYPES = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_dir() -> Path:
    candidates = [
        DATA,
        DATA / "1st-krones-vision-ai-challenge",
        ROOT,
        Path("/kaggle/input/1st-krones-vision-ai-challenge"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "train_images").exists():
            return candidate
    return DATA


def ensure_train_csv(path: Path) -> Path:
    if path.exists():
        return path
    from make_teacher_soft_labels import build_soft_labels, load_teacher_tags, DEFAULT_CONFIG, DEFAULT_PREPROCESS

    teachers = load_teacher_tags(DEFAULT_CONFIG)
    build_soft_labels(teachers, DEFAULT_PREPROCESS, path.parent, ["p2", "p1"])
    if not path.exists():
        raise FileNotFoundError(f"Could not create {path}")
    return path


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def crop_roi(image: np.ndarray, row: pd.Series, padding: int = 8) -> np.ndarray:
    values = [row.get("roi_x"), row.get("roi_y"), row.get("roi_w"), row.get("roi_h")]
    if any(pd.isna(value) for value in values):
        return image
    x, y, w, h = [float(value) for value in values]
    ih, iw = image.shape[:2]
    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(iw, int(x + w) + padding)
    y2 = min(ih, int(y + h) + padding)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def read_image(image_dir: Path, image_id: str, img_size: int) -> np.ndarray:
    stem = Path(str(image_id)).stem
    candidates = [image_dir / str(image_id)]
    candidates += [image_dir / f"{stem}{ext}" for ext in [".png", ".jpg", ".jpeg"]]
    for path in candidates:
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.zeros((img_size, img_size, 3), dtype=np.uint8)


def make_transforms(img_size: int) -> tuple[A.Compose, A.Compose]:
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.08,
            rotate_limit=6,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.65,
        ),
        A.RandomBrightnessContrast(0.08, 0.08, p=0.35),
        A.HueSaturationValue(4, 6, 4, p=0.20),
        A.CoarseDropout(max_holes=4, max_height=18, max_width=18, p=0.20),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    valid_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return train_tf, valid_tf


class BottleStudentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform: A.Compose, img_size: int):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = read_image(self.image_dir, str(row["image_id"]), self.img_size)
        image = crop_roi(image, row)
        image = self.transform(image=image)["image"]

        target = float(row.get("target", 0.0))
        soft = float(row.get("target_soft", target))
        btype = int(row.get("btype_id", -1))
        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.float32),
            "target_soft": torch.tensor(soft, dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
            "distill_weight": torch.tensor(float(row.get("distill_weight", 0.0)), dtype=torch.float32),
            "btype": torch.tensor(btype, dtype=torch.long),
        }


class EffNetV2Student(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
        dropout: float = 0.25,
        btype_embed_dim: int = 8,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        features = int(self.backbone.num_features)
        self.btype_embedding = nn.Embedding(N_BOTTLE_TYPES + 1, btype_embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features + btype_embed_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.btype_head = nn.Linear(features, N_BOTTLE_TYPES)

    def forward(self, image: torch.Tensor, btype: torch.Tensor, return_btype: bool = False):
        features = self.backbone(image)
        btype_index = torch.clamp(btype.long(), min=-1, max=N_BOTTLE_TYPES - 1) + 1
        btype_emb = self.btype_embedding(btype_index)
        logits = self.classifier(torch.cat([features, btype_emb], dim=1)).squeeze(1)
        if return_btype:
            return logits, self.btype_head(features)
        return logits


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.sum().clamp_min(1e-6)
    return (loss * weight).sum() / denom


def compute_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    distill_alpha: float,
    btype_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    image = batch["image"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    target_soft = batch["target_soft"].to(device, non_blocking=True)
    sample_weight = batch["sample_weight"].to(device, non_blocking=True)
    distill_weight = batch["distill_weight"].to(device, non_blocking=True)
    btype = batch["btype"].to(device, non_blocking=True)

    logits, btype_logits = model(image, btype, return_btype=True)
    hard = weighted_mean(
        F.binary_cross_entropy_with_logits(logits, target, reduction="none"),
        sample_weight,
    )
    if distill_weight.sum() > 0:
        soft = weighted_mean(
            F.binary_cross_entropy_with_logits(logits, target_soft, reduction="none"),
            distill_weight,
        )
    else:
        soft = logits.new_tensor(0.0)

    valid_btype = (btype >= 0) & (btype < N_BOTTLE_TYPES)
    if valid_btype.any():
        aux = F.cross_entropy(btype_logits[valid_btype], btype[valid_btype])
    else:
        aux = logits.new_tensor(0.0)

    loss = hard + distill_alpha * soft + btype_weight * aux
    metrics = {
        "hard": float(hard.detach().cpu()),
        "soft": float(soft.detach().cpu()),
        "btype": float(aux.detach().cpu()),
    }
    return loss, metrics


def best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.20, 0.805, 0.005):
        score = f1_score(y_true, (probs >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return best_t, best_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    probs, targets, losses = [], [], []
    for batch in tqdm(loader, desc="valid", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        btype = batch["btype"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(image, btype)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        losses.append(float(loss.detach().cpu()))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    probs_np = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    targets_np = np.concatenate(targets).astype(int) if targets else np.array([], dtype=np.int64)
    threshold, score = best_threshold(targets_np, probs_np) if len(targets_np) else (0.5, 0.0)
    return float(np.mean(losses) if losses else math.nan), threshold, score, targets_np, probs_np


def split_train_valid(df: pd.DataFrame, valid_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = (df["btype_id"].fillna(-1).clip(lower=-1).astype(int) + 1) * 2 + df["target"].astype(int)
    train_idx, valid_idx = train_test_split(
        np.arange(len(df)),
        test_size=valid_fraction,
        random_state=seed,
        stratify=strat,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    data_dir = Path(args.data_dir) if args.data_dir else find_data_dir()
    train_csv = ensure_train_csv(Path(args.train_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    if args.max_train_rows and args.max_train_rows < len(df):
        df = df.sample(args.max_train_rows, random_state=args.seed).reset_index(drop=True)
    if "target_soft" not in df.columns:
        df["target_soft"] = df["target"].astype(float)
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    if "distill_weight" not in df.columns:
        df["distill_weight"] = 0.0
    df["btype_id"] = df["btype_id"].fillna(-1).astype(int)

    train_df, valid_df = split_train_valid(df, args.valid_fraction, args.seed)
    train_tf, valid_tf = make_transforms(args.img_size)
    train_ds = BottleStudentDataset(train_df, data_dir / "train_images", train_tf, args.img_size)
    valid_ds = BottleStudentDataset(valid_df, data_dir / "train_images", valid_tf, args.img_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = choose_device(args.device)
    model = EffNetV2Student(pretrained=not args.no_pretrained, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.disable_amp))

    best = {"f1": -1.0, "threshold": 0.5, "epoch": -1}
    best_path = out_dir / "final_effnetv2_s_best.pth"
    started = time.time()
    epoch_times = []
    print(f"data_dir: {data_dir}")
    print(f"train_csv: {train_csv}")
    print(f"device: {device}")
    print(f"train/valid: {len(train_df)}/{len(valid_df)}")

    for epoch in range(args.epochs):
        epoch_started = time.time()
        model.train()
        running = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.disable_amp)):
                loss, parts = compute_loss(
                    model,
                    batch,
                    device,
                    distill_alpha=args.distill_alpha,
                    btype_weight=args.btype_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running.append(float(loss.detach().cpu()))
            pbar.set_postfix(
                loss=f"{np.mean(running[-50:]):.4f}",
                hard=f"{parts['hard']:.4f}",
                soft=f"{parts['soft']:.4f}",
            )

        val_loss, threshold, val_f1, y_true, probs = evaluate(model, valid_loader, device)
        elapsed_epoch = time.time() - epoch_started
        epoch_times.append(elapsed_epoch)
        eta = np.mean(epoch_times) * (args.epochs - epoch - 1)
        print(
            f"epoch {epoch + 1}: train_loss={np.mean(running):.5f} "
            f"val_loss={val_loss:.5f} val_f1={val_f1:.5f} "
            f"threshold={threshold:.4f} eta_min={eta / 60:.1f}"
        )

        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "threshold": threshold, "epoch": epoch + 1}
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "img_size": args.img_size,
                    "threshold": threshold,
                    "f1": val_f1,
                    "epoch": epoch + 1,
                    "mean": MEAN,
                    "std": STD,
                    "uses_bottle_type_metadata": True,
                    "uses_roi_crop": True,
                },
                best_path,
            )
            np.save(out_dir / "valid_probs_final_effnetv2_s.npy", probs)
            np.save(out_dir / "valid_targets_final_effnetv2_s.npy", y_true)
            print(f"  saved best checkpoint: {best_path}")
        empty_device_cache(device)

    total_minutes = (time.time() - started) / 60
    onnx_path = export_onnx(best_path, Path(args.onnx_path), args.img_size)
    metadata_path = write_metadata(out_dir, onnx_path, best, args)
    summary = {
        "best_checkpoint": str(best_path),
        "onnx_path": str(onnx_path),
        "metadata_path": str(metadata_path),
        "best": best,
        "minutes": total_minutes,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }
    (out_dir / "final_student_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"finished in {total_minutes:.1f} min")
    print(f"best f1={best['f1']:.5f} threshold={best['threshold']:.4f} epoch={best['epoch']}")
    print(f"onnx: {onnx_path}")
    return summary


def export_onnx(checkpoint_path: Path, onnx_path: Path, img_size: int) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EffNetV2Student(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dummy_image = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    dummy_btype = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_image, dummy_btype),
        onnx_path,
        input_names=["image", "btype_id"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "btype_id": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
    )
    return onnx_path


def write_metadata(out_dir: Path, onnx_path: Path, best: dict, args: argparse.Namespace) -> Path:
    metadata = {
        "model_name": MODEL_NAME,
        "model_file": onnx_path.name,
        "img_size": int(args.img_size),
        "batch_size": int(args.infer_batch_size),
        "threshold": float(best.get("threshold", 0.5)),
        "output_activation": "sigmoid",
        "mean": MEAN,
        "std": STD,
        "input_names": ["image", "btype_id"],
        "onnx_opset": 18,
        "uses_roi_crop": True,
        "uses_bottle_type_metadata": True,
        "training_label_source": "COCO-derived target from train_annotations.json via outputs/preprocessing/final_train.csv",
        "teacher_distillation": "ConvNeXt Small + MaxViT Tiny soft probabilities when available",
        "robust_noisy_label_weighting": "hard-label loss is downweighted when teacher ensemble strongly disagrees with COCO target",
        "validation_f1": float(best.get("f1", -1.0)),
        "validation_epoch": int(best.get("epoch", -1)),
    }
    model_dir = ROOT / "final_submission" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for path in [out_dir / "model_metadata.json", model_dir / "model_metadata.json"]:
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir / "model_metadata.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the final EfficientNetV2-S student and export ONNX.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_OUT / "final_effnetv2_s.onnx")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--distill-alpha", type=float, default=0.50)
    parser.add_argument("--btype-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--export-only", type=Path, default=None, help="Export ONNX from an existing checkpoint and exit.")
    args = parser.parse_args()

    if args.export_only:
        best = torch.load(args.export_only, map_location="cpu")
        checkpoint_best = {
            "f1": float(best.get("f1", -1.0)),
            "threshold": float(best.get("threshold", 0.5)),
            "epoch": int(best.get("epoch", -1)),
        }
        onnx_path = export_onnx(args.export_only, args.onnx_path, int(best.get("img_size", args.img_size)))
        metadata_path = write_metadata(Path(args.out_dir), onnx_path, checkpoint_best, args)
        print(f"Exported: {onnx_path}")
        print(f"Metadata: {metadata_path}")
        return 0

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
