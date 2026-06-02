import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


TEACHERS = [
    {
        "tag": "lb_convnext_small",
        "title": "ConvNeXt Small",
        "model_name": "convnext_small.fb_in22k_ft_in1k",
        "notebook": "lb_convnext_small_feature_export.ipynb",
    },
    {
        "tag": "lb_maxvit",
        "title": "MaxViT Tiny",
        "model_name": "maxvit_rmlp_tiny_rw_256.sw_in1k",
        "notebook": "lb_maxvit_feature_export.ipynb",
    },
]


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


COMMON_IMPORTS = r"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
"""


PATHS_AND_CONFIG = r"""
def find_project_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "README.md").exists() and (p / "data").exists():
            return p
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return here


def find_data_root(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "1st-krones-vision-ai-challenge",
        project_root / "data",
        project_root,
    ]
    if Path("/kaggle/input").exists():
        candidates.extend([
            Path("/kaggle/input/1st-krones-vision-ai-challenge"),
            Path("/kaggle/input/krones-challenges"),
            Path("/kaggle/input/krones-challenges/Krones_Challenges"),
            Path("/kaggle/input/krones-challenge"),
            Path("/kaggle/input/krones"),
        ])
        for root, dirs, files in os.walk("/kaggle/input"):
            dset = set(dirs)
            if {"train_images", "test_images"}.issubset(dset):
                candidates.append(Path(root))
    for cand in candidates:
        if (cand / "train_images").exists() and (cand / "test_images").exists():
            return cand
    raise FileNotFoundError("Could not find data root with train_images/ and test_images/.")


def find_file(project_root: Path, relative_path: str, filename: str | None = None) -> Path:
    candidates = [project_root / relative_path]
    if Path("/kaggle/working").exists():
        candidates.append(Path("/kaggle/working") / relative_path)
    for cand in candidates:
        if cand.exists():
            return cand

    if filename is None:
        filename = Path(relative_path).name
    search_roots = [project_root]
    if Path("/kaggle/input").exists():
        search_roots.append(Path("/kaggle/input"))
    if Path("/kaggle/working").exists():
        search_roots.append(Path("/kaggle/working"))
    for base in search_roots:
        for root, _, files in os.walk(base):
            if filename in files:
                return Path(root) / filename
    raise FileNotFoundError(f"Could not find {relative_path!r}.")


def find_teacher_dir(project_root: Path, tag: str) -> Path:
    candidates = [
        project_root / "outputs" / tag,
        project_root / tag,
    ]
    if Path("/kaggle/working").exists():
        candidates.extend([
            Path("/kaggle/working") / "outputs" / tag,
            Path("/kaggle/working") / tag,
        ])
    if Path("/kaggle/input").exists():
        for root, dirs, files in os.walk("/kaggle/input"):
            root_path = Path(root)
            if f"model_{tag}_p2_fold0.pth" in files:
                candidates.append(root_path)
            if tag in dirs:
                candidates.append(root_path / tag)
    for cand in candidates:
        if (cand / f"model_{tag}_p2_fold0.pth").exists():
            return cand
    raise FileNotFoundError(f"Could not find P2 checkpoints for {tag}.")


PROJECT_ROOT = find_project_root()
DATA = find_data_root(PROJECT_ROOT)
SAVE = find_teacher_dir(PROJECT_ROOT, VERSION_TAG)
OUT = PROJECT_ROOT / "outputs" / "feature_distillation" / VERSION_TAG
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_IMG = DATA / "train_images"
TEST_IMG = DATA / "test_images"
FINAL_TRAIN_CSV = find_file(PROJECT_ROOT, "outputs/preprocessing/final_train.csv", "final_train.csv")
FINAL_TEST_CSV = find_file(PROJECT_ROOT, "outputs/preprocessing/final_test.csv", "final_test.csv")
ACTIVE_LABELS_CSV = SAVE / f"labels_train_active_{VERSION_TAG}.csv"

SEED = 42
N_FOLDS = 5
N_BOTTLE_TYPES = 3
IMG_SIZE = 256
DROPOUT = 0.3
DROP_PATH_RATE_P1 = 0.2
BATCH_SIZE = 32 if torch.cuda.is_available() else 8
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
PHASES_TO_EXPORT = ["p2"]  # P2 is the final fine-tuned teacher. Add "p1" only if you explicitly need P1 features too.
EXPORT_TEST_FEATURES = True
SAVE_FLOAT16 = True
ALLOW_ZERO_FOR_MISSING_IMAGES = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA:", DATA)
print("SAVE:", SAVE)
print("OUT:", OUT)
print("DEVICE:", DEVICE)
print("PHASES_TO_EXPORT:", PHASES_TO_EXPORT)
"""


DATA_AND_AUX = r"""
def load_train_and_test_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    active = pd.read_csv(ACTIVE_LABELS_CSV)
    final_train = pd.read_csv(FINAL_TRAIN_CSV)
    final_test = pd.read_csv(FINAL_TEST_CSV)

    meta_cols = [
        c for c in [
            "image_id",
            "target_source",
            "target_coco",
            "target_original",
            "label_mismatch",
            "label_reason",
            "label_reason_category",
            "conditional_hits",
            "bottle_type",
            "roi_x",
            "roi_y",
            "roi_w",
            "roi_h",
            "target_corrected",
            "categories_present",
            "good_categories",
            "always_faulty_categories",
            "category_areas_json",
            "target_note",
        ]
        if c in final_train.columns
    ]
    train_df = active.merge(final_train[meta_cols], on="image_id", how="left")
    if "stem" not in train_df.columns:
        train_df["stem"] = train_df["image_id"].map(lambda x: Path(str(x)).stem)
    train_df["row_idx"] = np.arange(len(train_df), dtype=np.int64)
    train_df["target"] = train_df["target"].astype(np.float32)
    train_df["btype_id"] = train_df["btype_id"].fillna(-1).astype(np.int64)

    sample_path = DATA / "sample_submission.csv"
    if sample_path.exists() and "image_id" in pd.read_csv(sample_path, nrows=1).columns:
        sample = pd.read_csv(sample_path)[["image_id"]]
        test_df = sample.merge(final_test, on="image_id", how="left")
    else:
        test_df = final_test.copy()
    if "stem" not in test_df.columns:
        test_df["stem"] = test_df["image_id"].map(lambda x: Path(str(x)).stem)
    test_df["row_idx"] = np.arange(len(test_df), dtype=np.int64)
    test_df["btype_id"] = test_df["btype_id"].fillna(-1).astype(np.int64)
    return train_df, test_df


def parse_area_dict(value) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def ordered_categories(train_df: pd.DataFrame) -> list[str]:
    cats = set()
    for value in train_df.get("category_areas_json", pd.Series(dtype=object)).fillna("{}"):
        cats.update(parse_area_dict(value).keys())
    if not cats:
        for value in train_df.get("categories_present", pd.Series(dtype=object)).fillna(""):
            cats.update([x.strip() for x in str(value).split("|") if x.strip()])
    cats = sorted(cats)
    if "No fault" in cats:
        cats = ["No fault"] + [c for c in cats if c != "No fault"]
    return cats


def export_auxiliary_targets(train_df: pd.DataFrame, out_dir: Path) -> dict:
    # The current teachers have binary heads, so these are not teacher logits.
    # They are label-side auxiliary signals built from COCO annotations.
    categories = ordered_categories(train_df)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n = len(train_df)
    c = len(categories)
    multihot = np.zeros((n, c), dtype=np.float32)
    areas = np.zeros((n, c), dtype=np.float32)

    for i, row in train_df.iterrows():
        area_dict = parse_area_dict(row.get("category_areas_json", "{}"))
        if not area_dict:
            present = [x.strip() for x in str(row.get("categories_present", "")).split("|") if x.strip()]
            area_dict = {name: 1.0 for name in present}
        for name, area in area_dict.items():
            if name in cat_to_idx and area > 0:
                j = cat_to_idx[name]
                multihot[i, j] = 1.0
                areas[i, j] = float(area)

    if "No fault" in cat_to_idx:
        no_fault_idx = cat_to_idx["No fault"]
        empty = multihot.sum(axis=1) == 0
        multihot[empty, no_fault_idx] = 1.0
        areas[empty, no_fault_idx] = 1.0

    primary = areas.argmax(axis=1).astype(np.int64)
    fault_present = train_df["target"].astype(np.float32).to_numpy().reshape(-1, 1)
    aux27 = np.concatenate([multihot, fault_present], axis=1).astype(np.float32)
    aux27_names = categories + ["Fault present"]

    np.save(out_dir / f"aux_coco_multihot_{VERSION_TAG}.npy", multihot)
    np.save(out_dir / f"aux_coco_area_{VERSION_TAG}.npy", areas)
    np.save(out_dir / f"aux_coco_primary_{VERSION_TAG}.npy", primary)
    np.save(out_dir / f"aux27_multitarget_{VERSION_TAG}.npy", aux27)
    (out_dir / f"aux_coco_categories_{VERSION_TAG}.json").write_text(json.dumps({
        "category_names": categories,
        "category_count": len(categories),
        "aux27_signal_names": aux27_names,
        "aux27_signal_count": len(aux27_names),
        "note": (
            "Current COCO preprocessing exposes the listed category names. "
            "aux27_multitarget appends Fault present to the COCO category multihot "
            "so the student has 27 auxiliary signals without inventing a fake category."
        ),
    }, indent=2))

    manifest = train_df[[
        "row_idx",
        "image_id",
        "target",
        "btype_id",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
        "categories_present",
        "category_areas_json",
    ]].copy()
    manifest["aux_primary_idx"] = primary
    manifest["aux_primary_name"] = [categories[i] if categories else "" for i in primary]
    manifest.to_csv(out_dir / f"distill_train_manifest_{VERSION_TAG}.csv", index=False)

    return {
        "category_count": len(categories),
        "category_names": categories,
        "aux27_signal_count": len(aux27_names),
        "aux27_signal_names": aux27_names,
        "files": {
            "aux_coco_multihot": str(out_dir / f"aux_coco_multihot_{VERSION_TAG}.npy"),
            "aux_coco_area": str(out_dir / f"aux_coco_area_{VERSION_TAG}.npy"),
            "aux_coco_primary": str(out_dir / f"aux_coco_primary_{VERSION_TAG}.npy"),
            "aux27_multitarget": str(out_dir / f"aux27_multitarget_{VERSION_TAG}.npy"),
            "distill_train_manifest": str(out_dir / f"distill_train_manifest_{VERSION_TAG}.csv"),
        },
    }


train_df, test_df = load_train_and_test_frames()
aux_manifest = export_auxiliary_targets(train_df, OUT)

print("train rows:", len(train_df), "test rows:", len(test_df))
print("COCO categories:", aux_manifest["category_count"])
print("Auxiliary signals:", aux_manifest["aux27_signal_count"])
print(aux_manifest["aux27_signal_names"])
"""


DATASET_AND_MODEL = r"""
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

valid_tf = A.Compose([
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])


def crop_roi(img: np.ndarray, row, padding: int = 8) -> np.ndarray:
    vals = []
    for col in ["roi_x", "roi_y", "roi_w", "roi_h"]:
        value = row.get(col, np.nan)
        vals.append(float(value) if pd.notna(value) else math.nan)
    if any(math.isnan(v) for v in vals):
        return img
    x, y, w, h = vals
    ih, iw = img.shape[:2]
    x1 = max(0, int(x) - padding)
    y1 = max(0, int(y) - padding)
    x2 = min(iw, int(x + w) + padding)
    y2 = min(ih, int(y + h) + padding)
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def resolve_image_path(img_dir: Path, image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
    candidates = [img_dir / str(image_id)]
    candidates.extend([img_dir / f"{stem}{ext}" for ext in [".png", ".jpg", ".jpeg"]])
    for p in candidates:
        if p.exists():
            return p
    return None


class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform, img_size: int = IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        path = resolve_image_path(self.img_dir, image_id)
        if path is None:
            if not ALLOW_ZERO_FOR_MISSING_IMAGES:
                raise FileNotFoundError(f"Missing image: {image_id} under {self.img_dir}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cv2 could not read {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_roi(img, row)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_t = self.transform(image=img)["image"]
        return {
            "image": img_t,
            "row_idx": int(row["row_idx"]),
            "image_id": image_id,
            "btype_id": int(row.get("btype_id", -1)),
        }


class BottleClsMT(nn.Module):
    def __init__(
        self,
        name: str = MODEL_NAME,
        n_btypes: int = N_BOTTLE_TYPES,
        dropout: float = DROPOUT,
        drop_path_rate: float = DROP_PATH_RATE_P1,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        f = self.backbone.num_features
        self.gabor_head = None
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(f, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, 1),
        )
        self.btype_head = nn.Sequential(
            nn.Dropout(dropout * 0.3),
            nn.Linear(f, n_btypes),
        )
        self._feat_map = None

    def forward_features(self, x):
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, x, gabor_x=None, return_btype=False):
        feat = self.forward_features(x)
        cls = self.cls_head(feat).view(-1)
        if return_btype:
            return cls, self.btype_head(feat)
        return cls


def load_teacher(phase: str, fold: int) -> BottleClsMT:
    ckpt_path = SAVE / f"model_{VERSION_TAG}_{phase}_fold{fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = BottleClsMT(pretrained=False).to(DEVICE)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [k for k in missing if k.startswith(("backbone.", "cls_head.", "btype_head."))]
    if critical_missing or unexpected:
        print("Missing keys:", missing[:20])
        print("Unexpected keys:", unexpected[:20])
        raise RuntimeError(f"Checkpoint did not match model cleanly: {ckpt_path}")
    model.eval()
    return model


def make_loader(df: pd.DataFrame, img_dir: Path, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        FeatureDataset(df, img_dir, valid_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


print("Model wrapper ready:", MODEL_NAME)
print("First checkpoint:", SAVE / f"model_{VERSION_TAG}_p2_fold0.pth")
"""


EXPORT_HELPERS = r"""
@torch.inference_mode()
def extract_features_and_logits(model: BottleClsMT, loader: DataLoader, desc: str):
    features = []
    logits = []
    btype_logits = []
    row_indices = []
    image_ids = []
    btypes = []

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(DEVICE, non_blocking=True)
        feat = model.forward_features(x)
        logit = model.cls_head(feat).view(-1)
        bt_logit = model.btype_head(feat)

        features.append(feat.detach().cpu())
        logits.append(logit.detach().cpu())
        btype_logits.append(bt_logit.detach().cpu())
        row_indices.extend(batch["row_idx"].numpy().tolist())
        image_ids.extend(list(batch["image_id"]))
        btypes.extend(batch["btype_id"].numpy().tolist())

    features = torch.cat(features, dim=0).numpy()
    logits = torch.cat(logits, dim=0).numpy().astype(np.float32)
    btype_logits = torch.cat(btype_logits, dim=0).numpy().astype(np.float32)
    row_indices = np.asarray(row_indices, dtype=np.int64)
    btypes = np.asarray(btypes, dtype=np.int64)
    if SAVE_FLOAT16:
        features = features.astype(np.float16)
    else:
        features = features.astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {
        "features": features,
        "logits": logits,
        "probs": probs.astype(np.float32),
        "btype_logits": btype_logits,
        "row_indices": row_indices,
        "image_ids": image_ids,
        "btype_id": btypes,
    }


def save_np(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    print(path.name, arr.shape, arr.dtype)


def export_phase(phase: str) -> dict:
    print(f"\n=== Exporting {VERSION_TAG} {phase.upper()} features ===")
    phase_manifest = {
        "version_tag": VERSION_TAG,
        "model_name": MODEL_NAME,
        "phase": phase,
        "folds": [],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    full_features = None
    full_logits = np.full(len(train_df), np.nan, dtype=np.float32)
    full_probs = np.full(len(train_df), np.nan, dtype=np.float32)
    full_btype_logits = np.full((len(train_df), N_BOTTLE_TYPES), np.nan, dtype=np.float32)

    test_feature_sum = None
    test_logit_sum = np.zeros(len(test_df), dtype=np.float32)
    test_prob_sum = np.zeros(len(test_df), dtype=np.float32)

    for fold in range(N_FOLDS):
        fold_start = time.time()
        idx_path = SAVE / f"oof_idx_{VERSION_TAG}_{phase}_fold{fold}.npy"
        if not idx_path.exists():
            raise FileNotFoundError(idx_path)
        val_idx = np.load(idx_path).astype(np.int64)
        val_df = train_df.iloc[val_idx].copy()
        val_df["row_idx"] = val_idx

        model = load_teacher(phase, fold)
        val_out = extract_features_and_logits(
            model,
            make_loader(val_df, TRAIN_IMG),
            desc=f"{phase} fold {fold} train-oof",
        )

        if full_features is None:
            feature_dim = int(val_out["features"].shape[1])
            full_dtype = np.float16 if SAVE_FLOAT16 else np.float32
            full_features = np.zeros((len(train_df), feature_dim), dtype=full_dtype)
            phase_manifest["feature_dim"] = feature_dim

        rows = val_out["row_indices"]
        full_features[rows] = val_out["features"]
        full_logits[rows] = val_out["logits"]
        full_probs[rows] = val_out["probs"]
        full_btype_logits[rows] = val_out["btype_logits"]

        fold_prefix = OUT / f"{VERSION_TAG}_{phase}_fold{fold}"
        save_np(fold_prefix.with_name(f"features_{VERSION_TAG}_{phase}_fold{fold}_oof.npy"), val_out["features"])
        save_np(fold_prefix.with_name(f"features_idx_{VERSION_TAG}_{phase}_fold{fold}_oof.npy"), rows)
        save_np(fold_prefix.with_name(f"binary_logits_{VERSION_TAG}_{phase}_fold{fold}_oof.npy"), val_out["logits"])
        save_np(fold_prefix.with_name(f"binary_probs_{VERSION_TAG}_{phase}_fold{fold}_oof.npy"), val_out["probs"])
        save_np(fold_prefix.with_name(f"btype_logits_{VERSION_TAG}_{phase}_fold{fold}_oof.npy"), val_out["btype_logits"])

        fold_info = {
            "fold": fold,
            "checkpoint": str(SAVE / f"model_{VERSION_TAG}_{phase}_fold{fold}.pth"),
            "oof_count": int(len(rows)),
            "feature_dim": int(val_out["features"].shape[1]),
            "elapsed_sec_oof": round(time.time() - fold_start, 2),
        }

        if EXPORT_TEST_FEATURES:
            test_out = extract_features_and_logits(
                model,
                make_loader(test_df, TEST_IMG),
                desc=f"{phase} fold {fold} test",
            )
            test_feat = test_out["features"]
            if test_feature_sum is None:
                test_feature_sum = test_feat.astype(np.float32)
            else:
                test_feature_sum += test_feat.astype(np.float32)
            test_logit_sum += test_out["logits"]
            test_prob_sum += test_out["probs"]
            save_np(OUT / f"features_{VERSION_TAG}_{phase}_fold{fold}_test.npy", test_feat)
            save_np(OUT / f"binary_logits_{VERSION_TAG}_{phase}_fold{fold}_test.npy", test_out["logits"])
            save_np(OUT / f"binary_probs_{VERSION_TAG}_{phase}_fold{fold}_test.npy", test_out["probs"])
            fold_info["test_count"] = int(len(test_df))
            fold_info["elapsed_sec_total"] = round(time.time() - fold_start, 2)

        phase_manifest["folds"].append(fold_info)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if full_features is None:
        raise RuntimeError("No features were exported.")
    if np.isnan(full_logits).any():
        missing = int(np.isnan(full_logits).sum())
        raise RuntimeError(f"OOF export incomplete: {missing} rows missing.")

    save_np(OUT / f"features_{VERSION_TAG}_{phase}_oof.npy", full_features)
    save_np(OUT / f"binary_logits_{VERSION_TAG}_{phase}_oof.npy", full_logits)
    save_np(OUT / f"binary_probs_{VERSION_TAG}_{phase}_oof.npy", full_probs)
    save_np(OUT / f"btype_logits_{VERSION_TAG}_{phase}_oof.npy", full_btype_logits)

    if EXPORT_TEST_FEATURES and test_feature_sum is not None:
        test_feature_mean = (test_feature_sum / N_FOLDS).astype(np.float16 if SAVE_FLOAT16 else np.float32)
        test_logit_mean = test_logit_sum / N_FOLDS
        test_prob_mean = test_prob_sum / N_FOLDS
        save_np(OUT / f"features_{VERSION_TAG}_{phase}_test_mean.npy", test_feature_mean)
        save_np(OUT / f"binary_logits_{VERSION_TAG}_{phase}_test_mean.npy", test_logit_mean)
        save_np(OUT / f"binary_probs_{VERSION_TAG}_{phase}_test_mean.npy", test_prob_mean)

    phase_manifest["files"] = {
        "features_oof": str(OUT / f"features_{VERSION_TAG}_{phase}_oof.npy"),
        "binary_logits_oof": str(OUT / f"binary_logits_{VERSION_TAG}_{phase}_oof.npy"),
        "binary_probs_oof": str(OUT / f"binary_probs_{VERSION_TAG}_{phase}_oof.npy"),
        "btype_logits_oof": str(OUT / f"btype_logits_{VERSION_TAG}_{phase}_oof.npy"),
        "features_test_mean": str(OUT / f"features_{VERSION_TAG}_{phase}_test_mean.npy"),
        "binary_logits_test_mean": str(OUT / f"binary_logits_{VERSION_TAG}_{phase}_test_mean.npy"),
        "binary_probs_test_mean": str(OUT / f"binary_probs_{VERSION_TAG}_{phase}_test_mean.npy"),
    }
    return phase_manifest
"""


RUN_EXPORT = r"""
all_phase_manifests = []
for phase in PHASES_TO_EXPORT:
    all_phase_manifests.append(export_phase(phase))

final_manifest = {
    "version_tag": VERSION_TAG,
    "title": TEACHER_TITLE,
    "model_name": MODEL_NAME,
    "teacher_dir": str(SAVE),
    "output_dir": str(OUT),
    "phases": all_phase_manifests,
    "auxiliary_targets": aux_manifest,
    "notes": [
        "This notebook performs feature export only; it does not train or fine-tune.",
        "The saved P2 checkpoints are binary teachers. True 27-way teacher logits are not available from these checkpoints.",
        "Use features_*_p2_oof.npy plus aux27_multitarget_*.npy for feature KD, RKD/SimKD, and subclass/proxy auxiliary supervision.",
    ],
}

manifest_path = OUT / f"feature_distill_manifest_{VERSION_TAG}.json"
manifest_path.write_text(json.dumps(final_manifest, indent=2))
print("\nWrote manifest:", manifest_path)
print(json.dumps({
    "version_tag": VERSION_TAG,
    "output_dir": str(OUT),
    "phases": [m["phase"] for m in all_phase_manifests],
    "auxiliary_signal_count": aux_manifest["aux27_signal_count"],
    "feature_dim": all_phase_manifests[-1].get("feature_dim"),
}, indent=2))
"""


LOSS_REFERENCE = r"""
def feature_kd_loss(student_feat, teacher_feat, projector=None, normalize=True):
    if projector is not None:
        student_feat = projector(student_feat)
    if normalize:
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)
    return F.mse_loss(student_feat, teacher_feat)


def rkd_distance_loss(student_feat, teacher_feat, eps=1e-6):
    def pdist(x):
        d = torch.cdist(x, x, p=2)
        positive = d[d > eps]
        mean = positive.mean() if positive.numel() else d.mean().clamp_min(eps)
        return d / mean.clamp_min(eps)
    return F.smooth_l1_loss(pdist(student_feat), pdist(teacher_feat))


def binary_kd_loss(student_logits, teacher_logits, temperature=2.0):
    t = float(temperature)
    return F.binary_cross_entropy_with_logits(student_logits / t, torch.sigmoid(teacher_logits / t)) * (t * t)


def aux_multitarget_loss(student_aux_logits, aux27_targets, pos_weight=None):
    return F.binary_cross_entropy_with_logits(student_aux_logits, aux27_targets, pos_weight=pos_weight)


print("Reference KD losses loaded. Keep the aux head for training, discard it for final ONNX export.")
"""


def make_notebook(spec: dict):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(
            f"""
# Feature Distillation Continuation: {spec["title"]}

This notebook continues after P2 fine-tuning. It does not train again.

It loads the existing `{spec["tag"]}` fold checkpoints, re-runs inference once, and exports:

- penultimate teacher features for feature KD / SimKD / RKD
- binary teacher logits and probabilities
- bottle-type auxiliary logits
- COCO-derived auxiliary subclass/proxy labels

Important: these checkpoints were trained as binary teachers. They cannot emit true 27-way teacher logits unless a 27-way auxiliary teacher head is trained later. The COCO auxiliary arrays exported here are label-side proxy targets for the student.
"""
        ),
        code(
            f"""
VERSION_TAG = {spec["tag"]!r}
TEACHER_TITLE = {spec["title"]!r}
MODEL_NAME = {spec["model_name"]!r}
"""
        ),
        code(COMMON_IMPORTS),
        code(PATHS_AND_CONFIG),
        md(
            """
## COCO Auxiliary Targets

The workaround for binary distillation is to give the student richer side targets.
This cell converts COCO category annotations into a multihot subclass target plus a 27-signal array: actual COCO categories plus `Fault present`.
"""
        ),
        code(DATA_AND_AUX),
        md(
            """
## Dataset And Teacher Model

The model wrapper mirrors the saved P1/P2 notebooks: timm backbone with `num_classes=0`, one binary head, and one bottle-type auxiliary head.
"""
        ),
        code(DATASET_AND_MODEL),
        md(
            """
## Export Teacher Features

Run this cell to export all P2 fold features. It uses the existing fold checkpoints and the stored OOF indices, so every training image receives the feature vector from the fold where that image was validation data.
"""
        ),
        code(EXPORT_HELPERS),
        code(RUN_EXPORT),
        md(
            """
## KD Loss Reference

These functions are not required for the export. They are here so the next student-training notebook can consume the exported arrays cleanly.
"""
        ),
        code(LOSS_REFERENCE),
    ]
    return nb


def main() -> int:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in TEACHERS:
        nb = make_notebook(spec)
        out = out_dir / spec["notebook"]
        nbf.write(nb, out)
        written.append(str(out.relative_to(ROOT)))
        print(f"Wrote {out}")

    manifest = {
        "purpose": "Feature-distillation continuation notebooks generated after P2 teacher fine-tuning.",
        "notebooks": written,
        "teachers": TEACHERS,
        "outputs": "outputs/feature_distillation/<teacher_tag>/",
        "default_phase": "p2",
    }
    manifest_path = ROOT / "outputs" / "feature_distillation_notebooks.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
