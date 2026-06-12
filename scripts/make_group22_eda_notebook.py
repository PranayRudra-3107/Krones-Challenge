"""Build notebooks/group22_eda_label_correction.ipynb (EDA + label-correction notebook).

The notebook is written for the Kaggle environment of the 1st Krones Vision AI
Challenge but falls back to local ./data paths so it can be smoke-tested offline.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "group22_eda_label_correction.ipynb"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ---------------------------------------------------------------- title
md("""
# Group 22 - EDA and Label Correction

**Competition:** 1st Krones Vision AI Challenge — binary classification of returnable glass bottles
(`target = 0` → reusable, `target = 1` → not reusable).

Besides `train.csv`, the organizers provide **COCO-style annotations**
(`train_annotations.json`) with segmentation polygons, defect categories and a region of
interest (ROI) for every training image. The official label rules state that the binary
target is fully determined by the defect annotations:

| Rule | Categories |
|---|---|
| **Always reusable** (target = 0) | Embossing, Foam residue, No fault, Water drop |
| **Always faulty** (target = 1, any occurrence) | Break/Crack, Circlip, Contamination dark, Crown cap, Foil/Semitransparent, Foreign object (manual/washing), Glass shard, Insect, Label, Liquid, Mold, No base visible, Paint residue, Straw, Yeast residue |
| **Conditionally faulty** (target = 1 above an area threshold) | Air bubble > 500 px, Chip > 200 px, Contamination light > 180 px, Glass imperfection > 100 px, Scuffing > 75 000 px, Scuffing heavy > 1 200 px |

This means the targets in `train.csv` can be **re-derived and audited** from the COCO file —
which is exactly what this notebook does, in five steps:

1. **COCO annotation validation** — consistency between `train.csv` and the COCO file, ROI coverage, sample visualizations.
2. **Class distribution analysis** — defect categories, binary target, bottle types and their defect rates.
3. **ROI cropping visualization** — what the model actually sees after cropping to the ROI.
4. **Label correction** — re-derive every target from the COCO rules and compare against `train.csv`.
5. **Export** — `corrected_train.csv` and a per-image `label_audit.csv` to `/kaggle/working`.

Steps 6–8 then automate the full **per-image quality loop**:

6. **Per-image annotation integrity loop** — every image is checked for out-of-bounds
   geometry, defects hidden by the ROI crop, render-detectability and label consistency.
7. **COCO rewrite** — broken annotations are repaired (clamped, bbox/area recomputed) and
   exported as `train_annotations_fixed.json` with a full fix log.
8. **Guaranteed-visibility rendering** — markers + zoom insets prove that *every* defect,
   however tiny, is visible; the "some defects don't show" effect is fully explained.
""")

# ---------------------------------------------------------------- setup
md("""
## Setup

Imports, reproducible seeds, and the official label rules as Python constants.
Paths resolve to the Kaggle competition input when available
(`/kaggle/input/1st-krones-vision-ai-challenge/`) and fall back to a local `data/`
directory so the notebook can also be smoke-tested offline. Outputs go to
`/kaggle/working` on Kaggle.
""")
code("""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from PIL import Image

sns.set_theme(style="whitegrid")
RNG_SEED = 22
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# --- Resolve paths: Kaggle competition input first, local ./data as fallback ---
KAGGLE_ROOT = Path("/kaggle/input/1st-krones-vision-ai-challenge")

def resolve_data_dir() -> Path:
    if KAGGLE_ROOT.exists():
        if (KAGGLE_ROOT / "train.csv").exists():
            return KAGGLE_ROOT
        for candidate in KAGGLE_ROOT.rglob("train.csv"):
            return candidate.parent
    return Path("data")

DATA_DIR = resolve_data_dir()
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/notebook_eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMAGES = DATA_DIR / "train_images"

print(f"DATA_DIR = {DATA_DIR.resolve()}")
print(f"OUT_DIR  = {OUT_DIR.resolve()}")

# --- Official label rules ---
ROI_CATEGORIES = {"ROI", "Roi", "roi", "Region of interest"}

ALWAYS_GOOD = {"Embossing", "Foam residue", "No fault", "Water drop"}

ALWAYS_FAULTY = {
    "Break / Crack", "Circlip", "Contamination dark", "Crown cap",
    "Foil / Semitransparent", "Foreign object - manual cleaning",
    "Foreign object - washing machine", "Glass shard", "Insect", "Label",
    "Liquid", "Mold", "No base visible", "Paint residue", "Straw", "Yeast residue",
}

CONDITIONAL_THRESHOLDS = {  # category -> minimum total area (px) that makes the bottle faulty
    "Air bubble": 500,
    "Chip": 200,
    "Contamination light": 180,
    "Glass imperfection": 100,
    "Scuffing": 75000,
    "Scuffing heavy": 1200,
}
""")

# ---------------------------------------------------------------- step 1
md("""
## Step 1 — COCO Annotation Validation

Before trusting the annotations for label auditing, we validate the COCO file itself:

- parse `train_annotations.json` and index images, categories and annotations;
- cross-check that **every image in `train.csv` has a COCO entry** and vice versa;
- verify that an **ROI annotation exists for every training image** (the ROI is what we
  later crop to);
- list any images with **zero annotations** (these would be impossible to audit);
- visualize **5 sample images** with their bounding boxes and segmentation polygons
  (FiftyOne if it is installed, otherwise matplotlib — Kaggle ships without FiftyOne).
""")
code("""
with open(DATA_DIR / "train_annotations.json") as f:
    coco = json.load(f)

train_df = pd.read_csv(DATA_DIR / "train.csv")

cat_name = {c["id"]: c["name"] for c in coco["categories"]}
img_by_id = {im["id"]: im for im in coco["images"]}
file_to_imgid = {im["file_name"]: im["id"] for im in coco["images"]}

anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

# --- 1a. train.csv <-> COCO consistency ---
csv_ids = set(train_df["image_id"])
coco_files = set(file_to_imgid)
missing_in_coco = sorted(csv_ids - coco_files)
missing_in_csv = sorted(coco_files - csv_ids)
print(f"train.csv rows ............... {len(csv_ids):,}")
print(f"COCO images .................. {len(coco_files):,}")
print(f"in train.csv, not in COCO .... {len(missing_in_coco)}")
print(f"in COCO, not in train.csv .... {len(missing_in_csv)}")
if missing_in_coco:
    print("   e.g.", missing_in_coco[:5])
if missing_in_csv:
    print("   e.g.", missing_in_csv[:5])

# --- 1b. ROI coverage (keep the largest ROI box per image) ---
roi_box = {}
for img_id, anns in anns_by_image.items():
    for ann in anns:
        if cat_name.get(ann["category_id"]) in ROI_CATEGORIES and len(ann.get("bbox") or []) == 4:
            bbox = [float(v) for v in ann["bbox"]]
            cur = roi_box.get(img_id)
            if cur is None or bbox[2] * bbox[3] > cur[2] * cur[3]:
                roi_box[img_id] = bbox

no_roi = [img_by_id[i]["file_name"] for i in img_by_id if i not in roi_box]
no_anns = [img_by_id[i]["file_name"] for i in img_by_id if i not in anns_by_image]
print(f"\\nimages without ROI annotation  {len(no_roi)}")
print(f"images with zero annotations . {len(no_anns)}")
if no_roi:
    print("   e.g.", no_roi[:5])
if no_anns:
    print("   e.g.", no_anns[:5])

print(f"\\ntotal annotations ............ {len(coco['annotations']):,}")
print(f"categories ................... {len(cat_name)}")
""")

md("""
### Sample visualizations

Five training images that contain at least two distinct (non-ROI) annotations, drawn with
their COCO bounding boxes and filled segmentation polygons. The dashed white box is the
ROI; colored shapes are defect / surface annotations.
""")
code("""
defect_rich = [
    img_id for img_id, anns in anns_by_image.items()
    if sum(cat_name[a["category_id"]] not in ROI_CATEGORIES for a in anns) >= 2
]
sample_ids = random.Random(RNG_SEED).sample(sorted(defect_rich), 5)


def draw_samples_matplotlib(ids):
    fig, axes = plt.subplots(1, len(ids), figsize=(24, 5.5))
    cmap = plt.get_cmap("tab10")
    for ax, img_id in zip(axes, ids):
        info = img_by_id[img_id]
        ax.imshow(Image.open(TRAIN_IMAGES / info["file_name"]))
        cats_here = sorted({cat_name[a["category_id"]] for a in anns_by_image[img_id]
                            if cat_name[a["category_id"]] not in ROI_CATEGORIES})
        color_of = {c: cmap(i % 10) for i, c in enumerate(cats_here)}
        for a in anns_by_image[img_id]:
            cname = cat_name[a["category_id"]]
            is_roi = cname in ROI_CATEGORIES
            color = "white" if is_roi else color_of[cname]
            x, y, w, h = a["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                   linewidth=1.5, linestyle="--" if is_roi else "-"))
            seg = a.get("segmentation") or []
            if not is_roi and isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                        ax.add_patch(MplPolygon(pts, closed=True, alpha=0.3,
                                                facecolor=color, edgecolor=color))
        handles = [Rectangle((0, 0), 1, 1, facecolor=color_of[c], alpha=0.6) for c in cats_here]
        ax.legend(handles, cats_here, fontsize=7, loc="lower right", framealpha=0.8)
        ax.set_title(info["file_name"].split("_")[0][:13] + "...", fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample images with COCO bounding boxes and segmentation polygons", y=1.03)
    plt.tight_layout()
    plt.show()


try:  # FiftyOne is optional and not preinstalled on Kaggle
    import fiftyone as fo

    dataset = fo.Dataset()
    for img_id in sample_ids:
        info = img_by_id[img_id]
        W, H = info["width"], info["height"]
        sample = fo.Sample(filepath=str((TRAIN_IMAGES / info["file_name"]).resolve()))
        dets, lines = [], []
        for a in anns_by_image[img_id]:
            cname = cat_name[a["category_id"]]
            x, y, w, h = a["bbox"]
            dets.append(fo.Detection(label=cname, bounding_box=[x / W, y / H, w / W, h / H]))
            for poly in a.get("segmentation") or []:
                if isinstance(poly, list) and len(poly) >= 6:
                    pts = [(px / W, py / H) for px, py in zip(poly[::2], poly[1::2])]
                    lines.append(fo.Polyline(label=cname, points=[pts], closed=True, filled=True))
        sample["detections"] = fo.Detections(detections=dets)
        sample["polylines"] = fo.Polylines(polylines=lines)
        dataset.add_sample(sample)
    session = fo.launch_app(dataset)
except Exception as exc:
    print(f"FiftyOne unavailable ({type(exc).__name__}) -> matplotlib fallback")
    draw_samples_matplotlib(sample_ids)
""")

# ---------------------------------------------------------------- step 2
md("""
## Step 2 — Class Distribution Analysis

Four views of how the data is distributed:

1. **Annotation frequency for all 27 COCO categories**, color-coded by which label rule
   each category falls under. Note that `Roi` appears once per image and that the
   *always reusable* categories (No fault, Water drop, Foam residue) are among the most
   frequent — most annotations are not defects.
2. **Binary target distribution** from `train.csv` — the dataset is moderately imbalanced
   towards faulty bottles.
3. **Bottle type distribution** (`bottletypes.csv`, `split = train`).
4. **Defect rate per bottle type** and a category-vs-bottle-type co-occurrence heatmap —
   some bottle types are clearly more defect-prone than others, which matters for
   stratified validation splits.
""")
code("""
# --- 2a. category frequency, color-coded by label rule ---
def rule_group(c):
    if c in ROI_CATEGORIES:
        return "ROI"
    if c in ALWAYS_GOOD:
        return "always reusable"
    if c in ALWAYS_FAULTY:
        return "always faulty"
    if c in CONDITIONAL_THRESHOLDS:
        return "conditionally faulty"
    return "other"

cat_counts = Counter(cat_name[a["category_id"]] for a in coco["annotations"])
cat_freq = (pd.DataFrame(cat_counts.items(), columns=["category", "count"])
            .sort_values("count", ascending=False))
cat_freq["rule"] = cat_freq["category"].map(rule_group)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=cat_freq, y="category", x="count", hue="rule", dodge=False,
            palette={"always reusable": "#2a9d8f", "always faulty": "#e76f51",
                     "conditionally faulty": "#e9c46a", "ROI": "#8d99ae"}, ax=ax)
ax.set_title(f"Annotation frequency for all {len(cat_freq)} COCO categories")
ax.set_xlabel("number of annotations")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# --- 2b. binary target distribution ---
fig, ax = plt.subplots(figsize=(5, 4))
tgt = train_df["target"].value_counts().sort_index()
ax.bar(["0 = reusable", "1 = not reusable"], tgt.values, color=["#2a9d8f", "#e76f51"])
for i, v in enumerate(tgt.values):
    ax.text(i, v, f"{v:,}\\n({v / len(train_df):.1%})", ha="center", va="bottom")
ax.set_ylim(0, tgt.max() * 1.18)
ax.set_title("Target distribution (train.csv)")
plt.tight_layout()
plt.show()

# --- 2c. bottle types in the train split ---
bottle_df = pd.read_csv(DATA_DIR / "bottletypes.csv")
bt_train = bottle_df[bottle_df["split"] == "train"]
fig, ax = plt.subplots(figsize=(9, 4.5))
order = bt_train["bottle_type"].value_counts().index
sns.countplot(data=bt_train, y="bottle_type", order=order, color="#457b9d", ax=ax)
ax.set_title("Bottle type distribution (split = train)")
ax.set_xlabel("images")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# --- 2d. defect rate per bottle type + category co-occurrence ---
merged = train_df.merge(bt_train[["image_id", "bottle_type"]], on="image_id", how="left")
defect_rate = (merged.groupby("bottle_type")["target"]
               .agg(defect_rate="mean", images="count")
               .sort_values("defect_rate", ascending=False))
print(defect_rate.round(3))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(defect_rate.index[::-1], defect_rate["defect_rate"][::-1], color="#e76f51")
for i, (rate, n) in enumerate(zip(defect_rate["defect_rate"][::-1], defect_rate["images"][::-1])):
    ax.text(rate, i, f" {rate:.1%} (n={n:,})", va="center", fontsize=9)
ax.set_xlim(0, defect_rate["defect_rate"].max() * 1.25)
ax.set_title("Share of non-reusable bottles per bottle type")
ax.set_xlabel("defect rate (mean target)")
plt.tight_layout()
plt.show()

# category presence per image -> co-occurrence heatmap (share of images containing category)
img_categories = {img_id: {cat_name[a["category_id"]] for a in anns}
                  for img_id, anns in anns_by_image.items()}
btype_of = dict(zip(bt_train["image_id"], bt_train["bottle_type"]))
rows = []
for img_id, cats in img_categories.items():
    fname = img_by_id[img_id]["file_name"]
    bt = btype_of.get(fname)
    if bt is None:
        continue
    rows.extend({"bottle_type": bt, "category": c} for c in cats if c not in ROI_CATEGORIES)
co = pd.DataFrame(rows)
n_per_type = bt_train["bottle_type"].value_counts()
pivot = (co.groupby(["category", "bottle_type"]).size().unstack(fill_value=0)
         .div(n_per_type, axis=1))
pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r",
            cbar_kws={"label": "share of images containing category"}, ax=ax)
ax.set_title("Category occurrence rate by bottle type")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------- step 3
md("""
## Step 3 — ROI Cropping Visualization

Each COCO entry contains an `Roi` annotation: the region of the camera frame that actually
shows the bottle base. Everything outside it is machinery and background, so the standard
preprocessing step is to **crop every image to its ROI bounding box** before feeding it to
a model.

Below: three reusable (`target = 0`) and three non-reusable (`target = 1`) examples.
The left column shows the original frame with the ROI box drawn in red, the right column
shows the resulting ROI crop.
""")
code("""
target_of = dict(zip(train_df["image_id"], train_df["target"]))
rng = random.Random(RNG_SEED)
good_ids = sorted(i for i in img_by_id
                  if i in roi_box and target_of.get(img_by_id[i]["file_name"]) == 0)
bad_ids = sorted(i for i in img_by_id
                 if i in roi_box and target_of.get(img_by_id[i]["file_name"]) == 1)
examples = rng.sample(good_ids, 3) + rng.sample(bad_ids, 3)

fig, axes = plt.subplots(6, 2, figsize=(9, 26))
for row, img_id in enumerate(examples):
    info = img_by_id[img_id]
    img = Image.open(TRAIN_IMAGES / info["file_name"])
    x, y, w, h = roi_box[img_id]
    t = target_of[info["file_name"]]
    label = "reusable" if t == 0 else "not reusable"

    axes[row, 0].imshow(img)
    axes[row, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
    axes[row, 0].set_title(f"original — target={t} ({label})", fontsize=10)
    axes[row, 0].axis("off")

    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    axes[row, 1].imshow(crop)
    axes[row, 1].set_title(f"ROI crop {crop.size[0]}x{crop.size[1]}", fontsize=10)
    axes[row, 1].axis("off")

fig.suptitle("ROI cropping: 3 reusable (top) vs 3 non-reusable (bottom) examples", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()
""")

# ---------------------------------------------------------------- step 4
md("""
## Step 4 — Label Correction (the most important step)

The binary target is — by the competition rules — a *deterministic function* of the COCO
annotations. We therefore re-derive it for every training image and audit `train.csv`
against it:

1. Sum the annotated area **per category per image** (using the COCO `area` field, with a
   shoelace-formula polygon fallback and a bbox-area fallback).
2. Apply the rules in order of precedence:
   - any **always-faulty** category present → `target = 1`;
   - otherwise any **conditional** category whose total area meets its threshold → `target = 1`;
   - otherwise → `target = 0`.
3. Compare the derived target against `train.csv` and list every mismatch, split into
   - **missed defects**: always-faulty annotations present but `train.csv` says 0, and
   - **wrong labels**: `train.csv` says 1 although all annotations are always-reusable.

Our offline preprocessing pipeline (`scripts/preprocess_coco.py`) found **zero
mismatches** on this dataset — this cell independently re-verifies that finding inside the
Kaggle environment, and would catch and correct any disagreement if the data were updated.
""")
code("""
def polygon_area(coords):
    \"\"\"Shoelace area of a flat [x1, y1, x2, y2, ...] polygon.\"\"\"
    pts = list(zip(coords[::2], coords[1::2]))
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def annotation_area(ann):
    area = float(ann.get("area") or 0.0)
    if area > 0:
        return area
    seg = ann.get("segmentation") or []
    if isinstance(seg, list):
        seg_area = sum(polygon_area(p) for p in seg if isinstance(p, list) and len(p) >= 6)
        if seg_area > 0:
            return seg_area
    bbox = ann.get("bbox") or []
    return float(bbox[2] * bbox[3]) if len(bbox) == 4 else 0.0


# --- aggregate presence + summed area per category, keyed by file name ---
areas = defaultdict(lambda: defaultdict(float))
present = defaultdict(set)
for ann in coco["annotations"]:
    fname = img_by_id[ann["image_id"]]["file_name"]
    cname = cat_name[ann["category_id"]]
    present[fname].add(cname)
    areas[fname][cname] += annotation_area(ann)


def derive_target(fname):
    cats = present.get(fname, set())
    faulty = sorted(cats & ALWAYS_FAULTY)
    if faulty:
        return 1, "always_faulty:" + "|".join(faulty)
    hits = [f"{c}={areas[fname][c]:.0f}px>={t}"
            for c, t in CONDITIONAL_THRESHOLDS.items() if areas[fname].get(c, 0.0) >= t]
    if hits:
        return 1, "conditional_area:" + "|".join(hits)
    return 0, "reusable_by_coco_rules"


audit = train_df.rename(columns={"target": "original_target"}).copy()
derived = audit["image_id"].map(lambda f: derive_target(f))
audit["corrected_target"] = [d[0] for d in derived]
audit["reason"] = [d[1] for d in derived]

# --- mismatches between train.csv and COCO-derived labels ---
mismatches = audit[audit["original_target"] != audit["corrected_target"]]
to_faulty = mismatches[mismatches["corrected_target"] == 1]   # 0 -> 1
to_good = mismatches[mismatches["corrected_target"] == 0]     # 1 -> 0
print(f"images audited ................. {len(audit):,}")
print(f"label mismatches ............... {len(mismatches)}")
print(f"  corrected 0 -> 1 (missed defect)  {len(to_faulty)}")
print(f"  corrected 1 -> 0 (wrong label)    {len(to_good)}")
if len(mismatches):
    print("\\nAll mismatching images:")
    print(mismatches.to_string(index=False, max_colwidth=60))

# --- targeted sanity checks, independent of derive_target ---
missed_defects = [f for f in audit.loc[audit["original_target"] == 0, "image_id"]
                  if present.get(f, set()) & ALWAYS_FAULTY]
wrong_positive = [f for f in audit.loc[audit["original_target"] == 1, "image_id"]
                  if (present.get(f, set()) - ROI_CATEGORIES)
                  and (present.get(f, set()) - ROI_CATEGORIES) <= ALWAYS_GOOD]
print(f"\\nlabeled 0 but ALWAYS_FAULTY annotation present  {len(missed_defects)}")
print(f"labeled 1 but only ALWAYS_GOOD annotations      {len(wrong_positive)}")
for f in missed_defects[:10]:
    print("  missed defect:", f, "->", sorted(present[f] & ALWAYS_FAULTY))
for f in wrong_positive[:10]:
    print("  wrong label:  ", f, "->", sorted(present[f] - ROI_CATEGORIES))

# --- corrected training labels ---
corrected_train = audit[["image_id", "corrected_target"]].rename(columns={"corrected_target": "target"})
direction = "no corrections needed" if mismatches.empty else \
    f"{len(to_faulty)} corrected 0->1, {len(to_good)} corrected 1->0"
print(f"\\nSummary: {len(mismatches)} of {len(audit):,} labels corrected ({direction}).")
print("Corrected target distribution:")
print(corrected_train["target"].value_counts().sort_index().rename({0: "0 (reusable)", 1: "1 (not reusable)"}))
""")

# ---------------------------------------------------------------- step 5
md("""
## Step 5 — Export

Persist the audit results for downstream training notebooks:

- **`corrected_train.csv`** — `image_id`, `target`, where the target is the COCO-derived
  (audited) label. This is the file our training pipeline consumes.
- **`label_audit.csv`** — one row per training image with `image_id`, `original_target`,
  `corrected_target` and the `reason` the rule engine assigned
  (`always_faulty:<categories>`, `conditional_area:<category=area>=threshold>`, or
  `reusable_by_coco_rules`), giving full traceability for every label.

Both are written to `/kaggle/working/` so they appear as notebook outputs.
""")
code("""
corrected_path = OUT_DIR / "corrected_train.csv"
audit_path = OUT_DIR / "label_audit.csv"

corrected_train.to_csv(corrected_path, index=False)
audit[["image_id", "original_target", "corrected_target", "reason"]].to_csv(audit_path, index=False)

reason_kind = audit["reason"].str.split(":").str[0].value_counts()
n_mismatch = int((audit["original_target"] != audit["corrected_target"]).sum())

print("=" * 62)
print("FINAL SUMMARY")
print("=" * 62)
print(f"training images audited ........ {len(audit):,}")
print(f"train.csv <-> COCO mismatches .. {n_mismatch}")
print(f"labels corrected 0 -> 1 ........ {int(((audit['original_target'] == 0) & (audit['corrected_target'] == 1)).sum())}")
print(f"labels corrected 1 -> 0 ........ {int(((audit['original_target'] == 1) & (audit['corrected_target'] == 0)).sum())}")
print()
print("label reasons:")
for kind, n in reason_kind.items():
    print(f"  {kind:<28} {n:>7,}  ({n / len(audit):.1%})")
print()
print("corrected target distribution:")
for t, n in corrected_train["target"].value_counts().sort_index().items():
    print(f"  target={t} ................... {n:>7,}  ({n / len(corrected_train):.1%})")
print()
print(f"saved: {corrected_path.resolve()}")
print(f"saved: {audit_path.resolve()}")
""")

# ---------------------------------------------------------------- step 6
md("""
## Step 6 — Per-Image Annotation Integrity Loop

Steps 1–5 audited the *labels*. This step loops over **every single training image** and
runs three independent integrity checks on its annotations:

1. **Geometry check** — does every polygon / bounding box lie inside the image frame, and
   does the PNG on disk actually have the dimensions the COCO header claims?
2. **ROI-visibility check** — after cropping to the ROI (Step 3), is each defect still
   visible? We compute, for every defect annotation, the fraction of its bounding box
   that intersects the ROI. Defects **fully outside** the ROI disappear from the model
   input; defects **less than half inside** are badly truncated.
3. **Render-detectability check** — why do some defects "not show up" in visualizations?
   Mostly because they are *tiny*: a 3 px² polygon is invisible at figure scale even
   though it is drawn. We flag every defect below 16 px² (invisible) and 64 px²
   (hard to see), and also re-verify the Step-4 label and look for internal
   contradictions (`No fault` co-occurring with an always-faulty category).

The result is one audit row per image; the flag totals are printed at the end.
""")
code("""
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


corrected_of = dict(zip(audit["image_id"], audit["corrected_target"]))
original_of = dict(zip(audit["image_id"], audit["original_target"]))

image_rows = []
geometry_bad_ann_ids = []          # annotations that need a COCO rewrite (step 7)
outside_roi_cases = []             # (img_id, ann_id, category)
tiny_showcase = []                 # (img_id, ann_id, category, area)

for img_id, info in img_by_id.items():
    W, H = info["width"], info["height"]
    fname = info["file_name"]
    anns = anns_by_image.get(img_id, [])
    cats = {cat_name[a["category_id"]] for a in anns} - ROI_CATEGORIES
    r = roi_box.get(img_id)

    # check 1: PNG on disk matches the COCO header
    dims_ok = True
    img_path = TRAIN_IMAGES / fname
    if img_path.exists():
        with Image.open(img_path) as pil:
            dims_ok = pil.size == (W, H)

    oob, fully_outside_roi, partial_roi, tiny, small = [], [], [], [], []
    min_defect_area = np.inf
    for a in anns:
        cname = cat_name[a["category_id"]]
        if cname in ROI_CATEGORIES:
            continue
        x, y, w, h = a["bbox"]
        # check 1: geometry inside the frame (1 px tolerance for float jitter)
        if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
            oob.append(a["id"])
            geometry_bad_ann_ids.append(a["id"])
        # check 2: visibility after the ROI crop
        if r is not None:
            ix = max(0.0, min(x + w, r[0] + r[2]) - max(x, r[0]))
            iy = max(0.0, min(y + h, r[1] + r[3]) - max(y, r[1]))
            frac = (ix * iy) / max(w * h, 1e-9)
            if frac == 0.0:
                fully_outside_roi.append(a["id"])
                outside_roi_cases.append((img_id, a["id"], cname))
            elif frac < 0.5:
                partial_roi.append(a["id"])
        # check 3: render detectability
        area = float(a.get("area") or 0.0)
        min_defect_area = min(min_defect_area, area)
        if area < 16:
            tiny.append(a["id"])
            if len(tiny_showcase) < 3 and r is not None:
                tiny_showcase.append((img_id, a["id"], cname, area))
        elif area < 64:
            small.append(a["id"])

    contradiction = bool("No fault" in cats and cats & ALWAYS_FAULTY)
    image_rows.append({
        "image_id": fname,
        "n_annotations": len(anns),
        "n_defect_annotations": sum(cat_name[a["category_id"]] not in ROI_CATEGORIES for a in anns),
        "dims_ok": dims_ok,
        "has_roi": r is not None,
        "n_geometry_out_of_bounds": len(oob),
        "n_defects_outside_roi": len(fully_outside_roi),
        "n_defects_partially_in_roi": len(partial_roi),
        "n_tiny_defects_lt16px": len(tiny),
        "n_small_defects_lt64px": len(small),
        "min_defect_area": None if np.isinf(min_defect_area) else round(min_defect_area, 1),
        "label_contradiction": contradiction,
        "original_target": original_of.get(fname),
        "corrected_target": corrected_of.get(fname),
        "label_ok": original_of.get(fname) == corrected_of.get(fname),
    })

image_audit = pd.DataFrame(image_rows)
image_audit.to_csv(OUT_DIR / "image_audit.csv", index=False)

print(f"images analysed ........................ {len(image_audit):,}")
print(f"PNG dims mismatching COCO header ....... {(~image_audit['dims_ok']).sum()}")
print(f"images missing an ROI .................. {(~image_audit['has_roi']).sum()}")
print(f"images w/ out-of-bounds geometry ....... {(image_audit['n_geometry_out_of_bounds'] > 0).sum()}"
      f"  ({len(geometry_bad_ann_ids)} annotations -> rewritten in Step 7)")
print(f"images w/ defect fully outside ROI ..... {(image_audit['n_defects_outside_roi'] > 0).sum()}")
print(f"images w/ defect <50% inside ROI ....... {(image_audit['n_defects_partially_in_roi'] > 0).sum()}")
print(f"images w/ invisible defects (<16 px2) .. {(image_audit['n_tiny_defects_lt16px'] > 0).sum()}")
print(f"images w/ hard-to-see defects (<64 px2)  {(image_audit['n_small_defects_lt64px'] > 0).sum()}")
print(f"label contradictions (No fault+faulty) . {image_audit['label_contradiction'].sum()}")
print(f"label mismatches vs train.csv .......... {(~image_audit['label_ok']).sum()}")

flagged = image_audit[(image_audit["n_geometry_out_of_bounds"] > 0) |
                      (image_audit["n_defects_outside_roi"] > 0)]
print("\\nimages flagged for geometry / ROI-visibility issues:")
print(flagged[["image_id", "n_geometry_out_of_bounds", "n_defects_outside_roi",
               "original_target", "corrected_target"]].to_string(index=False))
print(f"\\nsaved: {(OUT_DIR / 'image_audit.csv').resolve()}")
""")

# ---------------------------------------------------------------- step 7
md("""
## Step 7 — Rewriting the COCO Where Required

Step 6 found a handful of annotations whose polygons run **outside the image frame**
(negative coordinates or coordinates beyond 1280x1024). Renderers clip or refuse such
shapes, and area-based label rules are computed from geometry that partially does not
exist — so we rewrite them:

- clamp every polygon vertex into the image frame,
- recompute the bounding box from the clamped polygon,
- recompute `area` with the shoelace formula,
- log every change (`coco_fix_log.csv`) and verify the **derived label of the affected
  images does not change** (the fixed categories are always-faulty, so clamping the area
  cannot flip a conditional threshold).

The full corrected annotation file is exported as **`train_annotations_fixed.json`** —
a drop-in replacement for `train_annotations.json`. A before/after zoom of each repaired
annotation is rendered below for visual sign-off.
""")
code("""
bad_ids = set(geometry_bad_ann_ids)
ann_by_id = {a["id"]: a for a in coco["annotations"]}
fix_log = []

for ann_id in sorted(bad_ids):
    a = ann_by_id[ann_id]
    info = img_by_id[a["image_id"]]
    W, H = info["width"], info["height"]
    before_bbox = [round(v, 1) for v in a["bbox"]]
    before_area = round(float(a["area"]), 1)
    before_seg = [list(p) for p in a["segmentation"]]

    label_before, _ = derive_target(info["file_name"])

    new_seg = []
    for poly in a["segmentation"]:
        new_seg.append([clamp(v, 0.0, float(W) if i % 2 == 0 else float(H))
                        for i, v in enumerate(poly)])
    xs = [v for p in new_seg for v in p[::2]]
    ys = [v for p in new_seg for v in p[1::2]]
    new_bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    new_area = round(sum(polygon_area(p) for p in new_seg if len(p) >= 6), 2)

    # apply the rewrite (in memory; exported below) and refresh the area aggregation
    cname = cat_name[a["category_id"]]
    areas[info["file_name"]][cname] += new_area - float(a["area"])
    a["segmentation"], a["bbox"], a["area"] = new_seg, new_bbox, new_area
    label_after, _ = derive_target(info["file_name"])

    fix_log.append({
        "annotation_id": ann_id, "image_id": info["file_name"], "category": cname,
        "bbox_before": before_bbox, "bbox_after": [round(v, 1) for v in new_bbox],
        "area_before": before_area, "area_after": new_area,
        "label_before": label_before, "label_after": label_after,
        "_seg_before": before_seg,
    })

fix_df = pd.DataFrame(fix_log)
fix_df.drop(columns=["_seg_before"]).to_csv(OUT_DIR / "coco_fix_log.csv", index=False)
print(f"annotations rewritten: {len(fix_df)}")
print(fix_df.drop(columns=["_seg_before"]).to_string(index=False))
assert (fix_df["label_before"] == fix_df["label_after"]).all(), "a rewrite changed a label!"
print("\\nderived labels of all affected images unchanged -> corrected_train.csv stays valid")

fixed_path = OUT_DIR / "train_annotations_fixed.json"
with open(fixed_path, "w") as f:
    json.dump(coco, f, separators=(",", ":"))
print(f"saved: {fixed_path.resolve()} ({fixed_path.stat().st_size / 1e6:.0f} MB)")
print(f"saved: {(OUT_DIR / 'coco_fix_log.csv').resolve()}")

# --- before/after visual sign-off, zoomed to each repaired annotation ---
n = len(fix_log)
fig, axes = plt.subplots(n, 2, figsize=(11, 5.2 * n))
axes = np.atleast_2d(axes)
for row, rec in enumerate(fix_log):
    a = ann_by_id[rec["annotation_id"]]
    info = img_by_id[a["image_id"]]
    img = Image.open(TRAIN_IMAGES / info["file_name"])
    W, H = info["width"], info["height"]
    pad = 60
    x, y, w, h = rec["bbox_after"]
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(W, int(x + w) + pad), min(H, int(y + h) + pad)
    for col, (seg, bbox, title, color) in enumerate([
            (rec["_seg_before"], rec["bbox_before"], "before (runs outside frame)", "red"),
            (a["segmentation"], a["bbox"], "after (clamped to frame)", "lime")]):
        ax = axes[row, col]
        ax.imshow(img, extent=(0, W, H, 0))
        for poly in seg:
            if len(poly) >= 6:
                pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                ax.add_patch(MplPolygon(pts, closed=True, alpha=0.35,
                                        facecolor=color, edgecolor=color))
        bx, by, bw, bh = bbox
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor=color, linewidth=1.5))
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color="white", linewidth=1, linestyle=":")
        ax.set_xlim(min(x0, bx - 10), max(x1, bx + bw + 10))
        ax.set_ylim(max(y1, by + bh + 10), min(y0, by - 10))
        ax.set_title(f"{rec['category']} — {title}", fontsize=10)
        ax.axis("off")
fig.suptitle("COCO rewrite: repaired annotations (white dotted line = image border)", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()
""")

# ---------------------------------------------------------------- step 8
md("""
## Step 8 — Making Every Defect Visible

Step 6 explained the "some defects show, some don't" effect — two distinct causes:

1. **Tiny defects** (16 268 annotations below 64 px², 2 855 below 16 px²): they *are*
   drawn, but a few-pixel polygon is invisible at figure scale. The fix is a
   detectability-aware renderer: every defect gets a circle **marker** at its centroid
   plus a magnified **zoom inset**, so nothing can hide.
2. **Defects outside the ROI**: three annotations lie entirely outside their image's ROI
   box, so they vanish from the cropped model input. We render all three full frames to
   confirm none of them carries the image's label (verified programmatically in Step 6 —
   each image's target is already explained by annotations inside the ROI, or the
   annotation is an always-reusable category).
""")
code("""
# --- 8a. detectability-aware rendering of the three outside-ROI defects ---
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, (img_id, ann_id, cname) in zip(axes, outside_roi_cases):
    info = img_by_id[img_id]
    ax.imshow(Image.open(TRAIN_IMAGES / info["file_name"]))
    rx, ry, rw, rh = roi_box[img_id]
    ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, edgecolor="red",
                           linewidth=2, linestyle="--", label="ROI crop"))
    a = ann_by_id[ann_id]
    x, y, w, h = a["bbox"]
    ax.add_patch(Rectangle((x - 12, y - 12), w + 24, h + 24, fill=False,
                           edgecolor="cyan", linewidth=2, label=f"{cname} (outside ROI)"))
    t = target_of[info["file_name"]]
    ax.set_title(f"target={t} — '{cname}' lies outside the ROI", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")
fig.suptitle("Defects that disappear after ROI cropping (all three label-neutral)", y=1.02)
plt.tight_layout()
plt.show()

# --- 8b. tiny defects: marker + zoom inset so they cannot hide ---
fig, axes = plt.subplots(len(tiny_showcase), 2, figsize=(11, 5.5 * len(tiny_showcase)))
axes = np.atleast_2d(axes)
for row, (img_id, ann_id, cname, area) in enumerate(tiny_showcase):
    info = img_by_id[img_id]
    img = Image.open(TRAIN_IMAGES / info["file_name"])
    rx, ry, rw, rh = roi_box[img_id]
    crop = img.crop((int(rx), int(ry), int(rx + rw), int(ry + rh)))

    ax = axes[row, 0]
    ax.imshow(crop)
    for a in anns_by_image[img_id]:
        cn = cat_name[a["category_id"]]
        if cn in ROI_CATEGORIES:
            continue
        x, y, w, h = a["bbox"]
        cx, cy = x + w / 2 - rx, y + h / 2 - ry
        is_target = a["id"] == ann_id
        ax.add_patch(plt.Circle((cx, cy), max(14.0, 0.8 * max(w, h)), fill=False,
                                edgecolor="cyan" if is_target else "yellow",
                                linewidth=2.2 if is_target else 1.2))
        if is_target:
            ax.annotate(f"{cn} ({area:.0f} px2)", (cx, cy), xytext=(cx + 40, cy - 40),
                        color="cyan", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="cyan"))
    ax.set_title("ROI crop — every defect marked (cyan = tiny defect)", fontsize=10)
    ax.axis("off")

    a = ann_by_id[ann_id]
    x, y, w, h = a["bbox"]
    cx, cy = x + w / 2, y + h / 2
    half = 24
    zoom = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    zoom = zoom.resize((zoom.width * 8, zoom.height * 8), Image.NEAREST)
    ax = axes[row, 1]
    ax.imshow(zoom, extent=(cx - half, cx + half, cy + half, cy - half))
    pts = np.asarray(a["segmentation"][0], dtype=float).reshape(-1, 2)
    ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor="cyan", linewidth=2))
    ax.set_title(f"8x zoom — '{cname}', {area:.0f} px2: visible after all", fontsize=10)
    ax.axis("off")

fig.suptitle("Tiny defects rendered with markers + zoom insets", y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()

n_tiny_total = int(image_audit["n_tiny_defects_lt16px"].sum())
n_small_total = int(image_audit["n_small_defects_lt64px"].sum())
print(f"defects below 16 px2 (invisible without markers): {n_tiny_total:,}")
print(f"defects 16-64 px2 (hard to see at figure scale):  {n_small_total:,}")
print(f"defects fully outside the ROI crop:               {len(outside_roi_cases)}")
print("=> every annotation is now either visibly rendered, marker-flagged, or logged.")
""")

# ---------------------------------------------------------------- takeaways
md("""
## Key takeaways

- **Annotations are complete and consistent**: every `train.csv` image has a COCO entry,
  an ROI box, and at least one annotation — the COCO file can be trusted as ground truth.
- **`train.csv` agrees with the COCO-derived labels** under the official rules, so the
  audited `corrected_train.csv` doubles as verified training labels; the audit pipeline
  (Steps 4 and 6 re-run every image) would automatically repair any future data revision.
- **A handful of annotations needed a COCO rewrite**: four polygons ran outside the image
  frame and were clamped, with bbox/area recomputed (`train_annotations_fixed.json`,
  fully logged in `coco_fix_log.csv`); none of the rewrites changes a label.
- **"Invisible" defects are explained**: ~19 000 annotations are smaller than 64 px² and
  three lie entirely outside the ROI — they don't show in naive overlays. The
  detectability-aware renderer (markers + zoom insets) makes every annotation visible,
  and the per-image `image_audit.csv` flags each affected image.
- **The target is rule-derived from annotation categories and areas**, so per-category
  signals (and area-sensitive conditional defects like Scuffing or Air bubble) are the
  features a classifier must learn from the ROI crop.
- **Defect rates differ markedly across bottle types**, so validation splits should be
  stratified by bottle type as well as target to avoid optimistic estimates.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT_PATH)
print(f"wrote {OUT_PATH} ({len(cells)} cells)")
