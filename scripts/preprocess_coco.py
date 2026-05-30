import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "outputs" / "preprocessing"

ROI_CATEGORIES = {"ROI", "Roi", "roi", "Region of interest"}

GOOD_CATEGORIES = {
    "Embossing",
    "Foam residue",
    "No fault",
    "Water drop",
}

CONDITIONAL_THRESHOLDS = {
    "Air bubble": 500,
    "Chip": 200,
    "Contamination light": 180,
    "Glass imperfection": 100,
    "Scuffing": 75000,
    "Scuffing heavy": 1200,
}

FAULTY_CATEGORIES = {
    "Break / Crack",
    "Circlip",
    "Contamination dark",
    "Crown cap",
    "Foil / Semitransparent",
    "Foreign object - manual cleaning",
    "Foreign object - washing machine",
    "Glass shard",
    "Insect",
    "Label",
    "Liquid",
    "Mold",
    "No base visible",
    "Paint residue",
    "Straw",
    "Yeast residue",
}

BOTTLE_TYPE_CANONICAL = {
    "VICHY": 0,
    "EURO": 1,
    "NRW": 2,
}


def stem(value: str) -> str:
    return Path(str(value)).stem


def canonical_bottle_type(value: str) -> int:
    text = str(value).lower()
    if "vichy" in text:
        return BOTTLE_TYPE_CANONICAL["VICHY"]
    if "euro" in text:
        return BOTTLE_TYPE_CANONICAL["EURO"]
    if "nrw" in text:
        return BOTTLE_TYPE_CANONICAL["NRW"]
    return -1


def polygon_area(coords) -> float:
    points = list(zip(coords[::2], coords[1::2]))
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def annotation_area(annotation: dict) -> float:
    area = float(annotation.get("area") or 0.0)
    if area > 0:
        return area

    segmentation = annotation.get("segmentation") or []
    if isinstance(segmentation, list):
        seg_area = sum(
            polygon_area(poly)
            for poly in segmentation
            if isinstance(poly, list) and len(poly) >= 6
        )
        if seg_area > 0:
            return seg_area

    bbox = annotation.get("bbox") or []
    if len(bbox) == 4:
        return float(bbox[2] * bbox[3])
    return 0.0


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def aggregate_coco(annotation_data: dict) -> dict:
    category_names = {item["id"]: item["name"] for item in annotation_data["categories"]}
    image_stems = {
        item["id"]: stem(item["file_name"])
        for item in annotation_data["images"]
    }

    areas = defaultdict(lambda: defaultdict(float))
    exists = defaultdict(set)
    roi = {}

    for annotation in annotation_data["annotations"]:
        image_stem = image_stems.get(annotation["image_id"])
        if image_stem is None:
            continue

        category = category_names.get(
            annotation["category_id"],
            f"unknown_{annotation['category_id']}",
        )
        exists[image_stem].add(category)
        areas[image_stem][category] += annotation_area(annotation)

        if category in ROI_CATEGORIES:
            bbox = annotation.get("bbox") or []
            if len(bbox) == 4:
                current = roi.get(image_stem)
                bbox_area = float(bbox[2] * bbox[3])
                current_area = float(current[2] * current[3]) if current else -1.0
                if bbox_area > current_area:
                    roi[image_stem] = [float(x) for x in bbox]

    return {
        "category_names": category_names,
        "image_stems": image_stems,
        "areas": areas,
        "exists": exists,
        "roi": roi,
    }


def derive_label(image_stem: str, exists: dict, areas: dict) -> tuple[int, str, str, str]:
    present = exists.get(image_stem, set())
    category_areas = areas.get(image_stem, {})

    always_faulty = sorted(category for category in present if category in FAULTY_CATEGORIES)
    if always_faulty:
        return 1, "always_faulty", always_faulty[0], ""

    conditional_hits = []
    for category, threshold in CONDITIONAL_THRESHOLDS.items():
        area = float(category_areas.get(category, 0.0))
        if area >= threshold:
            conditional_hits.append(f"{category}:{area:.1f}>={threshold}")

    if conditional_hits:
        first_category = conditional_hits[0].split(":", 1)[0]
        return 1, "conditional_area", first_category, "|".join(conditional_hits)

    return 0, "reusable_by_coco_rules", "", ""


def join_pipe(values) -> str:
    return "|".join(sorted(str(value) for value in values))


def area_json(category_areas: dict) -> str:
    compact = {
        category: round(float(area), 3)
        for category, area in sorted(category_areas.items())
        if category not in ROI_CATEGORIES
    }
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def roi_frame(roi: dict, image_ids=None) -> pd.DataFrame:
    rows = []
    stems = [stem(image_id) for image_id in image_ids] if image_ids is not None else sorted(roi)
    for image_stem in stems:
        bbox = roi.get(image_stem)
        if bbox:
            x, y, w, h = bbox
            rows.append({
                "stem": image_stem,
                "roi_x": x,
                "roi_y": y,
                "roi_w": w,
                "roi_h": h,
            })
        else:
            rows.append({
                "stem": image_stem,
                "roi_x": None,
                "roi_y": None,
                "roi_w": None,
                "roi_h": None,
            })
    return pd.DataFrame(rows)


def preprocess(data_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    bottletypes = pd.read_csv(data_dir / "bottletypes.csv")
    train_ann = aggregate_coco(load_json(data_dir / "train_annotations.json"))
    test_ann_path = data_dir / "test_annotations_roi_only.json"
    test_ann = aggregate_coco(load_json(test_ann_path)) if test_ann_path.exists() else None

    train["stem"] = train["image_id"].map(stem)
    bottletypes["stem"] = bottletypes["image_id"].map(stem)
    bottletypes["btype_id"] = bottletypes["bottle_type"].map(canonical_bottle_type)

    bottle_train = bottletypes[bottletypes["split"].eq("train")][
        ["stem", "bottle_type", "btype_id"]
    ]
    train_out = train.merge(bottle_train, on="stem", how="left")

    rows = []
    for row in train_out.itertuples(index=False):
        coco_target, reason, reason_category, conditional_hits = derive_label(
            row.stem,
            train_ann["exists"],
            train_ann["areas"],
        )
        categories = train_ann["exists"].get(row.stem, set())
        category_areas = train_ann["areas"].get(row.stem, {})
        roi_bbox = train_ann["roi"].get(row.stem)
        x, y, w, h = roi_bbox if roi_bbox else (None, None, None, None)
        original_target = int(row.target)
        rows.append({
            "image_id": row.image_id,
            "stem": row.stem,
            "target_original": original_target,
            "target_coco": coco_target,
            "target_corrected": coco_target,
            "label_mismatch": bool(original_target != coco_target),
            "label_reason": reason,
            "label_reason_category": reason_category,
            "conditional_hits": conditional_hits,
            "bottle_type": getattr(row, "bottle_type", None),
            "btype_id": getattr(row, "btype_id", -1),
            "roi_x": x,
            "roi_y": y,
            "roi_w": w,
            "roi_h": h,
            "categories_present": join_pipe(category for category in categories if category not in ROI_CATEGORIES),
            "good_categories": join_pipe(category for category in categories if category in GOOD_CATEGORIES),
            "always_faulty_categories": join_pipe(category for category in categories if category in FAULTY_CATEGORIES),
            "category_areas_json": area_json(category_areas),
        })

    audit = pd.DataFrame(rows)
    labels_corrected = audit[["image_id", "target_corrected"]].rename(
        columns={"target_corrected": "target"}
    )
    final_train = audit.copy()
    final_train.insert(2, "target", final_train["target_coco"].astype(int))
    final_train["target_source"] = "coco_annotations"
    final_train["target_note"] = "Derived from COCO categories and area thresholds; train.csv is kept only for audit."
    first_cols = [
        "image_id",
        "stem",
        "target",
        "target_source",
        "target_coco",
        "target_original",
        "label_mismatch",
        "label_reason",
        "label_reason_category",
        "conditional_hits",
        "bottle_type",
        "btype_id",
        "roi_x",
        "roi_y",
        "roi_w",
        "roi_h",
    ]
    final_train = final_train[
        first_cols + [column for column in final_train.columns if column not in first_cols]
    ]
    mismatches = audit[audit["label_mismatch"]].copy()
    train_roi = roi_frame(train_ann["roi"], train["image_id"])

    audit.to_csv(out_dir / "train_preprocessed.csv", index=False)
    audit.to_csv(out_dir / "label_coco_audit.csv", index=False)
    labels_corrected.to_csv(out_dir / "labels_corrected.csv", index=False)
    final_train.to_csv(out_dir / "final_train.csv", index=False)
    mismatches.to_csv(out_dir / "label_mismatches.csv", index=False)
    train_roi.to_csv(out_dir / "train_roi.csv", index=False)

    test_roi_rows = 0
    final_test_rows = 0
    if test_ann is not None:
        sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
        test_roi = roi_frame(test_ann["roi"], sample_submission["image_id"])
        test_roi.to_csv(out_dir / "test_roi.csv", index=False)
        test_bottle = bottletypes[bottletypes["split"].eq("test")][
            ["stem", "bottle_type", "btype_id"]
        ]
        final_test = sample_submission[["image_id"]].copy()
        final_test["stem"] = final_test["image_id"].map(stem)
        final_test = final_test.merge(test_bottle, on="stem", how="left")
        final_test = final_test.merge(test_roi, on="stem", how="left")
        final_test.to_csv(out_dir / "final_test.csv", index=False)
        test_roi_rows = len(test_roi)
        final_test_rows = len(final_test)

    known_categories = GOOD_CATEGORIES | set(CONDITIONAL_THRESHOLDS) | FAULTY_CATEGORIES | ROI_CATEGORIES
    unknown_categories = sorted(
        set(train_ann["category_names"].values()) - known_categories
    )

    summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "train_rows": int(len(train)),
        "original_target_counts": {
            str(key): int(value)
            for key, value in Counter(train["target"].astype(int)).items()
        },
        "coco_target_counts": {
            str(key): int(value)
            for key, value in Counter(audit["target_coco"].astype(int)).items()
        },
        "label_mismatches": int(len(mismatches)),
        "final_label_source": "target column in final_train.csv is derived from train_annotations.json COCO data",
        "final_train_path": str(out_dir / "final_train.csv"),
        "final_test_path": str(out_dir / "final_test.csv") if final_test_rows else "",
        "train_roi_rows": int(len(train_roi)),
        "train_roi_missing": int(train_roi["roi_x"].isna().sum()),
        "test_roi_rows": int(test_roi_rows),
        "final_test_rows": int(final_test_rows),
        "unknown_categories": unknown_categories,
        "category_counts": {
            category: int(count)
            for category, count in Counter(
                category
                for categories in train_ann["exists"].values()
                for category in categories
            ).most_common()
        },
        "conditional_thresholds": CONDITIONAL_THRESHOLDS,
    }

    with (out_dir / "preprocessing_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Krones COCO annotations.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summary = preprocess(args.data_dir, args.out_dir)
    print(f"Data: {summary['data_dir']}")
    print(f"Output: {summary['out_dir']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Original target counts: {summary['original_target_counts']}")
    print(f"COCO target counts: {summary['coco_target_counts']}")
    print(f"Label mismatches: {summary['label_mismatches']}")
    print(f"Train ROI missing: {summary['train_roi_missing']}")
    print(f"Unknown categories: {summary['unknown_categories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
