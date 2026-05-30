from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_data_dir() -> Path:
    candidates = [
        ROOT / "data",
        ROOT / "data" / "1st-krones-vision-ai-challenge",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "train_annotations.json").exists():
            return candidate
    return candidates[0]


DATA = find_data_dir()


def status(ok: bool, message: str) -> None:
    mark = "OK" if ok else "MISSING"
    print(f"[{mark}] {message}")


def main() -> int:
    print(f"Project root: {ROOT}")
    print(f"Dataset path: {DATA}")

    required = [
        DATA / "train.csv",
        DATA / "train_annotations.json",
        DATA / "bottletypes.csv",
        DATA / "train_images",
        DATA / "test_images",
    ]
    for path in required:
        status(path.exists(), str(path.relative_to(ROOT)))

    try:
        import torch
        import timm
        import albumentations
        import cv2
        import sklearn
        import cleanlab
        import onnx
        import onnxscript
        import onnxruntime
        import kaggle

        print(f"torch: {torch.__version__}")
        print(f"timm: {timm.__version__}")
        print(f"albumentations: {albumentations.__version__}")
        print(f"opencv: {cv2.__version__}")
        print(f"sklearn: {sklearn.__version__}")
        print(f"cleanlab: {cleanlab.__version__}")
        print(f"onnx: {onnx.__version__}")
        print(f"onnxscript: {onnxscript.__version__}")
        print(f"onnxruntime: {onnxruntime.__version__}")
        print("kaggle package: installed")
        print(f"cuda available: {torch.cuda.is_available()}")
        print(
            "mps available: "
            f"{torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}"
        )
    except Exception as exc:
        print(f"[ERROR] Python dependency check failed: {exc!r}")
        return 1

    missing = [path for path in required if not path.exists()]
    if missing:
        print("\nDataset is not complete yet. Finish the download/extract step, then rerun this check.")
        return 2

    token_path = Path.home() / ".kaggle" / "access_token"
    status(token_path.exists(), "~/.kaggle/access_token")

    print("\nSetup looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
