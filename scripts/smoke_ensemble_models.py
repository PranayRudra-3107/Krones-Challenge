import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import timm
import torch


MODELS = [
    ("teacher_convnext_small", "convnext_small.fb_in22k_ft_in1k"),
    ("teacher_maxvit", "maxvit_rmlp_tiny_rw_256.sw_in1k"),
    ("final_student_effnetv2_s", "tf_efficientnetv2_s.in21k_ft_in1k"),
]


def main() -> int:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    for tag, model_name in MODELS:
        print(f"\n{tag}: {model_name}")
        model = timm.create_model(model_name, pretrained=False, num_classes=0).to(device).eval()
        x = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"  features: {tuple(y.shape)}")
        del model, x, y
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    print("\nTeacher backbones and final student instantiate and run a forward pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
