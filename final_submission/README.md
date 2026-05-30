# Final Submission Package

This folder is the clean final-submission target.

Files:
- `final_evaluation.ipynb`: standalone ONNX inference notebook.
- `final_training.ipynb`: final student training controller.
- `model/model_metadata.json`: preprocessing and threshold metadata.

Before final sharing, attach `final_effnetv2_s.onnx` as a Kaggle dataset/input
next to `model_metadata.json`, then run `final_evaluation.ipynb` end to end and
confirm it writes `submission.csv`.

The final ONNX expects two inputs: image tensor and `btype_id`. The evaluation
notebook loads `bottletypes.csv` and passes that metadata automatically.
