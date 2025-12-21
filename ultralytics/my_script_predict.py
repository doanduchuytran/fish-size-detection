from ultralytics import YOLO

# Load trained weights (update path if different)
model = YOLO("my_outputs/train_outputs/yolov8_pose_proposed/weights/best.pt")

# Run validation on the dataset defined in fish_keypoint.yaml
results = model.val(
    data="datasets/data.yaml",
    name="yolov8_pose_proposed_predict_test",
    project="my_outputs/predict_outputs",
    split="test",
    imgsz=640,
    batch=64,
    device='cpu',
    save_json=True,   # saves COCO-style metrics file
    verbose=True,
    save_conf=True  # save per-keypoint confidence scores in the JSON file
)

# Save metrics to JSON
import json
from pathlib import Path

save_dir = Path(results.save_dir)

# Extract key metrics as percentages
summary_pct = {
    "object_mAP50": float(results.results_dict.get("metrics/mAP50(B)", float("nan"))) * 100,
    "object_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", float("nan"))) * 100,
    "pose_mAP50": float(results.results_dict.get("metrics/mAP50(P)", float("nan"))) * 100,
    "pose_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(P)", float("nan"))) * 100,
}

# Save alongside other metrics
(save_dir / "metrics_summary.json").write_text(json.dumps(summary_pct, indent=2))

print("mAP summary (%):", summary_pct)