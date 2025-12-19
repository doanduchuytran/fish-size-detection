# validate.py
from ultralytics import YOLO


def main():
    # 1) Path to your trained weights
    # Change this if your exp folder or name is different
    model_path = "outputs/modified_yolov8_pose2/weights/best.pt"

    # 2) Load the model
    model = YOLO(model_path)

    # 3) Run validation
    # - data: your dataset yaml
    # - split: which subset in data.yaml ("val" by default)
    # - save: save images with predictions
    # - plots: save confusion matrix, PR curve, etc.
    results = model.val(
        data="data/data.yaml",   # dataset config
        split="test",             # evaluate on validation set
        batch=16,                # adjust if needed
        device=0,                # 0 = first GPU, or "cpu"
        project="outputs",       # where to store validation results
        name="modified_yolov8_pose2_val",         # folder name inside project
        save=True,
        plots=True
    )

    # 4) Print a short summary
    print(results)  # prints metrics (mAP, keypoint metrics, etc.)


if __name__ == "__main__":
    main()