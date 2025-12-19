from ultralytics import YOLO

MODEL_CONFIGS = [
    # ("yolov8n-pose.pt",  "yolo8n_pose"),
    # ("yolo11n-pose.pt",  "yolo11n_pose"),
    # ("yolo11s-pose.pt",  "yolo11s_pose"),
    # ("yolo12n.pt",  "yolo12n"),
    ("ultralytics/cfg/models/v8/yolov8-pose.yaml", "default_yolov8_pose"),
    ("ultralytics/cfg/models/v8/yolov8-pose-se.yaml", "modified_yolov8_pose"),
    # add yolo8s/11s/12s, etc. if you want
]

COMMON = dict(
    data="data/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    project="outputs",
    optimizer="SGD",
    lr0=0.01,
    weight_decay=0.01,
    patience=50,
    workers=2,
)

if __name__ == "__main__":
    for weights, name in MODEL_CONFIGS:
        print(f"\n=== Training {weights} ({name}) ===")
        model = YOLO(weights)
        model.train(name=name, **COMMON)
