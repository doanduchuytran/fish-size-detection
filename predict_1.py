# predict.py
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 pose prediction (bbox + 4 keypoints)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/exp1/weights/best.pt",
        help="Path to model .pt file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/val",
        help="Path to image, folder, video, or webcam index",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device to use, e.g. "0" for GPU0, "cpu" for CPU',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="outputs",
        help="Base directory to save predictions",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pred_exp1",
        help="Sub-folder name inside project for saving results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load model
    model = YOLO(args.model)

    # 2) Run prediction
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,           # save annotated images/video
        project=args.project,
        name=args.name,
        show=False           # set True if you want to display (requires GUI)
    )

    # 3) Print a brief summary
    print(f"Saved predictions to: {args.project}/{args.name}")
    for i, r in enumerate(results):
        print(f"Result {i}: {r.path}, {len(r.boxes)} boxes")


if __name__ == "__main__":
    main()
