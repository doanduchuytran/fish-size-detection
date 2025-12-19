import os
import cv2
from ultralytics import YOLO

# =========================
# USER CONFIG (EDIT IF NEEDED)
# =========================
MODEL_PATH = "outputs/modified_yolov8_pose2/weights/best.pt"   # your trained model
VIDEO_PATH = "videos/inputs/f2.mp4"                         # input video
OUTPUT_PATH = "videos/inputs/f2.mp4"             # output video

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.7
DEVICE = "cpu"                                   # force CPU
# =========================


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0

    print("Running inference on CPU...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=DEVICE,
            verbose=False
        )

        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    writer.release()

    print("===================================")
    print("Inference finished successfully!")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("===================================")


if __name__ == "__main__":
    main()
