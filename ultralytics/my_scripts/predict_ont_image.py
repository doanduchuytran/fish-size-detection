from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Paths
# -----------------------------
model_path = "outputs\yolov8_pose_proposed\weights\best.pt"          # path to your trained model
image_path = "data\images\test\2155.png"         # input image
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# Run inference
# -----------------------------
results = model(image_path, conf=0.25)

# -----------------------------
# Get annotated image
# -----------------------------
annotated_img = results[0].plot()  # bounding box + keypoints drawn

# -----------------------------
# Save result
# -----------------------------
output_path = os.path.join(output_dir, "result.jpg")
cv2.imwrite(output_path, annotated_img)

print(f"Prediction saved to: {output_path}")
