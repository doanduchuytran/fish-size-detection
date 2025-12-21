# fish-size-detection

This repository implements fish detection and pose (keypoint) estimation for aquaculture monitoring.
It is built on top of the **Ultrlytics** YOLO framework (cloned into this repo) and adds:
- Custom dataset configuration (FishKP-style: bounding box + 4 keypoints)
- Training / evaluation scripts
## Overview
## Dataset
## Model Architecture
## Installation
**Option A - Recommended: Use a Python virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r installation/requirements.txt
```
## Training
## Inference
```bash
python predict.py
```
## Results
## Docker Usage
## Citation
## License

