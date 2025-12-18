# ========= Base image (GPU, CUDA 11.8 suitable for Pascal) =========
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Install a Pascal-friendly PyTorch build (CUDA 11.8) ----
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    "opencv-python<4.10" \
    ultralytics

# ---- Install Ultralytics ----
RUN pip install --no-cache-dir ultralytics

# Default: open a shell, you run train.py manually
CMD ["bash"]
