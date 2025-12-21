#!/bin/bash

# Name of the container
CONTAINER_NAME="yolov8-cpu"

# Host project folder (change if needed)
PROJECT_DIR=/home/huytran/ntust/artificial_intelligence/fish-size-detection

# Image name (build your CPU image with this tag)
IMAGE_NAME="yolov8-cpu"

# Start container if it already exists
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Starting existing container: $CONTAINER_NAME"
    docker start -ai "$CONTAINER_NAME"
else
    echo "Creating and starting a new container: $CONTAINER_NAME"
    docker run -it \
        --shm-size=4g \
        -v "$PROJECT_DIR":/workspace \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
fi
