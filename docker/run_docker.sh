#!/bin/bash

# Name of the container
CONTAINER_NAME="yolo-train-gpu"

# Start container if it already exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Starting existing container: $CONTAINER_NAME"
    docker start -ai $CONTAINER_NAME
else
    echo "Creating and starting a new container: $CONTAINER_NAME"
    docker run -d --gpus all \
        --shm-size=4g \
        -v ~/ai_course_final_project/my_fish_size_project:/workspace \
        --name $CONTAINER_NAME \
        yolo-ultralytics-gpu
fi
