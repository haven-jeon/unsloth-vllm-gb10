#!/bin/bash

# Default image name
IMAGE_NAME="gogamza/unsloth-vllm-gb10:260126"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "Build complete."
echo "To push the image, run: docker push $IMAGE_NAME"
