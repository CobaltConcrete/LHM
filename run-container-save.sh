#!/usr/bin/env bash
# Launch LHM dev container (Podman + CUDA 12.1 local image)

set -euo pipefail

IMAGE="localhost/lhm-cu121"
NAME="lhm-dev"

mkdir -p "$HOME/qingrong/_shared/models" "$HOME/qingrong/_shared/hf-cache"

echo "[INFO] Using image: $IMAGE"

# If container already exists, just reattach to it
if podman container exists "$NAME"; then
    echo "[INFO] Reattaching to existing container: $NAME"
    exec podman start -ai "$NAME"
fi

# First run: create a fresh container (no --rm so it persists on exit)
echo "[INFO] Creating new container from image: $IMAGE"
exec podman run -it \
    --name "$NAME" \
    --log-level=debug \
    --device nvidia.com/gpu=all \
    --security-opt label=disable \
    -v "$PWD":/workspaces/LHM \
    "$IMAGE" \
    bash