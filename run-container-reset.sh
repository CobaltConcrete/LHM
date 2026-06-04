#!/usr/bin/env bash
# Launch LHM dev container (Podman + CUDA 12.1 local image)

set -euo pipefail

IMAGE="localhost/lhm-cu121"
NAME="lhm-dev"

mkdir -p "$HOME/qingrong/_shared/models" "$HOME/qingrong/_shared/hf-cache"

echo "[INFO] Using image: $IMAGE"

# Remove any prior container
podman rm -f "$NAME" >/dev/null 2>&1 || true

# Run container
exec podman run -it --rm \
    --name "$NAME" \
    --log-level=debug \
    --device nvidia.com/gpu=all \
    --security-opt label=disable \
    --security-opt label=disable \
    -v "$PWD":/workspaces/LHM \
    localhost/lhm-cu121 \
    bash