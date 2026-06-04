#!/usr/bin/env bash
# Build lhm-cu121 container image

set -euo pipefail

IMAGE="localhost/lhm-cu121"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[INFO] Building image: $IMAGE"
echo "[INFO] Context: $SCRIPT_DIR"

podman build \
    --tag "$IMAGE" \
    --file "$SCRIPT_DIR/Dockerfile" \
    "$SCRIPT_DIR"

echo "[OK] Build complete: $IMAGE"
echo "[INFO] Run with: ./run-container.sh"