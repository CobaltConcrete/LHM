#!/bin/bash
docker run -it \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/projects/LHM:/workspaces/LHM \
  -p 8080:8080 \
  cobaltconcrete/lhm \
  /bin/bash