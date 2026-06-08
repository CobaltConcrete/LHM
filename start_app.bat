@echo off

docker run -it ^
  --gpus all ^
  -v %USERPROFILE%\Projects\LHM:/workspaces/LHM ^
  -p 8080:8080 ^
  cobaltconcrete/lhm ^
  /bin/bash