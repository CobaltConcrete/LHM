#!/bin/bash
set -e

echo "Downloading LHM prior model..."
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM_prior_model.tar
tar -xvf LHM_prior_model.tar

echo "Downloading motion video..."
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/motion_video.tar
tar -xvf ./motion_video.tar 

echo "Installing dependencies..."
pip install -U modelscope hf_transfer huggingface_hub

echo "Enabling Hugging Face fast transfer..."
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Downloading ModelScope models + HF snapshots..."
python3 <<'PYTHON'
import os

# Ensure HF fast transfer is enabled inside Python subprocess
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from modelscope import snapshot_download

# MINI model
snapshot_download(
    model_id='Damo_XR_Lab/LHM-MINI',
    cache_dir='./pretrained_models'
)

# 500M-HF model
snapshot_download(
    model_id='Damo_XR_Lab/LHM-500M-HF',
    cache_dir='./pretrained_models'
)

# 1B-HF model
snapshot_download(
    model_id='Damo_XR_Lab/LHM-1B-HF',
    cache_dir='./pretrained_models'
)

print("All models downloaded successfully.")
PYTHON

echo "Done."