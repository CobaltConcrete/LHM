FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# ---------------- system dependencies ----------------
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    ffmpeg libgl1 libglib2.0-0 \
    ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

# make python default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# upgrade pip tooling
RUN pip install -U pip setuptools wheel

WORKDIR /workspaces/LHM

# ---------------- PyTorch stack (CUDA 12.1) ----------------
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install -U xformers==0.0.26.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------- core deps ----------------
RUN pip install --no-build-isolation chumpy

# BasicSR (avoid torchvision conflict)
RUN pip uninstall -y basicsr || true && \
    pip install git+https://github.com/XPixelGroup/BasicSR

# ---------------- repo requirements ----------------
COPY requirements.txt /workspaces/LHM/requirements.txt
RUN pip install -r /workspaces/LHM/requirements.txt

# ---------------- CUDA extensions ----------------
RUN pip install --no-build-isolation \
    git+https://github.com/facebookresearch/pytorch3d.git

RUN pip install git+https://github.com/hitsz-zuoqi/sam2/

RUN pip install --no-build-isolation \
    git+https://github.com/ashawkey/diff-gaussian-rasterization/

RUN pip install --no-build-isolation \
    git+https://github.com/camenduru/simple-knn/

# ---------------- sanity check ----------------
RUN nvcc --version && \
    python -c "import torch; print(torch.__version__, torch.version.cuda)"