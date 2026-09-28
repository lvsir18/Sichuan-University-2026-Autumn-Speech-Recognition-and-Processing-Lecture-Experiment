FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYTORCH_INDEX_URL=https://mirrors.nju.edu.cn/pytorch/whl/cu130

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY CMGAN/src/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url ${PYTORCH_INDEX_URL} \
        --extra-index-url ${PIP_INDEX_URL} \
        torch==2.11.0 torchaudio==2.11.0 \
    && python -m pip install \
        --index-url ${PIP_INDEX_URL} \
        --extra-index-url ${PYTORCH_INDEX_URL} \
        -r /tmp/requirements.txt \
    && python -m pip install \
        --force-reinstall --no-deps \
        --index-url ${PIP_INDEX_URL} \
        torchcodec==0.11.1 \
    && python -m pip install --index-url ${PIP_INDEX_URL} pesq==0.0.4

COPY . /workspace

CMD ["python", "-c", "import torch, torchaudio; print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(16,16,device='cuda'); torch.cuda.synchronize(); print('CUDA tensor test:', tuple(x.shape))"]
