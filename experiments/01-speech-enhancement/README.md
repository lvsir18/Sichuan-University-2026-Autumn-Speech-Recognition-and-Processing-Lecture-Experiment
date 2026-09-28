# 实验一：语音增强

## 实验环境

### CMGAN（Windows，单张 NVIDIA GPU）

使用独立 Conda 环境 `cmgan-modern`，Python 3.10，PyTorch/TorchAudio 2.11.0（CUDA 13.0）。PyTorch wheel 已包含 CUDA 运行库，不需要另装 CUDA Toolkit。以下命令从仓库根目录依次执行；Conda 包和普通 Python 包使用清华镜像，PyTorch CUDA wheel 使用南京大学镜像：

```powershell
conda create -n cmgan-modern python=3.10 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate cmgan-modern
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 --timeout 120 --retries 10
conda install --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge pesq=0.0.4 "ffmpeg<9"
python -m pip install -r experiments/01-speech-enhancement/CMGAN/src/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 --timeout 120 --retries 10
```

- `conda create`：从清华 `pkgs/main` 镜像创建名为 `cmgan-modern` 的环境，并安装 Python 3.10 和 pip。
- `conda activate`：切换到新环境，后续安装都写入该环境。
- `pip install torch ... torchaudio ...`：安装匹配的 PyTorch 和 TorchAudio CUDA 13.0 版本；清华 PyPI 镜像提供普通依赖，南京大学镜像提供 PyTorch CUDA wheel。
- `conda install ... pesq ... ffmpeg`：从清华的 conda-forge 镜像安装 PESQ 指标依赖和 TorchCodec 所需的 FFmpeg 共享库。
- `pip install -r .../requirements.txt`：安装 CMGAN 的 Python 依赖，包括 TorchCodec。pip 源参数只对当前命令有效，因此这里也显式指定清华 PyPI 和南京大学 PyTorch 镜像。

安装完成并激活环境后，执行以下命令，检查 Python、PyTorch/TorchAudio 是否能导入，并实际运行一次 CUDA 张量运算：

```powershell
python -c "import sys, torch, torchaudio; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(16, 16, device='cuda'); y=x @ x; torch.cuda.synchronize(); print('CUDA tensor test:', tuple(y.shape))"
```

检查通过时，`CUDA available` 应为 `True`，输出会显示 GPU 名称和 `CUDA tensor test: (16, 16)`。若 CUDA 不可用，命令会在断言处明确失败。

### CMGAN（Linux，Conda，单张 NVIDIA GPU）

以下命令从仓库根目录依次执行。需要 Linux NVIDIA 驱动和可用的 GPU；PyTorch CUDA wheel 自带 CUDA 运行库，不需要另装 CUDA Toolkit。

```bash
conda create -n cmgan-modern python=3.10 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate cmgan-modern
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 --timeout 120 --retries 10
conda install --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge pesq=0.0.4 "ffmpeg<9"
python -m pip install -r experiments/01-speech-enhancement/CMGAN/src/requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 --timeout 120 --retries 10
python -m pip install --force-reinstall --no-deps torchcodec==0.11.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 10
```

最后一条命令仅从清华 PyPI 镜像重新安装 CPU 版 TorchCodec，避免前面的 CUDA wheel 镜像影响 TorchCodec 版本选择。CMGAN 只用 TorchCodec 读取 WAV，CPU 版解码即可，模型仍由 PyTorch 使用 GPU 训练。`--no-deps` 可避免这一步改动已安装的 PyTorch 和 TorchAudio。

验证 Python、CUDA 和 GPU：

```bash
python -c 'import sys, torch, torchaudio; print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__); print("TorchAudio:", torchaudio.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0)); x=torch.randn(16, 16, device="cuda"); y=x @ x; torch.cuda.synchronize(); print("CUDA tensor test:", tuple(y.shape))'
```

### 环境问题记录

- `No module named 'numpy'` 或 `No module named 'einops'`：尚未安装 `CMGAN/src/requirements.txt` 中的 Python 依赖。
- `No module named 'pesq'`：PESQ 单独由 conda-forge 安装，不在 pip 依赖文件里。
- `No module named 'torchcodec'`：TorchAudio 2.11 的音频读取需要 TorchCodec；依赖文件已固定 `torchcodec==0.11.1` 以匹配 PyTorch 2.11。
- `Could not load libtorchcodec` 且提示找不到 FFmpeg DLL/共享库：在已激活的 Conda 环境安装 `ffmpeg<9`；Windows 需要共享 DLL，Linux 需要 FFmpeg 共享库在环境库路径中可见。
- Linux 报 `libnppicc.so.13: cannot open shared object file`：当前 CUDA 版 TorchCodec 缺少 CUDA 13 的 NPP 运行库。CMGAN 读取 WAV 不需要 CUDA 解码；运行 Linux 环境步骤中的最后一条 pip 命令，从清华 PyPI 镜像安装 CPU 版 TorchCodec，PyTorch 仍会用 GPU 训练。

### Docker Compose（NVIDIA GPU）

Docker 环境固定 Python 3.10、PyTorch/TorchAudio 2.11.0（CUDA 13.0）、FFmpeg 和 PESQ。TorchCodec 固定使用 CPU wheel，因为它在本实验中只负责读取音频；PyTorch 仍使用 CUDA GPU 训练。需要 Docker Compose，以及宿主机 NVIDIA 驱动和 Docker 的 NVIDIA GPU 支持；Windows Docker Desktop 需启用 WSL2 GPU 支持。镜像不包含 VoiceBank-DEMAND 数据，运行时会挂载本地实验目录，checkpoint、日志和输出也会保存在本地。

在仓库根目录执行：

```bash
cd experiments/01-speech-enhancement
docker compose config
docker compose build
```

Dockerfile 默认通过清华 PyPI 镜像安装普通 Python 依赖，通过南京大学 PyTorch 镜像安装 CUDA 13.0 版 PyTorch。需要换源时可在构建时覆盖：

```bash
docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple --build-arg PYTORCH_INDEX_URL=https://mirrors.nju.edu.cn/pytorch/whl/cu130
```

基础镜像 `python:3.10-slim-bookworm` 由 Docker Engine 从镜像仓库拉取，不能通过 pip 源参数更换；如 Docker Hub 拉取较慢，可在 Docker Desktop 的 **Settings → Docker Engine** 中配置你可用的 registry mirror。系统 apt 包仍从 Debian 软件源下载。

`docker compose config` 检查 Compose 配置；构建完成后，运行容器默认命令，检查 PyTorch 是否识别 GPU 并执行 CUDA 张量运算：

```bash
docker compose run --rm cmgan
```

确认 GPU 检查通过后，可以在容器中运行实验脚本：

```bash
docker compose run --rm cmgan python script/run_cmgan_smoke_train.py
docker compose run --rm cmgan python script/run_cmgan_train.py --batch-size 4 --cut-len 32000 --epochs 120 --gpu 0
docker compose run --rm cmgan python script/evaluate_cmgan_test.py --checkpoint CMGAN/src/best_ckpt/ckpt --cut-len 64000 --overwrite
docker compose run --rm cmgan python script/plot_waveform_spectrogram.py
```

训练和评测命令需要本地 `VoiceBank-DEMAND` 数据目录已准备好。`docker compose run --rm cmgan` 每次都会新建并在结束后删除容器；代码、数据、checkpoint、TensorBoard 日志和 `output` 通过目录挂载保留在宿主机。

## 实验步骤

以下本机 Python 命令从 `experiments/01-speech-enhancement` 目录执行，并先激活 `cmgan-modern` 环境。若使用 Docker，先进入该目录，再执行各步骤附带的 Docker 命令；容器会把实验目录挂载到 `/workspace`。

### 1. 下载并放置 VoiceBank-DEMAND 数据集

从[爱丁堡大学 DataShare 数据集页面](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757)下载。训练集选择 **28 位说话人**版本，与测试集一起下载以下四个压缩包：`clean_trainset_28spk_wav.zip`、`noisy_trainset_28spk_wav.zip`、`clean_testset_wav.zip`、`noisy_testset_wav.zip`。将四个压缩包分别解压到 `VoiceBank-DEMAND/raw/` 下对应的同名目录：

```text
VoiceBank-DEMAND/raw/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

运行批量转换脚本。它会检查每组干净/带噪文件名是否配对，将 WAV 转为 16 kHz 单声道，并整理到训练和测试目录：

```powershell
python script/prepare_voicebank_16k.py
```

Docker：

```powershell
docker compose run --rm cmgan python script/prepare_voicebank_16k.py
```

脚本生成的目录结构为：

```text
experiments/01-speech-enhancement/VoiceBank-DEMAND/
├── train/
│   ├── clean/    # clean_trainset_28spk_wav.zip
│   └── noisy/    # noisy_trainset_28spk_wav.zip
└── test/
    ├── clean/    # clean_testset_wav.zip
    └── noisy/    # noisy_testset_wav.zip
```

数据集原始录音为 48 kHz；本实验的训练和评测脚本按 **16 kHz、单声道**处理。若使用已有的 16 kHz 数据，可跳过批量转换，直接按上面的目标结构放置。目标目录已有 WAV 时，脚本会停止以免覆盖；确认要重新转换时添加 `--overwrite`。数据目录已加入 `.gitignore`，不会随代码提交。DataShare 页面也列出了数据集许可条款，请按页面要求使用。

### 2. 快速检查训练流程

用一组内置语音跑一轮训练，确认环境、数据读取和模型更新正常：

```powershell
python script/run_cmgan_smoke_train.py
```

Docker：

```powershell
docker compose run --rm cmgan python script/run_cmgan_smoke_train.py
```

### 3. 训练 CMGAN

使用 VoiceBank-DEMAND 训练集训练模型。下面示例设置 batch size 为 4、每段语音 2 秒，最多训练 120 轮：

```powershell
python script/run_cmgan_train.py --batch-size 4 --cut-len 32000 --epochs 120 --gpu 0
```

Docker：

```powershell
docker compose run --rm cmgan python script/run_cmgan_train.py --batch-size 4 --cut-len 32000 --epochs 120 --gpu 0
```

checkpoint 默认保存在 `experiments/01-speech-enhancement/CMGAN/checkpoints`，TensorBoard 日志保存在 `CMGAN/runs`。其他参数可通过 `--help` 查看。

### 4. 增强测试集并计算指标

默认使用 `experiments/01-speech-enhancement/CMGAN/src/best_ckpt/ckpt` 这个 checkpoint，对 VoiceBank-DEMAND 测试集进行增强，并计算 SNR、SI-SDR、STOI 和 PESQ：

```powershell
python script/evaluate_cmgan_test.py --checkpoint CMGAN/src/best_ckpt/ckpt --cut-len 64000 --overwrite
```

Docker：

```powershell
docker compose run --rm cmgan python script/evaluate_cmgan_test.py --checkpoint CMGAN/src/best_ckpt/ckpt --cut-len 64000 --overwrite
```

`--cut-len 64000` 表示分段长度为 64000 个采样点（16 kHz 下约 4 秒），可降低显存占用。结果写入 `experiments/01-speech-enhancement/output`，包括增强音频、逐条指标 CSV 和汇总 CSV。输出目录非空时需指定 `--overwrite`。

要评测其他 checkpoint，将 `--checkpoint` 后面的路径改为目标 checkpoint 文件。例如：

```powershell
python script/evaluate_cmgan_test.py --checkpoint CMGAN/checkpoints/你的checkpoint文件名 --cut-len 64000 --overwrite
```

checkpoint 路径可以是相对仓库根目录的路径，也可以是绝对路径；确保指向实际存在的 checkpoint 文件。

Docker 中的路径相对于实验目录 `/workspace`，例如：

```powershell
docker compose run --rm cmgan python script/evaluate_cmgan_test.py --checkpoint CMGAN/checkpoints/你的checkpoint文件名 --cut-len 64000 --overwrite
```

### 5. 查看评测结果

评测完成后，打开 `experiments/01-speech-enhancement/reports/metrics_dashboard.html`。可以直接用浏览器打开，也可以从仓库根目录运行：

```powershell
Start-Process experiments/01-speech-enhancement/reports/metrics_dashboard.html
```

Dashboard 在宿主机浏览器打开，不需要在容器中运行。页面会显示内置的示例汇总结果。点击右上角“选择 metrics_summary.csv”，选择评测生成的 `experiments/01-speech-enhancement/output/metrics_summary.csv`，即可更新 SNR、SI-SDR、STOI、PESQ 图表和数据表。重新评测后，重新选择新的汇总 CSV 即可查看最新结果。

### 6. 绘制波形和语谱图

默认使用测试样本 `p232_160` 的干净、带噪和增强语音，生成 SVG 与 PNG：

```powershell
python script/plot_waveform_spectrogram.py
```

Docker：

```powershell
docker compose run --rm cmgan python script/plot_waveform_spectrogram.py
```

图像写入 `experiments/01-speech-enhancement/output/waveform_spectrogram_comparison.svg` 和同名 `.png`（PNG 生成需要 Pillow）。脚本可用 `--clean`、`--noisy`、`--enhanced` 指定其他音频；用 `--output` 指定 SVG 输出路径，PNG 会保存为同目录同名文件。
