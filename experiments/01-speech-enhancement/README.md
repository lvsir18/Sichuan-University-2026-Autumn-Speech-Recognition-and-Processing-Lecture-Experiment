# 实验一：语音增强

负责人在 `feat/exp01-speech-enhancement` 分支完成本目录。

## 实验目标

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

## 实验步骤

以下命令从仓库根目录运行；运行前先激活 `cmgan-modern` 环境。

### 1. 快速检查训练流程

用一组内置语音跑一轮训练，确认环境、数据读取和模型更新正常：

```powershell
python experiments/01-speech-enhancement/script/run_cmgan_smoke_train.py
```

### 2. 训练 CMGAN

使用 VoiceBank-DEMAND 训练集训练模型。下面示例设置 batch size 为 4、每段语音 2 秒，最多训练 120 轮：

```powershell
python experiments/01-speech-enhancement/script/run_cmgan_train.py --batch-size 4 --cut-len 32000 --epochs 120 --gpu 0
```

checkpoint 默认保存在 `experiments/01-speech-enhancement/CMGAN/checkpoints`，TensorBoard 日志保存在 `CMGAN/runs`。其他参数可通过 `--help` 查看。

### 3. 增强测试集并计算指标

默认使用 `experiments/01-speech-enhancement/CMGAN/src/best_ckpt/ckpt` 这个 checkpoint，对 VoiceBank-DEMAND 测试集进行增强，并计算 SNR、SI-SDR、STOI 和 PESQ：

```powershell
python experiments/01-speech-enhancement/script/evaluate_cmgan_test.py --checkpoint experiments/01-speech-enhancement/CMGAN/src/best_ckpt/ckpt --cut-len 64000 --overwrite
```

`--cut-len 64000` 表示分段长度为 64000 个采样点（16 kHz 下约 4 秒），可降低显存占用。结果写入 `experiments/01-speech-enhancement/output`，包括增强音频、逐条指标 CSV 和汇总 CSV。输出目录非空时需指定 `--overwrite`。

要评测其他 checkpoint，将 `--checkpoint` 后面的路径改为目标 checkpoint 文件。例如：

```powershell
python experiments/01-speech-enhancement/script/evaluate_cmgan_test.py --checkpoint experiments/01-speech-enhancement/CMGAN/checkpoints/你的checkpoint文件名 --cut-len 64000 --overwrite
```

checkpoint 路径可以是相对仓库根目录的路径，也可以是绝对路径；确保指向实际存在的 checkpoint 文件。

### 4. 查看评测结果

评测完成后，打开 `experiments/01-speech-enhancement/reports/metrics_dashboard.html`。可以直接用浏览器打开，也可以从仓库根目录运行：

```powershell
Start-Process experiments/01-speech-enhancement/reports/metrics_dashboard.html
```

页面会显示内置的示例汇总结果。点击右上角“选择 metrics_summary.csv”，选择评测生成的 `experiments/01-speech-enhancement/output/metrics_summary.csv`，即可更新 SNR、SI-SDR、STOI、PESQ 图表和数据表。重新评测后，重新选择新的汇总 CSV 即可查看最新结果。

### 5. 绘制波形和语谱图

默认使用测试样本 `p232_160` 的干净、带噪和增强语音，生成 SVG 与 PNG：

```powershell
python experiments/01-speech-enhancement/script/plot_waveform_spectrogram.py
```

图像写入 `experiments/01-speech-enhancement/output/waveform_spectrogram_comparison.svg` 和同名 `.png`（PNG 生成需要 Pillow）。脚本可用 `--clean`、`--noisy`、`--enhanced` 指定其他音频；用 `--output` 指定 SVG 输出路径，PNG 会保存为同目录同名文件。

## 实验报告与提交要求

- 数据与噪声来源、许可证、固定样本清单。
- 独立 Dockerfile、Compose 和依赖文件。
- 中文实验界面或可操作 Notebook。
- 学生实验指导书、结果导出和基础测试。
