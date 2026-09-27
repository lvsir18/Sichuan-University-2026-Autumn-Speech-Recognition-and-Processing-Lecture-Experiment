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
conda install --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge pesq=0.0.4
cd experiments/01-speech-enhancement/CMGAN/src
python -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
```

- `conda create`：从清华 `pkgs/main` 镜像创建名为 `cmgan-modern` 的环境，并安装 Python 3.10 和 pip。
- `conda activate`：切换到新环境，后续安装都写入该环境。
- `pip install torch ... torchaudio ...`：安装匹配的 PyTorch 和 TorchAudio CUDA 13.0 版本；清华 PyPI 镜像提供普通依赖，南京大学镜像提供 PyTorch CUDA wheel。
- `conda install ... pesq=0.0.4`：从清华的 conda-forge 镜像安装 PESQ 语音质量指标依赖，项目训练和评测会用到。
- `cd .../src`：进入 CMGAN 源码目录，后续运行命令从这里执行。
- `pip install -r requirements.txt`：安装 CMGAN 其余 Python 依赖。pip 命令的源参数只对当前命令有效，因此这里也显式指定清华 PyPI 和南京大学 PyTorch 镜像。

安装完成后，在 `src` 目录执行以下命令，检查 Python、PyTorch/TorchAudio 是否能导入，并实际运行一次 CUDA 张量运算：

```powershell
python -c "import sys, torch, torchaudio; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); x=torch.randn(16, 16, device='cuda'); y=x @ x; torch.cuda.synchronize(); print('CUDA tensor test:', tuple(y.shape))"
```

检查通过时，`CUDA available` 应为 `True`，输出会显示 GPU 名称和 `CUDA tensor test: (16, 16)`。若 CUDA 不可用，命令会在断言处明确失败。

## 实验步骤

1. 构造或选择干净语音与噪声，按固定 SNR 混合。
2. 展示时域波形、频谱/语谱图和增强前后音频。
3. 比较传统谱减法与一个轻量神经网络增强模型。
4. 计算 SNR、SI-SDR、STOI；PESQ 仅在许可证和环境允许时启用。

## 结果与分析

## 实验报告与提交要求

- 数据与噪声来源、许可证、固定样本清单。
- 独立 Dockerfile、Compose 和依赖文件。
- 中文实验界面或可操作 Notebook。
- 学生实验指导书、结果导出和基础测试。
