# 声识实验台：语音识别课程实验

面向本科“语音识别”章节的 CPU 优先实验项目。项目围绕三条主线组织：完整 ASR 与评价指标、模型/配置控制变量对比、CTC 贪心/束搜索/n-gram 语言模型解码。提供中文 Web 实验台、FastAPI 接口、实验指导书、教师评分建议和 Docker 一键部署。


## 一键启动

前提：已安装 Docker Desktop（Windows/macOS）或 Docker Engine + Compose Plugin（Linux）。

```powershell
Copy-Item .env.example .env
.\start.ps1
```

Linux/macOS：

```bash
cp .env.example .env
sh start.sh
```

浏览器打开 <http://localhost:8000>。停止服务：

```bash
docker compose down
```

模型缓存在 Docker 命名卷中，`docker compose down` 不会删除；只有显式执行 `docker compose down -v` 才会删除模型卷。

## 固定课程数据集

课程统一使用 **FLEURS 普通话简体中文固定编号 0000～0011**，共 12 条 16 kHz 语音，内部版本标识为 `FLEURS-CMN-MINI-12-v1`，许可为 CC BY 4.0。项目不直接提交音频；启动后在页面顶部点击“下载并校验数据集”，系统会：

1. 从 FLEURS 文件镜像取得固定转写清单及 12 个独立 WAV，避免下载完整归档，也不依赖容易返回 500 的 Dataset Viewer `/rows` 接口；
2. 只下载这 12 条音频，不下载完整 FLEURS；
3. 用项目预置的 12 个 SHA-256 白名单逐一验收，再保存参考文本、编号和字节数；
4. 再次使用时先校验本地清单，校验通过则不重复下载。

也可在命令行准备或检查：

```powershell
python scripts\prepare_dataset.py
python scripts\prepare_dataset.py --check
```

容器内的数据保存在 `asr-dataset` 命名卷中。数据来源、引用和许可见 [data/DATASET_LICENSE.md](data/DATASET_LICENSE.md)。

## 两种运行模式

| 模式 | 网络/模型 | 结果标签 | 用途 |
| --- | --- | --- | --- |
| 内置演示 | 无需下载 | `simulated` | 低配置机房、流程教学、指标入门 |
| 固定 FLEURS 子集 | 12 条小数据 + 首次模型下载 | `measured` | 全班统一评价与模型对比 |
| 自行上传 | 首次运行下载所选模型 | `measured` | 扩展实验 |

演示结果是固定教学案例，不得用于宣称模型性能。真实模式默认 CPU + int8，推荐先用 tiny 或 base。模型下载需要能访问 Hugging Face；下载完成后可离线重复使用。

## 三个实验

1. **识别与评测**：从固定数据集选择单条语音 → 模型 → 文本 → CER/WER/SER/RTF，并展示替换/删除/插入。
2. **模型配置对比**：在全部 12 条固定语音上比较 tiny/base/small，计算微平均 CER、SER 和整体 RTF。
3. **解码与语言模型**：使用真实实现的 CTC prefix beam search，在固定帧概率上调整束宽与字符 bigram 权重。

详细讲义见 [实验一](docs/实验一-识别与评测.md)、[实验二](docs/实验二-模型配置对比.md)、[实验三](docs/实验三-解码与语言模型.md) 和 [教师指南](docs/教师指南.md)。

## 本地开发与测试

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
.\.venv\Scripts\python.exe -m pytest
```

API 文档位于 <http://localhost:8000/docs>，实验元数据位于 `GET /api/teaching/manifest`。

## 项目结构

```text
app/                 FastAPI、ASR 适配器、指标和 CTC 解码器
web/                 零构建依赖的中文单页实验台
data/                可机器读取的实验清单（不含受限数据集）
scripts/             固定数据集下载、SHA-256 校验和清单生成
docs/                三份学生实验指导书与教师指南
tests/               指标、解码、API 自动化测试
Dockerfile            CPU 容器镜像
docker-compose.yml    服务、模型缓存卷和输出目录
```

## 数据与合规

- 仓库不捆绑 THCHS-30、AISHELL、LibriSpeech 等数据集，使用前需自行确认并遵守其许可。
- 仅上传有权处理的音频；平台处理完成后会删除临时上传文件。
- 当前为单机教学工具，没有账号与多租户隔离，不建议直接暴露到公网。
- RTF 是整段离线推理耗时/音频时长，不等同于流式首字延迟。

## 技术说明

Faster-Whisper 模型在第一次真实请求时惰性加载，单进程内复用。上传默认限制 50 MB，可在 `.env` 调整。Docker 默认限制 6 GB 内存；运行 small 以上模型时可提高限制。生产教学部署建议由反向代理增加 HTTPS、身份认证和请求频控。

本项目自有代码按 MIT License 发布；Faster-Whisper、Whisper 模型以及所用数据集各自适用其上游许可。

## 常见故障

- `POST /api/dataset/prepare` 返回 502：先查看响应中的 `detail`。1.2 版已取消对 Dataset Viewer `/rows` 的依赖，改为独立 WAV 下载并按预置 SHA-256 校验。
- 固定数据集选项不可选：这是正常保护逻辑；只有页面显示“已就绪 · 12 条”后才开放，刷新页面可重新读取状态。
- 模型一直加载：检查 `docker compose exec asr-lab sh -c "du -sh /app/models"`。tiny 完整缓存远大于几 MB；出现 `.incomplete` 表示模型还没有下载完。
- Hugging Face 匿名下载被限速时，可在 `.env` 填写只读 `HF_TOKEN`，然后重新创建容器。
- 查看接口的详细错误：`docker compose logs --tail 100 asr-lab`，或在浏览器开发者工具的 Network 面板查看 JSON 响应。
