# 四川大学 2026 秋季《语音识别与处理》课程实验

本仓库集中维护四个相互独立、可通过 Docker 部署的课程实验。`main` 分支始终代表四个实验的完整可发布版本；各负责人在临时功能分支开发，并通过 Pull Request 合并。

## 实验目录

| 编号 | 实验 | 目录 | 状态 | 开发分支 |
| --- | --- | --- | --- | --- |
| 01 | 语音增强 | [`experiments/01-speech-enhancement`](experiments/01-speech-enhancement) | 待负责人实现 | `feat/exp01-speech-enhancement` |
| 02 | 语音识别 | [`experiments/02-speech-recognition`](experiments/02-speech-recognition) | 已实现 | `feat/exp02-speech-recognition` |
| 03 | 说话人识别 | [`experiments/03-speaker-recognition`](experiments/03-speaker-recognition) | 待负责人实现 | `feat/exp03-speaker-recognition` |
| 04 | 语音合成 | [`experiments/04-speech-synthesis`](experiments/04-speech-synthesis) | 待负责人实现 | `feat/exp04-speech-synthesis` |

## 设计原则

- 每个实验独立维护 `requirements.txt`、`Dockerfile`、`docker-compose.yml`、数据准备脚本和测试。
- 不在仓库中提交模型权重、完整语音数据、个人 `.env` 或运行结果。
- 每个实验必须提供无需教师手工配置的一键启动方式，并明确区分演示结果和真实测量结果。
- 公共规范放在仓库根目录；未经讨论，不要在多个实验之间建立代码级依赖。

选择独立依赖而不是根目录统一依赖，是因为语音增强、ASR、说话人识别和 TTS 的深度学习框架、音频库和模型版本可能不同。独立容器能减少依赖冲突，也便于四位负责人并行工作和独立验收。

## 快速开始：实验二

```powershell
cd experiments\02-speech-recognition
Copy-Item .env.example .env
.\start.ps1
```

打开 <http://localhost:8000>。其他实验完成后，以各自目录中的 README 为准。

## 协作流程

1. 从最新 `main` 创建自己的 `feat/expXX-name` 分支。
2. 原则上只修改自己负责的 `experiments/XX-name/`。
3. 在本地完成测试、Docker 配置检查和 README 操作验证。
4. 推送分支并创建目标为 `main` 的 Pull Request。
5. 至少一位其他成员审核，通过 CI 后使用 Squash merge。
6. 合并后删除功能分支；不要保留四个长期实验分支。

完整规则见 [CONTRIBUTING.md](CONTRIBUTING.md)，统一技术与验收要求见 [课程实验统一规范](docs/课程实验统一规范.md)。

## 许可与数据

本仓库自有代码使用 [MIT License](LICENSE)。第三方模型和数据集仍遵循各自许可证；每个实验必须在自己的 `data/` 或 README 中保留来源、版本、用途和引用信息。

