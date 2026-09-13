# 协作与提交规范

## 分支

`main` 是唯一长期分支。功能分支命名：

```text
feat/exp01-speech-enhancement
feat/exp02-speech-recognition
feat/exp03-speaker-recognition
feat/exp04-speech-synthesis
fix/expXX-short-description
docs/short-description
```

每次开始工作前：

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/expXX-name
```

已有功能分支同步主线：

```bash
git fetch origin
git rebase origin/main
```

## 目录所有权

每位负责人主要修改自己的实验目录。修改根目录、`.github/`、公共规范或其他人的实验前，应先在 Issue/群组中说明。不要复制其他实验的虚拟环境、模型或数据。

## 提交信息

采用简洁的 Conventional Commits 风格：

```text
feat(exp01): add noisy speech evaluation
fix(exp02): handle interrupted model download
docs(exp03): clarify enrollment protocol
test(exp04): cover text normalization
```

## Pull Request 清单

- 分支已基于最新 `main`。
- 只包含本次实验相关改动。
- README 给出目标、步骤、输入、输出、评价指标和一键启动命令。
- Docker Compose 配置能够解析。
- 自动化测试通过。
- 模型、数据、密钥和个人配置没有进入提交。
- 第三方数据与模型写明许可证和引用。

推荐使用 Squash merge。合并后删除功能分支。

