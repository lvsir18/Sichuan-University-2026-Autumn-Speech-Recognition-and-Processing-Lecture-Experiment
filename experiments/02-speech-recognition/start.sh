#!/usr/bin/env sh
set -eu
docker compose up --build -d
printf '%s\n' '语音识别实验平台已启动：http://localhost:8000'
printf '%s\n' '查看日志：docker compose logs -f asr-lab'

