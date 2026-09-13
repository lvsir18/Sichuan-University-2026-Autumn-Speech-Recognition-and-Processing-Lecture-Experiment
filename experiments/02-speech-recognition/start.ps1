$ErrorActionPreference = "Stop"
docker compose up --build -d
Write-Host "语音识别实验平台已启动：http://localhost:8000" -ForegroundColor Green
Write-Host "查看日志：docker compose logs -f asr-lab"

