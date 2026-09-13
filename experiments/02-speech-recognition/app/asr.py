"""可插拔 ASR 后端：教学演示适配器与 Faster-Whisper 真实推理。"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass
class RecognitionResult:
    text: str
    elapsed_seconds: float
    audio_seconds: float
    segments: list[dict]
    metadata: dict


DEMO_REFERENCES = {
    "campus": "人工智能正在改变我们的学习方式",
    "weather": "今天天气很好适合进行户外活动",
    "command": "请打开实验室的投影仪",
}

DEMO_OUTPUTS = {
    "tiny": {"campus": "人工智能正在改变我们学习方式", "weather": "今天天气很好适合户外活动", "command": "请打开实验的投影仪"},
    "base": {"campus": "人工智能正在改变我们的学习方式", "weather": "今天天气很好适合进行户外活动", "command": "请打开实验室的投影仪"},
    "small": {"campus": "人工智能正在改变我们的学习方式", "weather": "今天天气很好适合进行户外活动", "command": "请打开实验室的投影仪"},
}


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, EOFError, ZeroDivisionError):
        return 0.0


def recognize_demo(sample_id: str, model: str, beam_size: int) -> RecognitionResult:
    if sample_id not in DEMO_REFERENCES:
        raise ValueError("未知演示样本")
    started = perf_counter()
    text = DEMO_OUTPUTS.get(model, DEMO_OUTPUTS["base"])[sample_id]
    # 为课堂比较提供固定、明确标注为模拟值的延迟；不冒充实测性能。
    elapsed = {"tiny": 0.18, "base": 0.32, "small": 0.74}.get(model, 0.32) * (1 + max(beam_size - 1, 0) * 0.03)
    _ = perf_counter() - started
    return RecognitionResult(text, elapsed, 3.2, [{"start": 0.0, "end": 3.2, "text": text}], {"mode": "simulated", "notice": "教学演示结果，不代表真实模型性能"})


class WhisperRuntime:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], object] = {}

    def recognize(self, audio_path: Path, model_name: str, beam_size: int, language: str, device: str, compute_type: str) -> RecognitionResult:
        from faster_whisper import WhisperModel

        key = (model_name, device, compute_type)
        if key not in self._models:
            self._models[key] = WhisperModel(model_name, device=device, compute_type=compute_type, download_root="models")
        started = perf_counter()
        segments_iter, info = self._models[key].transcribe(str(audio_path), language=language or None, beam_size=beam_size, vad_filter=True)
        segments = [{"start": round(item.start, 3), "end": round(item.end, 3), "text": item.text.strip()} for item in segments_iter]
        elapsed = perf_counter() - started
        duration = wav_duration(audio_path) or max((item["end"] for item in segments), default=0.0)
        text = "".join(item["text"] for item in segments).strip()
        return RecognitionResult(text, elapsed, duration, segments, {
            "mode": "measured",
            "language": getattr(info, "language", language),
            "language_probability": round(getattr(info, "language_probability", 0.0), 4),
            "file_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        })


runtime = WhisperRuntime()
