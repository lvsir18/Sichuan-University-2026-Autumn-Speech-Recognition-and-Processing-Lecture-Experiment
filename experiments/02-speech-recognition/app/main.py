from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.asr import DEMO_REFERENCES, recognize_demo, runtime
from app.decoder import run_decoder_demo
from app.metrics import calculate_metrics
from scripts.prepare_dataset import DEFAULT_OUTPUT as DATASET_OUTPUT
from scripts.prepare_dataset import prepare_dataset, status as dataset_status


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

app = FastAPI(title="语音识别课程实验平台", version="1.2.0")
dataset_prepare_lock = threading.Lock()


class DemoRequest(BaseModel):
    sample_id: str = "campus"
    model: str = Field(default="tiny", pattern="^(tiny|base|small)$")
    beam_size: int = Field(default=1, ge=1, le=20)
    reference: str | None = None


class DecodeRequest(BaseModel):
    beam_width: int = Field(default=5, ge=1, le=20)
    lm_weight: float = Field(default=0.0, ge=0.0, le=2.0)


class DatasetRecognitionRequest(BaseModel):
    sample_id: str
    model: str = Field(default="tiny", pattern="^(tiny|base|small|medium|large-v3)$")
    beam_size: int = Field(default=5, ge=1, le=20)
    language: str = "zh"


class DatasetEvaluationRequest(BaseModel):
    model: str = Field(default="tiny", pattern="^(tiny|base|small|medium|large-v3)$")
    beam_size: int = Field(default=5, ge=1, le=20)
    language: str = "zh"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "asr-course-lab", "version": app.version}


@app.get("/api/catalog")
def catalog() -> dict:
    return {
        "samples": [{"id": key, "reference": value} for key, value in DEMO_REFERENCES.items()],
        "models": [
            {"id": "tiny", "description": "速度快，适合流程入门"},
            {"id": "base", "description": "准确率与速度较均衡"},
            {"id": "small", "description": "精度更高，CPU 推理较慢"},
        ],
        "engines": ["demo", "faster-whisper"],
    }


def _dataset_manifest() -> dict:
    current = dataset_status(DATASET_OUTPUT)
    if not current["ready"]:
        raise HTTPException(status_code=409, detail="固定数据集尚未准备，请先在数据集面板下载")
    return current["manifest"]


def _dataset_sample(sample_id: str) -> tuple[dict, Path]:
    manifest = _dataset_manifest()
    sample = next((item for item in manifest["samples"] if item["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail="数据集样本不存在")
    path = (DATASET_OUTPUT / sample["audio"]).resolve()
    if DATASET_OUTPUT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="数据集音频文件缺失")
    return sample, path


@app.get("/api/dataset/status")
def fixed_dataset_status() -> dict:
    current = dataset_status(DATASET_OUTPUT)
    manifest = current.get("manifest")
    return {
        "ready": current["ready"],
        "spec": current["spec"],
        "sample_count": len(manifest["samples"]) if manifest else 0,
        "total_bytes": manifest.get("total_bytes", 0) if manifest else 0,
        "samples": manifest.get("samples", []) if manifest else [],
    }


@app.post("/api/dataset/prepare")
def prepare_fixed_dataset() -> dict:
    if not dataset_prepare_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="数据集正在准备中，请稍候刷新")
    try:
        manifest = prepare_dataset(DATASET_OUTPUT)
        return {"ready": True, "sample_count": len(manifest["samples"]), "total_bytes": manifest["total_bytes"]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"数据集下载失败：{exc}") from exc
    finally:
        dataset_prepare_lock.release()


@app.get("/api/dataset/audio/{sample_id}")
def fixed_dataset_audio(sample_id: str) -> FileResponse:
    _, path = _dataset_sample(sample_id)
    return FileResponse(path)


def _response(result, reference: str, model: str, beam_size: int) -> dict:
    return {
        "reference": reference,
        "hypothesis": result.text,
        "model": model,
        "beam_size": beam_size,
        "elapsed_seconds": round(result.elapsed_seconds, 4),
        "audio_seconds": round(result.audio_seconds, 4),
        "segments": result.segments,
        "metrics": calculate_metrics(reference, result.text, result.elapsed_seconds, result.audio_seconds),
        "metadata": result.metadata,
    }


@app.post("/api/recognize/demo")
def recognize_demo_endpoint(request: DemoRequest) -> dict:
    try:
        result = recognize_demo(request.sample_id, request.model, request.beam_size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reference = request.reference if request.reference is not None else DEMO_REFERENCES[request.sample_id]
    return _response(result, reference, request.model, request.beam_size)


@app.post("/api/recognize/upload")
def recognize_upload(
    audio: UploadFile = File(...),
    reference: str = Form(...),
    model: str = Form("tiny"),
    beam_size: int = Form(5),
    language: str = Form("zh"),
    device: str = Form("cpu"),
    compute_type: str = Form("int8"),
) -> dict:
    if model not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(status_code=422, detail="不支持的模型")
    if not 1 <= beam_size <= 20:
        raise HTTPException(status_code=422, detail="beam_size 必须在 1 到 20 之间")
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}:
        raise HTTPException(status_code=415, detail="请上传 WAV/MP3/M4A/FLAC/OGG/WebM 音频")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        copied = 0
        while chunk := audio.file.read(1024 * 1024):
            copied += len(chunk)
            if copied > MAX_UPLOAD_BYTES:
                Path(temporary.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="音频文件超过大小限制")
            temporary.write(chunk)
        path = Path(temporary.name)
    try:
        result = runtime.recognize(path, model, beam_size, language, device, compute_type)
        return _response(result, reference, model, beam_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/recognize/dataset")
def recognize_dataset(request: DatasetRecognitionRequest) -> dict:
    sample, path = _dataset_sample(request.sample_id)
    try:
        result = runtime.recognize(path, request.model, request.beam_size, request.language, "cpu", "int8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc
    response = _response(result, sample["reference"], request.model, request.beam_size)
    response["sample_id"] = sample["id"]
    response["dataset"] = "FLEURS-CMN-MINI-12-v1"
    return response


@app.post("/api/evaluate/dataset")
def evaluate_dataset(request: DatasetEvaluationRequest) -> dict:
    manifest = _dataset_manifest()
    results = []
    for sample in manifest["samples"]:
        path = (DATASET_OUTPUT / sample["audio"]).resolve()
        try:
            recognition = runtime.recognize(path, request.model, request.beam_size, request.language, "cpu", "int8")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"样本 {sample['id']} 推理失败：{exc}") from exc
        item = _response(recognition, sample["reference"], request.model, request.beam_size)
        item["sample_id"] = sample["id"]
        results.append(item)
    total_errors = sum(item["metrics"]["char_edits"][key] for item in results for key in ("substitutions", "deletions", "insertions"))
    total_characters = sum(item["metrics"]["char_edits"]["reference_length"] for item in results)
    total_elapsed = sum(item["elapsed_seconds"] for item in results)
    total_audio = sum(item["audio_seconds"] for item in results)
    return {
        "dataset": "FLEURS-CMN-MINI-12-v1",
        "model": request.model,
        "beam_size": request.beam_size,
        "sample_count": len(results),
        "aggregate": {
            "cer_micro": round(total_errors / total_characters, 4) if total_characters else None,
            "ser": round(sum(item["metrics"]["ser"] for item in results) / len(results), 4),
            "rtf": round(total_elapsed / total_audio, 4) if total_audio else None,
            "elapsed_seconds": round(total_elapsed, 4),
            "audio_seconds": round(total_audio, 4),
            "total_characters": total_characters,
            "total_char_errors": total_errors,
        },
        "results": results,
    }


@app.post("/api/decode")
def decode(request: DecodeRequest) -> dict:
    result = run_decoder_demo(request.beam_width, request.lm_weight)
    result["metrics"] = {
        "greedy": calculate_metrics(result["reference"], result["greedy"], 0, 1),
        "beam": calculate_metrics(result["reference"], result["beam"], 0, 1),
    }
    return result


@app.get("/api/teaching/manifest")
def teaching_manifest() -> dict:
    return json.loads((ROOT / "data" / "experiment_manifest.json").read_text(encoding="utf-8"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
