"""下载并校验固定的 FLEURS 中文迷你测试集，不依赖 datasets/pyarrow。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = ROOT / "data" / "fixed_dataset.json"
DEFAULT_OUTPUT = ROOT / "data" / "fleurs-mini"
USER_AGENT = "asr-course-lab/1.1 (+educational-dataset-preparation)"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_bytes(url: str, timeout: int = 90) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get_content_type()


def find_audio_url(value: Any) -> str | None:
    if isinstance(value, dict):
        source = value.get("src")
        if isinstance(source, str) and source.startswith("https://"):
            return source
        for nested in value.values():
            found = find_audio_url(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_audio_url(nested)
            if found:
                return found
    return None


def extension_for(url: str, content_type: str) -> str:
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        return suffix
    return {"audio/flac": ".flac", "audio/mpeg": ".mp3", "audio/ogg": ".ogg"}.get(content_type, ".wav")


def validate_existing(output: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    identity = {key: spec[key] for key in ("dataset_id", "config", "split", "offset", "length")}
    if manifest.get("source") != identity or len(manifest.get("samples", [])) != spec["length"]:
        return None
    for sample in manifest["samples"]:
        path = output / sample["audio"]
        if not path.is_file() or sha256(path) != sample["sha256"]:
            return None
    return manifest


def fixed_mirror_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    transcript_bytes, _ = request_bytes(spec["transcript_url"])
    transcripts: dict[str, str] = {}
    for line in transcript_bytes.decode("utf-8-sig").splitlines():
        identifier, separator, text = line.strip().partition(" ")
        if separator and text:
            transcripts[identifier] = text.strip()
    rows = []
    for row_index in range(spec["offset"], spec["offset"] + spec["length"]):
        identifier = f"cmn_hans_cn_{row_index:04d}"
        raw_reference = transcripts.get(identifier)
        if not raw_reference:
            raise RuntimeError(f"转写清单中缺少 {identifier}")
        rows.append({
            "row_idx": row_index,
            "row": {
                "audio": {"src": spec["audio_url_template"].format(index=row_index)},
                "transcription": "".join(raw_reference.split()),
                "raw_transcription": raw_reference,
            },
        })
    return rows


def prepare_dataset(output: Path = DEFAULT_OUTPUT, spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec = read_json(spec_path)
    existing = validate_existing(output, spec)
    if existing:
        return existing

    rows = fixed_mirror_rows(spec)
    if len(rows) != spec["length"]:
        raise RuntimeError(f"数据接口返回 {len(rows)} 条，预期 {spec['length']} 条")

    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="fleurs-mini-", dir=str(output)))
    samples: list[dict[str, Any]] = []
    try:
        for item in rows:
            row = item.get("row", {})
            row_index = int(item["row_idx"])
            audio_url = find_audio_url(row.get("audio"))
            if not audio_url:
                raise RuntimeError(f"第 {row_index} 行没有可下载的音频 URL")
            audio_bytes, content_type = request_bytes(audio_url)
            if len(audio_bytes) < 1024:
                raise RuntimeError(f"第 {row_index} 行音频异常小")
            extension = extension_for(audio_url, content_type)
            filename = f"fleurs-test-{row_index:04d}{extension}"
            staged_audio = staging / filename
            staged_audio.write_bytes(audio_bytes)
            actual_sha256 = hashlib.sha256(audio_bytes).hexdigest()
            expected_sha256 = spec["expected_sha256"][row_index - spec["offset"]]
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"第 {row_index} 条音频 SHA-256 不匹配，已拒绝使用")
            reference = row.get("transcription") or row.get("raw_transcription")
            if not isinstance(reference, str) or not reference.strip():
                raise RuntimeError(f"第 {row_index} 行缺少参考文本")
            samples.append({
                "id": f"fleurs-{row_index:04d}",
                "row_index": row_index,
                "audio": f"audio/{filename}",
                "reference": reference.strip(),
                "raw_reference": str(row.get("raw_transcription") or reference).strip(),
                "gender": row.get("gender"),
                "num_samples": row.get("num_samples"),
                "bytes": len(audio_bytes),
                "sha256": actual_sha256,
            })
        for sample in samples:
            os.replace(staging / Path(sample["audio"]).name, output / sample["audio"])
        source = {key: spec[key] for key in ("dataset_id", "config", "split", "offset", "length")}
        manifest = {
            "schema_version": 1,
            "name": spec["name"],
            "course_dataset_id": spec["course_dataset_id"],
            "source": source,
            "origin_dataset": spec["origin_dataset"],
            "download_strategy": "固定文件镜像 + 预置 SHA-256 白名单",
            "license": spec["license"],
            "homepage": spec["homepage"],
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "total_bytes": sum(item["bytes"] for item in samples),
            "samples": samples,
        }
        temporary_manifest = output / "manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_manifest, output / "manifest.json")
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def status(output: Path = DEFAULT_OUTPUT, spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec = read_json(spec_path)
    manifest = validate_existing(output, spec)
    return {"ready": manifest is not None, "spec": spec, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description="准备固定的 FLEURS 中文迷你测试集")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    args = parser.parse_args()
    try:
        result = status(args.output) if args.check else prepare_dataset(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if (not args.check or result["ready"]) else 1
    except Exception as exc:
        print(f"数据准备失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
