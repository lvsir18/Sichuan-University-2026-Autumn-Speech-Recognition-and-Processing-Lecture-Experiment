"""Batch-convert extracted VoiceBank-DEMAND WAV files to 16 kHz mono."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


DATASETS = (
    ("clean_trainset_28spk_wav", "train/clean"),
    ("noisy_trainset_28spk_wav", "train/noisy"),
    ("clean_testset_wav", "test/clean"),
    ("noisy_testset_wav", "test/noisy"),
)


def wav_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"找不到解压目录：{directory}")

    files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.name.lower(),
    )
    by_name = {path.name: path for path in files}
    if len(by_name) != len(files):
        raise ValueError(f"目录中有重名 WAV 文件，无法平铺输出：{directory}")
    if not files:
        raise ValueError(f"目录中没有找到 WAV 文件：{directory}")
    return by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 VoiceBank-DEMAND 四个解压目录批量转换为 16 kHz 单声道 WAV。"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("VoiceBank-DEMAND/raw"),
        help="四个数据集解压目录所在位置（默认：VoiceBank-DEMAND/raw）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("VoiceBank-DEMAND"),
        help="整理后数据集的根目录（默认：VoiceBank-DEMAND）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖目标目录中同名 WAV 文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("未找到 ffmpeg。请先安装 FFmpeg 并确保 ffmpeg 命令可用。")

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    sources: dict[str, dict[str, Path]] = {}
    for source_name, _ in DATASETS:
        sources[source_name] = wav_files(raw_dir / source_name)

    for clean_source, noisy_source, split in (
        ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav", "train"),
        ("clean_testset_wav", "noisy_testset_wav", "test"),
    ):
        clean_names = set(sources[clean_source])
        noisy_names = set(sources[noisy_source])
        if clean_names != noisy_names:
            clean_only = sorted(clean_names - noisy_names)[:5]
            noisy_only = sorted(noisy_names - clean_names)[:5]
            raise ValueError(
                f"{split} 干净/带噪文件名不匹配。"
                f"仅 clean 存在：{clean_only}；仅 noisy 存在：{noisy_only}"
            )

    if not args.overwrite:
        existing = []
        for _, relative_output in DATASETS:
            destination = output_dir / relative_output
            if destination.is_dir():
                existing.extend(
                    path
                    for path in destination.iterdir()
                    if path.is_file() and path.suffix.lower() == ".wav"
                )
        if existing:
            raise FileExistsError(
                f"目标目录已有 {len(existing)} 个 WAV 文件；确认要重做时添加 --overwrite。"
            )

    overwrite_flag = "-y" if args.overwrite else "-n"
    total = sum(len(files) for files in sources.values())
    completed = 0
    for source_name, relative_output in DATASETS:
        destination = output_dir / relative_output
        destination.mkdir(parents=True, exist_ok=True)
        files = sources[source_name]
        print(f"{source_name} -> {relative_output}: {len(files)} files", flush=True)

        for filename, input_path in files.items():
            output_path = destination / filename
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                overwrite_flag,
                "-i",
                str(input_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
            subprocess.run(command, check=True)
            completed += 1
            if completed % 500 == 0 or completed == total:
                print(f"Converted {completed}/{total}", flush=True)

    print(f"完成：{total} 个 WAV 已转换为 16 kHz 单声道，输出到 {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
