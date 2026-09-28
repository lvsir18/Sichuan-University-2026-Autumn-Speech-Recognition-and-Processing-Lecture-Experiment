"""Enhance the VoiceBank-DEMAND test split and compute paired metrics.

Run from any working directory after activating the CMGAN environment:
    python experiments/01-speech-enhancement/script/evaluate_cmgan_test.py

Outputs are written to experiments/01-speech-enhancement/output by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CMGAN_SRC = EXPERIMENT_DIR / "CMGAN" / "src"
DEFAULT_CHECKPOINT = CMGAN_SRC / "best_ckpt" / "ckpt"
DEFAULT_TEST_DIR = EXPERIMENT_DIR / "VoiceBank-DEMAND" / "test"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or a torch device string")
    parser.add_argument("--cut-len", type=int, default=16000 * 16)
    parser.add_argument("--overwrite", action="store_true", help="overwrite prior evaluation outputs")
    return parser.parse_args()


def paired_wavs(test_dir: Path) -> list[tuple[str, Path, Path]]:
    clean_dir = test_dir / "clean"
    noisy_dir = test_dir / "noisy"
    for directory in (clean_dir, noisy_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"找不到数据目录：{directory}")

    clean_files = {path.name: path for path in clean_dir.glob("*.wav")}
    noisy_files = {path.name: path for path in noisy_dir.glob("*.wav")}
    if not clean_files or not noisy_files:
        raise ValueError(f"在 {test_dir} 下没有找到成对 WAV 文件")
    if clean_files.keys() != noisy_files.keys():
        clean_only = sorted(clean_files.keys() - noisy_files.keys())[:5]
        noisy_only = sorted(noisy_files.keys() - clean_files.keys())[:5]
        raise ValueError(
            "干净/带噪文件名不匹配。"
            f"仅 clean 存在的示例：{clean_only}；仅 noisy 存在的示例：{noisy_only}"
        )
    return [(name, noisy_files[name], clean_files[name]) for name in sorted(noisy_files)]


def resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 环境检测不到可用 GPU。")
    return device


def load_model(torch, device, checkpoint: Path):
    # The repository's train.py saves model.state_dict() directly, and the
    # upstream evaluation.py loads it into this same TSCNet architecture.
    from models.generator import TSCNet

    n_fft = 400
    model = TSCNet(num_channel=64, num_features=n_fft // 2 + 1).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def enhance_one(model, noisy_path: Path, device, torch, soundfile, power_compress, power_uncompress, cut_len: int):
    import numpy as np

    noisy_np, sample_rate = soundfile.read(noisy_path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"{noisy_path.name} 采样率为 {sample_rate} Hz，CMGAN 要求 16000 Hz。")
    noisy = torch.from_numpy(noisy_np.T.copy()).to(device)
    if noisy.shape[0] != 1:
        raise ValueError(f"当前评测脚本要求单声道语音，{noisy_path.name} 有 {noisy.shape[0]} 个声道。")

    n_fft = 400
    hop = 100
    length = noisy.size(-1)
    energy = torch.sum(noisy**2.0, dim=-1)
    if torch.any(energy <= 0):
        raise ValueError(f"带噪语音全静音，无法归一化：{noisy_path}")
    scale = torch.sqrt(noisy.size(-1) / energy)
    noisy = noisy * scale.unsqueeze(-1)

    frame_num = int(math.ceil(length / hop))
    padded_len = frame_num * hop
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(math.ceil(padded_len / cut_len))
        while hop % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    window = torch.hamming_window(n_fft, device=device)
    noisy_spec = torch.view_as_real(
        torch.stft(noisy, n_fft, hop, window=window, onesided=True, return_complex=True)
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    with torch.inference_mode():
        est_real, est_imag = model(noisy_spec)
    est_real = est_real.permute(0, 1, 3, 2)
    est_imag = est_imag.permute(0, 1, 3, 2)
    est_spec = power_uncompress(est_real, est_imag).squeeze(1)
    enhanced = torch.istft(
        torch.view_as_complex(est_spec.contiguous()),
        n_fft,
        hop,
        window=window,
        onesided=True,
    )
    enhanced = torch.flatten(enhanced / scale.unsqueeze(-1))[:length].cpu().numpy()
    return np.asarray(enhanced, dtype=np.float32), sample_rate


def snr_db(reference, estimate) -> float:
    import numpy as np

    error = reference - estimate
    signal_energy = float(np.sum(reference * reference))
    error_energy = float(np.sum(error * error))
    if signal_energy <= 0:
        return float("nan")
    if error_energy <= np.finfo(np.float64).tiny:
        return float("inf")
    return 10.0 * math.log10(signal_energy / error_energy)


def si_sdr_db(reference, estimate) -> float:
    import numpy as np

    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= np.finfo(np.float64).tiny:
        return float("nan")
    scale = float(np.dot(estimate, reference)) / reference_energy
    target = scale * reference
    residual = estimate - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if residual_energy <= np.finfo(np.float64).tiny:
        return float("inf")
    return 10.0 * math.log10(target_energy / residual_energy)


def paired_metrics(reference, estimate, sample_rate: int, pesq_fn, stoi_fn) -> dict[str, float]:
    import numpy as np

    length = min(len(reference), len(estimate))
    reference = np.asarray(reference[:length], dtype=np.float64)
    estimate = np.asarray(estimate[:length], dtype=np.float64)
    result = {
        "snr_db": snr_db(reference, estimate),
        "si_sdr_db": si_sdr_db(reference, estimate),
        "stoi": float("nan"),
        "pesq": float("nan"),
    }
    scoring_peak = max(float(np.max(np.abs(reference))), float(np.max(np.abs(estimate))), 1.0)
    scoring_reference = reference / scoring_peak
    scoring_estimate = estimate / scoring_peak
    try:
        result["stoi"] = float(stoi_fn(scoring_reference, scoring_estimate, sample_rate))
    except Exception as exc:
        result["stoi_error"] = str(exc)
    try:
        result["pesq"] = float(pesq_fn(sample_rate, scoring_reference, scoring_estimate, "wb"))
    except Exception as exc:
        result["pesq_error"] = str(exc)
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_progress(index: int, total: int, name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    elapsed_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    remaining_text = time.strftime("%H:%M:%S", time.gmtime(remaining))

    if not sys.stdout.isatty():
        if index == 1 or index % 50 == 0 or index == total:
            print(
                f"Processed {index}/{total} ({index / total:.1%}) | "
                f"elapsed {elapsed_text} | ETA {remaining_text} | {name}",
                flush=True,
            )
        return

    width = 24
    filled = round(width * index / total)
    bar = "#" * filled + "-" * (width - filled)
    status = (
        f"[{bar}] {index}/{total} ({index / total:.1%}) "
        f"elapsed {elapsed_text} | ETA {remaining_text} | {name}"
    )
    print("\r" + status.ljust(120), end="", flush=True)
    if index == total:
        print(flush=True)


def summarize(rows: list[dict]) -> list[dict]:
    import numpy as np

    metrics = [
        ("SNR", "snr_db", "dB"),
        ("SI-SDR", "si_sdr_db", "dB"),
        ("STOI", "stoi", "分"),
        ("PESQ", "pesq", "分"),
    ]
    summary = []
    for label, key, unit in metrics:
        noisy_values = np.asarray([row[f"noisy_{key}"] for row in rows], dtype=np.float64)
        enhanced_values = np.asarray([row[f"enhanced_{key}"] for row in rows], dtype=np.float64)
        valid = np.isfinite(noisy_values) & np.isfinite(enhanced_values)
        noisy_values = noisy_values[valid]
        enhanced_values = enhanced_values[valid]
        deltas = enhanced_values - noisy_values
        summary.append({
            "metric": label,
            "unit": unit,
            "valid_pairs": int(valid.sum()),
            "noisy_mean": float(np.mean(noisy_values)) if len(noisy_values) else float("nan"),
            "enhanced_mean": float(np.mean(enhanced_values)) if len(enhanced_values) else float("nan"),
            "delta_mean": float(np.mean(deltas)) if len(deltas) else float("nan"),
            "noisy_std": float(np.std(noisy_values, ddof=1)) if len(noisy_values) > 1 else float("nan"),
            "enhanced_std": float(np.std(enhanced_values, ddof=1)) if len(enhanced_values) > 1 else float("nan"),
        })
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    if not CMGAN_SRC.is_dir():
        raise FileNotFoundError(f"找不到 CMGAN 源码目录：{CMGAN_SRC}")
    if args.cut_len < 100:
        raise ValueError("--cut-len 必须至少为 100 个采样点。")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"输出目录非空：{output_dir}\n"
            "请先移走已有结果，或确认后添加 --overwrite。"
        )

    pairs = paired_wavs(test_dir)
    sys.path.insert(0, str(CMGAN_SRC))
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from pesq import pesq
        from models.generator import TSCNet
        from tools.compute_metrics import stoi
        from utils import power_compress, power_uncompress
    except Exception as exc:
        raise RuntimeError(
            "无法导入评测依赖。请先激活 CMGAN 环境，并安装项目依赖（含 PESQ）。"
        ) from exc

    device = resolve_device(torch, args.device)
    model = load_model(torch, device, checkpoint)
    enhanced_dir = output_dir / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Test pairs: {len(pairs)}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    rows = []
    started_at = time.perf_counter()
    for index, (name, noisy_path, clean_path) in enumerate(pairs, start=1):
        noisy, noisy_sr = sf.read(noisy_path, dtype="float32", always_2d=True)
        clean, clean_sr = sf.read(clean_path, dtype="float32", always_2d=True)
        if noisy_sr != 16000 or clean_sr != 16000:
            raise ValueError(f"{name}: noisy/clean 必须都是 16000 Hz（实际 {noisy_sr}/{clean_sr} Hz）")
        if noisy.shape[1] != 1 or clean.shape[1] != 1:
            raise ValueError(f"{name}: noisy/clean 文件必须为单声道")
        noisy = noisy[:, 0]
        clean = clean[:, 0]
        if len(noisy) != len(clean):
            raise ValueError(f"{name}: noisy/clean 长度不一致（{len(noisy)} / {len(clean)}）")

        enhanced, sample_rate = enhance_one(
            model, noisy_path, device, torch, sf, power_compress, power_uncompress, args.cut_len
        )
        sf.write(enhanced_dir / name, enhanced, sample_rate, subtype="PCM_16")
        noisy_metrics = paired_metrics(clean, noisy, sample_rate, pesq, stoi)
        enhanced_metrics = paired_metrics(clean, enhanced, sample_rate, pesq, stoi)

        row = {"file": name}
        for key in ("snr_db", "si_sdr_db", "stoi", "pesq"):
            row[f"noisy_{key}"] = noisy_metrics[key]
            row[f"enhanced_{key}"] = enhanced_metrics[key]
            row[f"delta_{key}"] = enhanced_metrics[key] - noisy_metrics[key]
        row["noisy_stoi_error"] = noisy_metrics.get("stoi_error", "")
        row["enhanced_stoi_error"] = enhanced_metrics.get("stoi_error", "")
        row["noisy_pesq_error"] = noisy_metrics.get("pesq_error", "")
        row["enhanced_pesq_error"] = enhanced_metrics.get("pesq_error", "")
        rows.append(row)
        print_progress(index, len(pairs), name, started_at)

    per_file_path = output_dir / "metrics_per_file.csv"
    summary_path = output_dir / "metrics_summary.csv"
    per_file_fields = ["file"]
    for key in ("snr_db", "si_sdr_db", "stoi", "pesq"):
        per_file_fields.extend([f"noisy_{key}", f"enhanced_{key}", f"delta_{key}"])
    per_file_fields.extend([
        "noisy_stoi_error", "enhanced_stoi_error", "noisy_pesq_error", "enhanced_pesq_error"
    ])
    write_csv(per_file_path, per_file_fields, rows)

    summary = summarize(rows)
    summary_fields = ["metric", "unit", "valid_pairs", "noisy_mean", "enhanced_mean", "delta_mean", "noisy_std", "enhanced_std"]
    write_csv(summary_path, summary_fields, summary)
    run_info = {
        "checkpoint": str(checkpoint),
        "test_dir": str(test_dir),
        "test_pairs": len(pairs),
        "sample_rate_hz": 16000,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "metric_aggregation": "per-file scores averaged across paired test utterances",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": ["enhanced/", "metrics_per_file.csv", "metrics_summary.csv"],
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nMean scores (enhanced - noisy):", flush=True)
    for item in summary:
        print(
            f"{item['metric']}: {item['noisy_mean']:.4f} -> {item['enhanced_mean']:.4f} "
            f"(Δ {item['delta_mean']:+.4f}, n={item['valid_pairs']})",
            flush=True,
        )
    print(f"\nSaved: {per_file_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Saved: {output_dir / 'run_info.json'}", flush=True)


if __name__ == "__main__":
    main()
