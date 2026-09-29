"""Run full CMGAN training with the VoiceBank-DEMAND dataset.

Run from the experiment directory with the prepared Python environment active:
    python ./script/run_cmgan_train.py
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def build_parser(experiment_dir):
    parser = argparse.ArgumentParser(
        description="Train CMGAN using the local VoiceBank-DEMAND dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=experiment_dir / "VoiceBank-DEMAND",
        help="dataset root containing train/ and test/ (default: VoiceBank-DEMAND)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--decay-epoch", type=int, default=12)
    parser.add_argument("--init-lr", type=float, default=5e-4)
    parser.add_argument("--cut-len", type=int, default=32000)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="physical GPU index from nvidia-smi (default: 0)",
    )
    parser.add_argument(
        "--save-model-dir",
        type=Path,
        default=experiment_dir / "CMGAN" / "checkpoints",
        help="directory for per-epoch generator weights",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard event directory (default: a timestamped CMGAN/runs subdirectory)",
    )
    return parser


def validate_dataset(data_dir, batch_size):
    counts = {}
    for split in ("train", "test"):
        clean_dir = data_dir / split / "clean"
        noisy_dir = data_dir / split / "noisy"
        for directory in (clean_dir, noisy_dir):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Dataset directory not found: {directory}\n"
                    "Expected VoiceBank-DEMAND/{train,test}/{clean,noisy}."
                )

        clean_files = {
            path.name
            for path in clean_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        }
        noisy_files = {
            path.name
            for path in noisy_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".wav"
        }
        if not clean_files or not noisy_files:
            raise ValueError(f"No WAV files found in {clean_dir} or {noisy_dir}.")
        if clean_files != noisy_files:
            clean_only = sorted(clean_files - noisy_files)[:5]
            noisy_only = sorted(noisy_files - clean_files)[:5]
            raise ValueError(
                f"Clean/noisy filenames do not match in the {split} split. "
                f"Clean-only examples: {clean_only}; noisy-only examples: {noisy_only}"
            )
        counts[split] = len(clean_files)

    if counts["train"] < batch_size:
        raise ValueError(
            f"Training split has {counts['train']} pairs, fewer than batch size "
            f"{batch_size}; the training DataLoader drops incomplete batches."
        )
    return counts


def main():
    experiment_dir = Path(__file__).resolve().parents[1]
    cmgan_dir = experiment_dir / "CMGAN"
    source_dir = cmgan_dir / "src"
    train_script = source_dir / "train.py"

    args = build_parser(experiment_dir).parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be a non-negative GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_dir = args.data_dir.expanduser().resolve()
    checkpoint_dir = args.save_model_dir.expanduser().resolve()
    tensorboard_dir = args.tensorboard_dir
    if tensorboard_dir is None:
        run_name = datetime.now().strftime("cmgan_%Y%m%d_%H%M%S")
        tensorboard_dir = experiment_dir / "CMGAN" / "runs" / run_name
    tensorboard_dir = tensorboard_dir.expanduser().resolve()

    if not train_script.is_file():
        raise FileNotFoundError(f"CMGAN training script not found: {train_script}")
    counts = validate_dataset(data_dir, args.batch_size)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"Could not import PyTorch with {sys.executable}. Activate the CMGAN environment first."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. CMGAN's train.py currently requires an NVIDIA GPU."
        )

    command = [
        sys.executable,
        str(train_script),
        "--data_dir",
        str(data_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--log_interval",
        str(args.log_interval),
        "--decay_epoch",
        str(args.decay_epoch),
        "--init_lr",
        str(args.init_lr),
        "--cut_len",
        str(args.cut_len),
        "--save_model_dir",
        str(checkpoint_dir),
        "--tensorboard_dir",
        str(tensorboard_dir),
    ]

    print(f"Python: {sys.executable}", flush=True)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"(physical index {args.gpu}, visible to PyTorch as cuda:0)",
        flush=True,
    )
    print(f"Dataset: {data_dir} (train={counts['train']}, test={counts['test']} pairs)", flush=True)
    print(f"Epochs: {args.epochs}; batch size: {args.batch_size}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    print(f"TensorBoard logs: {tensorboard_dir}", flush=True)
    subprocess.run(command, cwd=source_dir, check=True)
    print("CMGAN training completed.", flush=True)


if __name__ == "__main__":
    main()
