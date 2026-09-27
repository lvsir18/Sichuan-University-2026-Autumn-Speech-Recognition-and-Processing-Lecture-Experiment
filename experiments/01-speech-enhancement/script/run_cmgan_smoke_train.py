"""Run a one-epoch CMGAN smoke-training pass using one bundled audio pair.

Run this script with the Python interpreter from the prepared CMGAN environment:
    python experiments/01-speech-enhancement/script/run_cmgan_smoke_train.py
"""

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    experiment_dir = Path(__file__).resolve().parents[1]
    cmgan_dir = experiment_dir / "CMGAN"
    source_clean = cmgan_dir / "AudioSamples" / "clean" / "p257_143.wav"
    source_noisy = cmgan_dir / "AudioSamples" / "noisy" / "p257_143.wav"
    train_script = cmgan_dir / "src" / "train.py"
    dataset_dir = cmgan_dir / "smoke_dataset"
    checkpoint_dir = cmgan_dir / "src" / "smoke_ckpt"

    required_files = (source_clean, source_noisy, train_script)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Required CMGAN files not found:\n" + "\n".join(missing_files))

    # The training loader expects train/test splits with clean/noisy subfolders.
    for split in ("train", "test"):
        for kind in ("clean", "noisy"):
            (dataset_dir / split / kind).mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        shutil.copy2(source_clean, dataset_dir / split / "clean" / source_clean.name)
        shutil.copy2(source_noisy, dataset_dir / split / "noisy" / source_noisy.name)

    command = [
        sys.executable,
        str(train_script),
        "--data_dir",
        str(dataset_dir),
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--log_interval",
        "1",
        "--save_model_dir",
        str(checkpoint_dir),
    ]

    print(f"Running CMGAN smoke training with: {sys.executable}", flush=True)
    subprocess.run(command, cwd=cmgan_dir, check=True)
    print(f"Smoke training completed. Checkpoint directory: {checkpoint_dir}")


if __name__ == "__main__":
    main()
