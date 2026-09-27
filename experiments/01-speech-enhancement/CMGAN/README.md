# CMGAN: Conformer-Based Metric GAN for Monaural Speech Enhancement (https://ieeexplore.ieee.org/document/10508391)

## Abstract:
Recently, convolution-augmented transformer (Conformer) has achieved promising performance in automatic speech recognition (ASR) and time-domain speech enhancement (SE), as it can capture both local and global dependencies in the speech signal. In this paper, we propose a conformer-based metric generative adversarial network (CMGAN) for SE in the time-frequency (TF) domain. In the generator, we utilize two-stage conformer blocks to aggregate all magnitude and complex spectrogram information by modeling both time and frequency dependencies. The estimation of magnitude and complex spectrogram is decoupled in the decoder stage and then jointly incorporated to reconstruct the enhanced speech. In addition, a metric discriminator is employed to further improve the quality of the enhanced estimated speech by optimizing the generator with respect to a corresponding evaluation score. Quantitative analysis on Voice Bank+DEMAND dataset indicates the capability of CMGAN in outperforming various previous models with a margin, i.e., PESQ of 3.41 and SSNR of 11.10 dB. 

[Demo of audio samples](https://sherifabdulatif.github.io/cmgan/) 

A longer detailed version is now available on [IEEE/ACM Transactions on Audio, Speech, and Language Processing](https://ieeexplore.ieee.org/document/10508391), [arXiv Version](https://arxiv.org/abs/2209.11112).

The short manuscript is published in [INTERSPEECH2022](https://www.isca-speech.org/archive/interspeech_2022/cao22_interspeech.html). 

Source code is released!

## Run the project on Windows (single GPU)

Use Python 3.10 with matching PyTorch and TorchAudio 2.11.0 CUDA 13.0 wheels.
The CUDA runtime is bundled with the PyTorch wheels; a separate CUDA Toolkit installation is not required.

```powershell
conda create -n cmgan-modern python=3.10 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate cmgan-modern
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130 --timeout 120 --retries 10
conda install --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge pesq=0.0.4
cd src
python -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
```

Conda packages and ordinary Python packages use Tsinghua mirrors; the CUDA PyTorch wheels use the NJU mirror. The pip source options must be repeated for `requirements.txt` because they only apply to the command where they are specified.

Verify the installation:

```powershell
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Run the provided audio samples

From `src`, run enhancement and metric evaluation with the included checkpoint:

```powershell
python evaluation.py --test_dir "..\AudioSamples" --model_path ".\best_ckpt\ckpt" --save_dir ".\saved_tracks"
```

Enhanced wav files are saved in `src/saved_tracks`.

### Train

Download VCTK-DEMAND at 16 kHz and arrange it as follows:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```

Start single-GPU training from `src`:

```powershell
python train.py --data_dir "D:\path\to\VCTK-DEMAND" --batch_size 1
```

### Evaluate a dataset

For a VCTK-DEMAND test split, the directory should contain `clean` and `noisy`:

```powershell
python evaluation.py --test_dir "D:\path\to\VCTK-DEMAND\test" --model_path ".\saved_model\<checkpoint>"
```

## Model and Comparison:
The detailed architecture of CMGAN with both generator and discriminator. <br><br>
<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figures/Overview.PNG" width="600px">

Performance comparison on the Voice Bank+DEMAND dataset. “-” denotes the result is not provided in the
original paper. Model size represents the number of trainable parameters in million. <br><br>
<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figures/Table.PNG" width="600px">

## Long version citation:
```
@misc{abdulatif2022cmgan,
  title={CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement}, 
  author={Abdulatif, Sherif and Cao, Ruizhe and Yang, Bin},
  year={2022},
  eprint={2209.11112},
  archivePrefix={arXiv}
}
```


## Short version citation:
```
@inproceedings{cao22_interspeech,
  author={Cao, Ruizhe and Abdulatif, Sherif and Yang, Bin},
  title={{CMGAN: Conformer-based Metric GAN for Speech Enhancement}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={936--940},
  doi={10.21437/Interspeech.2022-517}
}
```
