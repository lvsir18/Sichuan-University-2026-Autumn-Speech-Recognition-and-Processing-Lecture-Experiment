# 固定实验数据集来源与许可

本项目的数据准备脚本通过 `FluidInference/fleurs-full` 文件镜像下载源自 `google/fleurs` 的 `cmn_hans_cn` 固定编号 0000～0011，课程内部版本为 `FLEURS-CMN-MINI-12-v1`。

- 数据集：FLEURS（Few-shot Learning Evaluation of Universal Representations of Speech）
- 项目主页：https://huggingface.co/datasets/google/fleurs
- 文件镜像：https://huggingface.co/datasets/FluidInference/fleurs-full
- 论文：https://arxiv.org/abs/2205.12446
- 许可：Creative Commons Attribution 4.0 International（CC BY 4.0）
- 许可全文：https://creativecommons.org/licenses/by/4.0/

FLEURS 数据卡列出的引用：

```bibtex
@inproceedings{conneau2023fleurs,
  title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages={798--805},
  year={2023},
  organization={IEEE}
}
```

音频不会提交到本代码仓库。项目配置预置了 12 个已核验文件的 SHA-256；下载器只接受完全匹配的文件。生成的 `data/fleurs-mini/manifest.json` 保存每条音频的编号、参考文本、大小和 SHA-256，以便课堂复核。
