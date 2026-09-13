"""ASR 评价指标：编辑距离、CER、WER、SER 和 RTF。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_length if self.reference_length else float(self.errors > 0)


def normalize_chinese(text: str) -> str:
    """保留中文、字母和数字，并统一英文大小写。"""
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower()))


def normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff']+", " ", text.lower(), flags=re.UNICODE)
    return [item for item in text.split() if item]


def edit_counts(reference: list[str] | str, hypothesis: list[str] | str) -> EditCounts:
    ref = list(reference)
    hyp = list(hypothesis)
    # 单元格存储 (总错误, 替换, 删除, 插入)，并采用稳定的平局顺序。
    table: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            previous = table[i - 1][j - 1]
            substitution = (previous[0] + 1, previous[1] + 1, previous[2], previous[3])
            previous = table[i - 1][j]
            deletion = (previous[0] + 1, previous[1], previous[2] + 1, previous[3])
            previous = table[i][j - 1]
            insertion = (previous[0] + 1, previous[1], previous[2], previous[3] + 1)
            table[i][j] = min(substitution, deletion, insertion, key=lambda value: (value[0], value[1], value[2], value[3]))
    _, substitutions, deletions, insertions = table[-1][-1]
    return EditCounts(substitutions, deletions, insertions, len(ref))


def calculate_metrics(reference: str, hypothesis: str, elapsed_seconds: float, audio_seconds: float) -> dict:
    char_edits = edit_counts(normalize_chinese(reference), normalize_chinese(hypothesis))
    word_edits = edit_counts(normalize_words(reference), normalize_words(hypothesis))
    return {
        "cer": round(char_edits.rate, 4),
        "wer": round(word_edits.rate, 4),
        "ser": float(normalize_chinese(reference) != normalize_chinese(hypothesis)),
        "rtf": round(elapsed_seconds / audio_seconds, 4) if audio_seconds > 0 else None,
        "char_edits": {
            "substitutions": char_edits.substitutions,
            "deletions": char_edits.deletions,
            "insertions": char_edits.insertions,
            "reference_length": char_edits.reference_length,
        },
        "word_edits": {
            "substitutions": word_edits.substitutions,
            "deletions": word_edits.deletions,
            "insertions": word_edits.insertions,
            "reference_length": word_edits.reference_length,
        },
    }

