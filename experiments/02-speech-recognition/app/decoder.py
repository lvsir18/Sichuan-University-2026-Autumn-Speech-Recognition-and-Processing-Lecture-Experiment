"""小型 CTC 解码器，用于直观比较贪心、束搜索和 n-gram 语言模型。"""

from __future__ import annotations

import math
from collections import defaultdict


NEG_INF = float("-inf")


def _log_add(*values: float) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def greedy_decode(probabilities: list[list[float]], tokens: list[str], blank: str = "_") -> str:
    sequence = [tokens[max(range(len(row)), key=row.__getitem__)] for row in probabilities]
    output: list[str] = []
    previous = None
    for token in sequence:
        if token != blank and token != previous:
            output.append(token)
        previous = token
    return "".join(output)


def _bigram_score(text: str, corpus: list[str]) -> float:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    contexts: dict[str, int] = defaultdict(int)
    vocabulary = set("".join(corpus))
    for sentence in corpus:
        padded = "^" + sentence + "$"
        for left, right in zip(padded, padded[1:]):
            counts[(left, right)] += 1
            contexts[left] += 1
    score = 0.0
    padded = "^" + text + "$"
    size = max(len(vocabulary) + 1, 1)
    for left, right in zip(padded, padded[1:]):
        score += math.log((counts[(left, right)] + 1) / (contexts[left] + size))
    return score


def prefix_beam_search(
    probabilities: list[list[float]],
    tokens: list[str],
    beam_width: int = 5,
    lm_weight: float = 0.0,
    corpus: list[str] | None = None,
    blank: str = "_",
) -> tuple[str, list[dict]]:
    blank_index = tokens.index(blank)
    beams: dict[str, tuple[float, float]] = {"": (0.0, NEG_INF)}
    for row in probabilities:
        next_beams: dict[str, tuple[float, float]] = defaultdict(lambda: (NEG_INF, NEG_INF))
        for prefix, (p_blank, p_non_blank) in beams.items():
            for index, probability in enumerate(row):
                logp = math.log(max(probability, 1e-12))
                if index == blank_index:
                    old_blank, old_non_blank = next_beams[prefix]
                    next_beams[prefix] = (_log_add(old_blank, p_blank + logp, p_non_blank + logp), old_non_blank)
                    continue
                token = tokens[index]
                end = prefix[-1:] if prefix else ""
                new_prefix = prefix + token
                if token == end:
                    old_blank, old_non_blank = next_beams[prefix]
                    next_beams[prefix] = (old_blank, _log_add(old_non_blank, p_non_blank + logp))
                    old_blank, old_non_blank = next_beams[new_prefix]
                    next_beams[new_prefix] = (old_blank, _log_add(old_non_blank, p_blank + logp))
                else:
                    old_blank, old_non_blank = next_beams[new_prefix]
                    next_beams[new_prefix] = (old_blank, _log_add(old_non_blank, p_blank + logp, p_non_blank + logp))
        def rank(item: tuple[str, tuple[float, float]]) -> float:
            prefix, values = item
            acoustic = _log_add(*values)
            return acoustic + lm_weight * _bigram_score(prefix, corpus or []) if corpus and lm_weight else acoustic
        beams = dict(sorted(next_beams.items(), key=rank, reverse=True)[:beam_width])
    candidates = []
    for text, values in beams.items():
        acoustic = _log_add(*values)
        lm_score = _bigram_score(text, corpus or []) if corpus else 0.0
        candidates.append({"text": text, "acoustic_score": round(acoustic, 4), "lm_score": round(lm_score, 4), "total_score": round(acoustic + lm_weight * lm_score, 4)})
    candidates.sort(key=lambda item: item["total_score"], reverse=True)
    return candidates[0]["text"], candidates


TOKENS = ["_", "今", "天", "气", "齐", "很", "好"]
# 每一帧的 CTC 概率。声学模型稍偏向“齐”，语言模型则能利用“天气”搭配纠正它。
DEMO_PROBABILITIES = [
    [0.05, 0.88, 0.02, 0.01, 0.01, 0.02, 0.01],
    [0.75, 0.04, 0.14, 0.02, 0.02, 0.02, 0.01],
    [0.04, 0.02, 0.86, 0.02, 0.02, 0.02, 0.02],
    [0.72, 0.02, 0.18, 0.02, 0.02, 0.02, 0.02],
    [0.04, 0.02, 0.86, 0.02, 0.02, 0.02, 0.02],
    [0.72, 0.02, 0.18, 0.02, 0.02, 0.02, 0.02],
    [0.04, 0.01, 0.02, 0.39, 0.48, 0.03, 0.03],
    [0.70, 0.02, 0.02, 0.08, 0.09, 0.05, 0.04],
    [0.05, 0.01, 0.02, 0.02, 0.02, 0.84, 0.04],
    [0.72, 0.02, 0.02, 0.02, 0.02, 0.14, 0.06],
    [0.05, 0.01, 0.01, 0.01, 0.01, 0.03, 0.88],
]
DEMO_CORPUS = ["今天天气很好", "今天天气不错", "天气很好", "明天天气晴朗", "今天心情很好"]


def run_decoder_demo(beam_width: int = 5, lm_weight: float = 0.0) -> dict:
    greedy = greedy_decode(DEMO_PROBABILITIES, TOKENS)
    beam, candidates = prefix_beam_search(DEMO_PROBABILITIES, TOKENS, beam_width, lm_weight, DEMO_CORPUS)
    return {
        "reference": "今天天气很好",
        "greedy": greedy,
        "beam": beam,
        "beam_width": beam_width,
        "lm_weight": lm_weight,
        "candidates": candidates,
        "tokens": TOKENS,
        "probabilities": DEMO_PROBABILITIES,
    }
