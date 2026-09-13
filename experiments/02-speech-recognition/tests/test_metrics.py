from app.metrics import calculate_metrics, edit_counts, normalize_chinese


def test_normalize_chinese_removes_punctuation():
    assert normalize_chinese("你好，ASR！") == "你好asr"


def test_edit_counts_reports_operations():
    result = edit_counts("天气很好", "天齐好")
    assert result.errors == 2
    assert result.reference_length == 4


def test_metrics_exact_match():
    metrics = calculate_metrics("语音识别", "语音识别", 0.5, 2.0)
    assert metrics["cer"] == 0
    assert metrics["ser"] == 0
    assert metrics["rtf"] == 0.25

