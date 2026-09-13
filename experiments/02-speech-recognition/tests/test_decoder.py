from app.decoder import greedy_decode, run_decoder_demo


def test_ctc_greedy_collapses_repeated_tokens_and_blank():
    tokens = ["_", "a", "b"]
    probabilities = [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]
    assert greedy_decode(probabilities, tokens) == "ab"


def test_decoder_demo_returns_ranked_candidates():
    result = run_decoder_demo(5, 0.2)
    assert result["greedy"]
    assert result["beam"] == result["candidates"][0]["text"]
    assert len(result["candidates"]) <= 5


def test_bigram_lm_can_correct_acoustic_homophone():
    acoustic_only = run_decoder_demo(10, 0.0)
    with_language_model = run_decoder_demo(10, 0.1)
    assert acoustic_only["beam"] == "今天天齐很好"
    assert with_language_model["beam"] == "今天天气很好"
