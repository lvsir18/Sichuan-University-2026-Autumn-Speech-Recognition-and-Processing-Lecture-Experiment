import json
import hashlib

from scripts import prepare_dataset as module


def test_find_audio_url_handles_nested_audio_value():
    value = {"array": None, "preview": [{"src": "https://example.test/sample.wav", "type": "audio/wav"}]}
    assert module.find_audio_url(value) == "https://example.test/sample.wav"


def test_prepare_and_validate_fixed_dataset(tmp_path, monkeypatch):
    spec = {
        "course_dataset_id": "test-mini-v1",
        "dataset_id": "example/fleurs",
        "origin_dataset": "google/fleurs",
        "config": "cmn_hans_cn",
        "split": "test",
        "offset": 0,
        "length": 2,
        "name": "test mini",
        "license": "CC BY 4.0",
        "homepage": "https://example.test",
        "transcript_url": "https://example.test/trans.txt",
        "audio_url_template": "https://example.test/cmn_hans_cn_{index:04d}.wav",
        "expected_sha256": [hashlib.sha256(b"RIFF" + b"\0" * 2048).hexdigest()] * 2,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "dataset"
    def fake_request(url, timeout=90):
        if url.endswith("trans.txt"):
            return "cmn_hans_cn_0000 你 好\ncmn_hans_cn_0001 语 音 识 别\n".encode(), "text/plain"
        return b"RIFF" + b"\0" * 2048, "audio/wav"

    monkeypatch.setattr(module, "request_bytes", fake_request)
    manifest = module.prepare_dataset(output, spec_path)
    assert len(manifest["samples"]) == 2
    assert module.validate_existing(output, spec) == manifest
    assert manifest["samples"][0]["reference"] == "你好"
