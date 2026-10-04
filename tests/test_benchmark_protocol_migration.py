import json

from script import benchmark_strategies


def test_legacy_ranking_is_not_reused_for_new_speaker_split(tmp_path, monkeypatch):
    path = tmp_path / "scores.json"
    legacy = {"strategies": {"old": {"female_test_acc": 99.0}}}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(benchmark_strategies, "LEADERBOARD_JSON", path)
    hashes = {"train": "new-train", "val": "new-val", "test": "new-test"}
    monkeypatch.setattr(benchmark_strategies, "split_hashes", lambda: hashes)
    assert benchmark_strategies.load_leaderboard() == {"strategies": {}}
    legacy["split_hashes"] = hashes
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert benchmark_strategies.load_leaderboard() == legacy
