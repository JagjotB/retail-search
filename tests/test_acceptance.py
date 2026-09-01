import json
from pathlib import Path

from retail_search.evaluation.acceptance import build_acceptance_report


def test_acceptance_fails_truthfully_when_gain_is_below_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    benchmark = {
        "generated_at": "2026-09-01T00:00:00Z",
        "split_manifest_hash": "hash",
        "dataset": {"valid_judgments": 601354},
        "systems": {"embedding_only_baseline": {"query_count": 5}},
        "model": {"ranker": "LightGBM LambdaMART"},
        "quality_gate": {"relative_ndcg_at_10_gain": 0.109},
    }
    (artifact_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    report = build_acceptance_report(artifact_dir, require_runtime_evidence=False)
    gain_check = next(item for item in report["checks"] if item["criterion"] == "relative_ndcg_at_10_gain")
    assert gain_check["passed"] is False
    assert report["overall_passed"] is False
