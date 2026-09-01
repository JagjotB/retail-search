def test_promoted_artifact_loads_in_fresh_manager(artifact_manager) -> None:
    bundle = artifact_manager.load()
    assert bundle.version == "fixture-v1"
    assert bundle.manifest["quality_gate_passed"] is True
    assert bundle.index.retrieve("gaming mouse", 1)
