from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from retail_search.ranking.predict import LightGBMReranker
from retail_search.retrieval.index import NumpyVectorIndex


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ArtifactBundle:
    version: str
    embedder: Any
    feature_builder: Any
    reranker: LightGBMReranker
    index: NumpyVectorIndex
    manifest: dict[str, Any]
    benchmark: dict[str, Any]
    curated_queries: list[dict[str, Any]]


class ArtifactManager:
    def __init__(self, artifact_dir: Path = Path("artifacts"), pointer_path: Path | None = None):
        self.artifact_dir = artifact_dir
        self.pointer_path = pointer_path or artifact_dir / "promoted.json"

    def publish(
        self,
        version: str,
        embedder: Any,
        feature_builder: Any,
        reranker: LightGBMReranker,
        index: NumpyVectorIndex,
        metadata: dict[str, Any],
        quality_gate_passed: bool,
    ) -> Path:
        if not quality_gate_passed:
            raise ValueError("Refusing to promote an artifact that failed the quality gate")
        version_dir = self.artifact_dir / "models" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        core_files = [
            version_dir / "embedder.joblib",
            version_dir / "feature_builder.joblib",
            version_dir / "ranker.txt",
            version_dir / "index" / "vectors.npy",
            version_dir / "index" / "products.json",
        ]
        cross_encoder = getattr(feature_builder, "cross_encoder", None)
        cross_encoder_source = (
            Path(cross_encoder.model_dir) if cross_encoder is not None else None
        )
        cross_encoder_destination = version_dir / "cross_encoder"
        if cross_encoder_source is not None and cross_encoder_source.exists():
            shutil.copytree(
                cross_encoder_source,
                cross_encoder_destination,
                dirs_exist_ok=True,
            )
        # Versions are immutable. This guard also makes a metadata/checksum
        # refresh safe when an index was loaded as a Windows memory map.
        if not all(path.exists() for path in core_files):
            joblib.dump(embedder, version_dir / "embedder.joblib", compress=3)
            joblib.dump(feature_builder, version_dir / "feature_builder.joblib", compress=3)
            reranker.save(version_dir / "ranker.txt")
            index.save(version_dir / "index")
        files = list(core_files)
        if cross_encoder_destination.exists():
            files.extend(
                sorted(path for path in cross_encoder_destination.rglob("*") if path.is_file())
            )
        manifest = {
            "version": version,
            "quality_gate_passed": True,
            "metadata": metadata,
            "files": {
                str(path.relative_to(version_dir)).replace("\\", "/"): {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in files
            },
        }
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pointer = {"version": version, "path": str(version_dir).replace("\\", "/")}
        temporary = self.pointer_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.pointer_path)
        return version_dir

    def _validate_files(self, version_dir: Path, manifest: dict[str, Any]) -> None:
        for name, expected in manifest["files"].items():
            path = version_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Artifact file is missing: {path}")
            if file_sha256(path) != expected["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {path}")

    def load(self) -> ArtifactBundle:
        if not self.pointer_path.exists():
            raise FileNotFoundError(
                f"Promoted artifact pointer not found at {self.pointer_path}; run the benchmark first"
            )
        pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        version_dir = Path(pointer["path"])
        manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        self._validate_files(version_dir, manifest)
        embedder = joblib.load(version_dir / "embedder.joblib")
        feature_builder = joblib.load(version_dir / "feature_builder.joblib")
        if (
            getattr(feature_builder, "cross_encoder", None) is not None
            and (version_dir / "cross_encoder").exists()
        ):
            feature_builder.cross_encoder.model_dir = str(version_dir / "cross_encoder")
        feature_names = manifest["metadata"]["feature_names"]
        reranker = LightGBMReranker.load(version_dir / "ranker.txt", feature_names)
        index = NumpyVectorIndex.load(version_dir / "index", embedder)
        benchmark = json.loads((self.artifact_dir / "benchmark.json").read_text(encoding="utf-8"))
        curated = json.loads((self.artifact_dir / "demo" / "curated_queries.json").read_text(encoding="utf-8"))
        return ArtifactBundle(pointer["version"], embedder, feature_builder, reranker, index, manifest, benchmark, curated)
