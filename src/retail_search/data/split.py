from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def stable_bucket(value: object, seed: int = 20260901) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_query_splits(
    frame: pd.DataFrame,
    validation_fraction: float = 0.15,
    seed: int = 20260901,
) -> pd.Series:
    """Keep Amazon's official test split frozen and hash-split only official train queries."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    query_source = frame[["query_id", "source_split"]].drop_duplicates()
    overlap = query_source.groupby("query_id")["source_split"].nunique()
    if (overlap > 1).any():
        raise ValueError("A query_id appears in multiple official source splits")
    mapping: dict[object, str] = {}
    for row in query_source.itertuples(index=False):
        if row.source_split == "test":
            mapping[row.query_id] = "test"
        elif stable_bucket(row.query_id, seed) < validation_fraction:
            mapping[row.query_id] = "validation"
        else:
            mapping[row.query_id] = "train"
    return frame["query_id"].map(mapping)


def assert_disjoint_queries(frame: pd.DataFrame) -> None:
    grouped = frame.groupby("query_id")["split"].nunique()
    if (grouped > 1).any():
        sample = grouped[grouped > 1].index[:5].tolist()
        raise AssertionError(f"Query leakage across splits: {sample}")


def write_split_manifests(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        query_ids = sorted(str(value) for value in frame.loc[frame["split"] == split, "query_id"].unique())
        payload = {"split": split, "query_count": len(query_ids), "query_ids": query_ids}
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path = output_dir / f"{split}_queries.json"
        path.write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hashes


def manifest_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()
