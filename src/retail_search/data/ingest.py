from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import yaml

from retail_search.adapters.amazon_esci import ESCI_GAIN
from retail_search.data.split import assert_disjoint_queries, write_split_manifests

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, progress: Callable[[int], None] | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers: dict[str, str] = {}
    existing = partial.stat().st_size if partial.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        downloaded = existing if mode == "ab" else 0
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded)
    partial.replace(destination)


def is_valid_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as stream:
        header = stream.read(4)
        stream.seek(-4, 2)
        footer = stream.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def download_official_dataset(config_path: Path = Path("configs/data.yaml")) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(config["paths"]["raw_dir"])
    base_url = config["source"]["base_url"].rstrip("/")
    files = [config["source"]["examples_file"], config["source"]["products_file"]]
    report: dict[str, object] = {"source": config["source"], "files": {}}
    for name in files:
        destination = raw_dir / name
        if not is_valid_parquet(destination):
            LOGGER.info("Downloading %s", name)
            download_file(f"{base_url}/{name}", destination)
        if not is_valid_parquet(destination):
            raise ValueError(
                f"Downloaded file is not a parquet file: {destination}. "
                "A Git LFS pointer may have been returned instead of media content."
            )
        report["files"][name] = {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def prepare_amazon_esci(config_path: Path = Path("configs/data.yaml")) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    examples_path = raw_dir / config["source"]["examples_file"]
    products_path = raw_dir / config["source"]["products_file"]
    if not examples_path.exists() or not products_path.exists():
        raise FileNotFoundError("Official parquet files are missing; run the download command first")

    examples = pd.read_parquet(
        examples_path,
        columns=[
            "query_id", "query", "product_id", "product_locale", "esci_label",
            "small_version", "split",
        ],
        filters=[("small_version", "=", int(config["scope"]["small_version"])),
                 ("product_locale", "=", config["scope"]["locale"])],
    )
    pre_filter_count = len(examples)
    examples = examples.rename(columns={"split": "source_split"})
    products = pd.read_parquet(
        products_path,
        columns=[
            "product_id", "product_locale", "product_title", "product_description",
            "product_bullet_point", "product_brand", "product_color",
        ],
        filters=[("product_locale", "=", config["scope"]["locale"])],
    )
    frame = examples.merge(products, on=["product_id", "product_locale"], how="left", validate="many_to_one")
    frame["query"] = frame["query"].fillna("").astype(str).str.strip()
    frame["title"] = frame["product_title"].fillna("").astype(str).str.strip()
    invalid = (frame["query"] == "") | (frame["title"] == "") | ~frame["esci_label"].isin(ESCI_GAIN)
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["relevance_label"] = frame["esci_label"].map(ESCI_GAIN).astype("int8")

    from retail_search.data.split import assign_query_splits

    frame["split"] = assign_query_splits(
        frame,
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    assert_disjoint_queries(frame)
    manifest_hashes = write_split_manifests(frame, processed_dir / "splits")
    keep = [
        "query_id", "query", "product_id", "product_locale", "title",
        "product_description", "product_bullet_point", "product_brand", "product_color",
        "esci_label", "relevance_label", "source_split", "split",
    ]
    frame[keep].to_parquet(processed_dir / "esci_us_task1.parquet", index=False, compression="zstd")
    judgments_per_query = frame.groupby("query_id").size()
    report = {
        "dataset": "Amazon Shopping Queries (ESCI) US Task-1 reduced",
        "source_repository": config["source"]["repository"],
        "source_revision": config["source"]["revision"],
        "pre_filter_judgments": pre_filter_count,
        "invalid_rows_dropped": invalid_count,
        "valid_judgments": len(frame),
        "unique_queries": int(frame["query_id"].nunique()),
        "unique_products": int(frame["product_id"].nunique()),
        "judgments_per_query": {
            "maximum": int(judgments_per_query.max()),
            "p50": float(judgments_per_query.quantile(0.50)),
            "p95": float(judgments_per_query.quantile(0.95)),
            "queries_over_50": int((judgments_per_query > 50).sum()),
            "queries_over_100": int((judgments_per_query > 100).sum()),
        },
        "label_distribution": {str(k): int(v) for k, v in frame["esci_label"].value_counts().sort_index().items()},
        "splits": {
            split: {
                "queries": int(part["query_id"].nunique()),
                "judgments": len(part),
                "manifest_sha256": manifest_hashes[split],
            }
            for split, part in frame.groupby("split", sort=True)
        },
        "minimum_judgments": int(config["scope"]["minimum_judgments"]),
        "minimum_met": len(frame) >= int(config["scope"]["minimum_judgments"]),
    }
    (processed_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["minimum_met"]:
        raise ValueError(
            f"Only {len(frame):,} valid judgments were found; "
            f"the configured minimum is {report['minimum_judgments']:,}"
        )
    return report
