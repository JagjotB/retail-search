from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _rich_text(frame: pd.DataFrame) -> list[str]:
    return (
        frame["title"].fillna("").astype(str)
        + " "
        + frame["product_brand"].fillna("").astype(str)
        + " "
        + frame["product_bullet_point"].fillna("").astype(str)
        + " "
        + frame["product_description"].fillna("").astype(str)
    ).str.strip().tolist()


def _pair_indices(frame: pd.DataFrame, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int, int]] = []
    for indices in frame.groupby("query_id", sort=False).indices.values():
        group_indices = np.asarray(indices, dtype=np.int64)
        labels = frame.iloc[group_indices]["relevance_label"].to_numpy(dtype=np.int8)
        for high_position, high_label in enumerate(labels):
            lower_positions = np.flatnonzero(labels < high_label)
            if not len(lower_positions):
                continue
            by_label = {}
            for low_position in lower_positions:
                by_label.setdefault(int(labels[low_position]), []).append(int(low_position))
            for low_label, positions in by_label.items():
                chosen = positions[int(rng.integers(0, len(positions)))]
                pairs.append(
                    (
                        int(group_indices[high_position]),
                        int(group_indices[chosen]),
                        int(high_label - low_label),
                    )
                )
    if len(pairs) > maximum:
        selected = rng.choice(len(pairs), size=maximum, replace=False)
        pair_array = np.asarray(pairs, dtype=np.int64)[selected]
    else:
        pair_array = np.asarray(pairs, dtype=np.int64)
    rng.shuffle(pair_array)
    return pair_array


def _split_manifest_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("data/processed/splits").glob("*_queries.json")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train_cross_encoder(config: dict[str, Any], force: bool = False) -> Path:
    output_dir = Path(config["local_dir"])
    output_xml = output_dir / "openvino" / "model.xml"
    if output_xml.exists() and not force:
        return output_dir
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Cross-encoder training requires the training extra: pip install -e '.[training]'"
        ) from error
    import openvino as ov

    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    frame = pd.read_parquet("data/processed/esci_us_task1.parquet")
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    queries = train["query"].astype(str).tolist()
    documents = _rich_text(train)
    labels = train["relevance_label"].to_numpy(dtype=np.int8)
    order = np.arange(len(train), dtype=np.int64)
    LOGGER.info("Fine-tuning cross-encoder on %s training judgments", f"{len(order):,}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["repository"], revision=config["revision"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["repository"],
        revision=config["revision"],
        num_labels=1,
        use_safetensors=True,
        attn_implementation="eager",
    ).to(device)
    LOGGER.info("Training cross-encoder on %s", device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=0.01
    )
    batch_size = int(settings["batch_size"])
    max_length = int(config["max_length"])
    total_steps = int(settings["epochs"]) * int(np.ceil(len(order) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    rng = np.random.default_rng(seed)
    dcg_targets = np.asarray([0.0, 1.0 / 7.0, 3.0 / 7.0, 1.0], dtype=np.float32)
    for epoch in range(int(settings["epochs"])):
        rng.shuffle(order)
        running = 0.0
        for batch_number, start in enumerate(range(0, len(order), batch_size), 1):
            batch = order[start : start + batch_size]
            encoded = tokenizer(
                [queries[index] for index in batch],
                [documents[index] for index in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits.reshape(-1)
            targets = torch.as_tensor(
                dcg_targets[labels[batch]], dtype=torch.float32, device=device
            )
            loss = functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            if batch_number % 200 == 0:
                LOGGER.info(
                    "Cross-encoder epoch %d batch %d/%d loss %.4f",
                    epoch + 1,
                    batch_number,
                    int(np.ceil(len(order) / batch_size)),
                    running / 200,
                )
                running = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.to("cpu").eval()
    example = tokenizer(
        ["wireless gaming mouse"],
        ["low latency wireless optical gaming mouse"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    converted = ov.convert_model(model, example_input=dict(example))
    converted.reshape({port: ov.PartialShape([-1, -1]) for port in converted.inputs})
    (output_dir / "openvino").mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, output_xml, compress_to_fp16=True)
    manifest = {
        "base_repository": config["repository"],
        "base_revision": config["revision"],
        "dataset": "Amazon ESCI US Task-1 reduced, project train queries only",
        "split_manifest_hash": _split_manifest_hash(),
        "training_judgments": len(order),
        "objective": "pointwise BCE on DCG-normalized soft gains E=1,S=3/7,C=1/7,I=0",
        "epochs": int(settings["epochs"]),
        "seed": seed,
        "max_length": max_length,
        "training_device": str(device),
        "license": "Apache-2.0",
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir
