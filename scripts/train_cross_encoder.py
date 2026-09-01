from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from retail_search.ranking.fine_tune_cross_encoder import train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(Path("configs/ranking.yaml").read_text(encoding="utf-8"))
    output = train_cross_encoder(config["cross_encoder"], force=args.force)
    print(output)


if __name__ == "__main__":
    main()
