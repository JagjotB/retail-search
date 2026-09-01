from __future__ import annotations

import argparse
import json

from retail_search.data.ingest import download_official_dataset, prepare_amazon_esci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not args.skip_download:
        print(json.dumps(download_official_dataset(), indent=2))
    print(json.dumps(prepare_amazon_esci(), indent=2))


if __name__ == "__main__":
    main()
