from __future__ import annotations

import json
import logging

from retail_search.evaluation.benchmark import run_full_benchmark

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_full_benchmark()
    print(json.dumps(result["quality_gate"], indent=2))
    raise SystemExit(0 if result["quality_gate"]["passed"] else 1)
