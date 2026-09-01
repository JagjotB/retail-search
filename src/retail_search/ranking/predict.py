from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_search.core import Reranker


class LightGBMReranker(Reranker):
    def __init__(self, booster: lgb.Booster, feature_names: Sequence[str]):
        self.booster = booster
        self.feature_names = list(feature_names)

    def score(self, features: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing reranker features: {sorted(missing)}")
        return np.asarray(
            self.booster.predict(features[self.feature_names], num_iteration=self.booster.best_iteration),
            dtype=np.float32,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path, feature_names: Sequence[str]) -> "LightGBMReranker":
        return cls(lgb.Booster(model_file=str(path)), feature_names)
