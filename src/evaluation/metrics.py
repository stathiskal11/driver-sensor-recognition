from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class BinaryPredictionMetrics:
    loss: float
    batch_count: int
    example_count: int
    positive_count: int
    positive_rate: float
    mean_probability: float
    roc_auc: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def summarize_binary_predictions(
    *,
    loss: float,
    batch_count: int,
    labels: list[float],
    probabilities: list[float],
) -> BinaryPredictionMetrics:
    label_array = np.asarray(labels, dtype=np.float32)
    probability_array = np.asarray(probabilities, dtype=np.float32)
    example_count = int(label_array.size)
    positive_count = int(label_array.sum()) if example_count else 0
    positive_rate = float(label_array.mean()) if example_count else 0.0
    mean_probability = (
        float(probability_array.mean()) if probability_array.size else 0.0
    )
    roc_auc: float | None = None
    if example_count and np.unique(label_array).size >= 2:
        roc_auc = float(roc_auc_score(label_array, probability_array))
    return BinaryPredictionMetrics(
        loss=loss,
        batch_count=batch_count,
        example_count=example_count,
        positive_count=positive_count,
        positive_rate=positive_rate,
        mean_probability=mean_probability,
        roc_auc=roc_auc,
    )
