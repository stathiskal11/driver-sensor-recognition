from __future__ import annotations

import csv
import random
from pathlib import Path


def make_participant_split(
    index_csv_path: str | Path,
    seed: int = 0,
    train_count: int = 20,
    val_count: int = 4,
    test_count: int = 4,
) -> dict[str, list[str]]:
    participant_ids: set[str] = set()
    with Path(index_csv_path).open("r", encoding="utf-8", newline="") as index_file:
        reader = csv.DictReader(index_file)
        for row in reader:
            participant_ids.add(row["participant_id"])

    participants = sorted(participant_ids)
    expected_total = train_count + val_count + test_count
    if len(participants) < expected_total:
        raise ValueError(
            f"Not enough participants for split: found {len(participants)}, "
            f"need at least {expected_total}."
        )

    rng = random.Random(seed)
    rng.shuffle(participants)

    train_ids = participants[:train_count]
    val_ids = participants[train_count : train_count + val_count]
    test_ids = participants[train_count + val_count : expected_total]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
