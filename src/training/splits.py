from __future__ import annotations

"""Participant-level split helpers for paper-style evaluation."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SplitIndexSummary:
    participant_count: int
    sample_count: int
    positive_count: int
    positive_rate: float


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

    # Splitting at participant level prevents leakage across train/val/test.
    train_ids = participants[:train_count]
    val_ids = participants[train_count : train_count + val_count]
    test_ids = participants[train_count + val_count : expected_total]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def summarize_participant_slices(
    index_csv_path: str | Path,
    splits: dict[str, list[str]],
) -> dict[str, SplitIndexSummary]:
    index_frame = pd.read_csv(
        index_csv_path,
        usecols=["participant_id", "label"],
        dtype={"participant_id": "string", "label": "int8"},
    )

    summaries: dict[str, SplitIndexSummary] = {}
    for split_name, participant_ids in splits.items():
        participant_set = {str(participant_id) for participant_id in participant_ids}
        split_frame = index_frame[index_frame["participant_id"].isin(participant_set)]
        sample_count = int(len(split_frame))
        positive_count = int(split_frame["label"].sum()) if sample_count else 0
        positive_rate = (
            float(positive_count / sample_count) if sample_count else 0.0
        )
        summaries[split_name] = SplitIndexSummary(
            participant_count=len(participant_set),
            sample_count=sample_count,
            positive_count=positive_count,
            positive_rate=positive_rate,
        )

    return summaries


def select_subset_sample_ids(
    index_csv_path: str | Path,
    participant_ids: list[str],
    limit_samples: int | None,
    *,
    seed: int = 0,
    strategy: str = "balanced",
) -> list[int] | None:
    if limit_samples is None:
        return None

    if strategy not in {"head", "random", "balanced"}:
        raise ValueError(
            f"Unsupported subset strategy: {strategy!r}. "
            "Expected one of: head, random, balanced."
        )

    index_frame = pd.read_csv(
        index_csv_path,
        usecols=["sample_id", "participant_id", "label"],
        dtype={"sample_id": "int64", "participant_id": "string", "label": "int8"},
    )
    participant_set = {str(participant_id) for participant_id in participant_ids}
    split_frame = index_frame[index_frame["participant_id"].isin(participant_set)]

    if len(split_frame) <= limit_samples:
        return split_frame["sample_id"].astype(int).tolist()

    if strategy == "head":
        subset = split_frame.head(limit_samples)
        return subset["sample_id"].astype(int).tolist()
    elif strategy == "random":
        subset = split_frame.sample(n=limit_samples, random_state=seed, replace=False)
        return subset["sample_id"].astype(int).tolist()
    else:
        # Balanced subsets make quick CPU sanity runs more informative.
        positive_frame = split_frame[split_frame["label"] == 1]
        negative_frame = split_frame[split_frame["label"] == 0]

        if len(positive_frame) == 0 or len(negative_frame) == 0:
            subset = split_frame.sample(
                n=limit_samples, random_state=seed, replace=False
            )
            return subset["sample_id"].astype(int).tolist()
        else:
            positive_target = min(len(positive_frame), max(1, limit_samples // 2))
            negative_target = min(len(negative_frame), limit_samples - positive_target)

            remaining = limit_samples - positive_target - negative_target
            if remaining > 0:
                extra_positive = min(remaining, len(positive_frame) - positive_target)
                positive_target += extra_positive
                remaining -= extra_positive

            if remaining > 0:
                extra_negative = min(remaining, len(negative_frame) - negative_target)
                negative_target += extra_negative

            positive_subset = positive_frame.sample(
                n=positive_target, random_state=seed, replace=False
            )
            negative_subset = negative_frame.sample(
                n=negative_target, random_state=seed + 1, replace=False
            )
            positive_ids = positive_subset["sample_id"].astype(int).tolist()
            negative_ids = negative_subset["sample_id"].astype(int).tolist()

            ordered_ids: list[int] = []
            max_length = max(len(positive_ids), len(negative_ids))
            for idx in range(max_length):
                if idx < len(positive_ids):
                    ordered_ids.append(positive_ids[idx])
                if idx < len(negative_ids):
                    ordered_ids.append(negative_ids[idx])
            return ordered_ids[:limit_samples]
