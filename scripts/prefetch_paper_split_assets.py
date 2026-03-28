from __future__ import annotations

"""Prefetch the exact assets needed by limited paper-baseline split subsets."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import prefetch_subset_assets
from src.training import make_participant_split, select_subset_sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch the exact assets needed by one or more limited paper split groups."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to hdbd.tar.gz. If omitted, the loader searches common local paths.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "paper_window_index.csv",
        help="Path to the generated paper window index CSV.",
    )
    parser.add_argument(
        "--heatmap-variant",
        choices=["sigma16", "sigma32", "sigma64", "laplace"],
        default="sigma64",
        help="Which precomputed heatmap archive to prefetch.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Base participant split seed.",
    )
    parser.add_argument(
        "--num-split-groups",
        type=int,
        default=1,
        help="How many consecutive split seeds to prefetch, starting at --split-seed.",
    )
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Required train subset size used by the later training run.",
    )
    parser.add_argument(
        "--limit-val-samples",
        type=int,
        default=None,
        help="Required validation subset size used by the later training run.",
    )
    parser.add_argument(
        "--limit-test-samples",
        type=int,
        default=None,
        help="Required test subset size used by the later training run.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also prefetch the held-out test subset assets.",
    )
    parser.add_argument(
        "--subset-strategy",
        choices=["head", "random", "balanced"],
        default="random",
        help="Subset strategy used by the later training run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local cache directory for extracted inner HDBD archives and prefetched assets.",
    )
    return parser.parse_args()


def require_limited_subset(sample_ids: list[int] | None, split_name: str) -> list[int]:
    if sample_ids is None:
        raise ValueError(
            f"Prefetch for split={split_name!r} needs an explicit limit_*_samples value. "
            "Full-split prefetch would be too large for the intended fast-cache workflow."
        )
    return sample_ids


def main() -> None:
    args = parse_args()

    combined_sample_ids: list[int] = []
    seen_sample_ids: set[int] = set()

    for split_offset in range(args.num_split_groups):
        current_seed = args.split_seed + split_offset
        splits = make_participant_split(args.index, seed=current_seed)

        train_sample_ids = require_limited_subset(
            select_subset_sample_ids(
                args.index,
                splits["train"],
                args.limit_train_samples,
                seed=current_seed,
                strategy=args.subset_strategy,
            ),
            "train",
        )
        val_sample_ids = require_limited_subset(
            select_subset_sample_ids(
                args.index,
                splits["val"],
                args.limit_val_samples,
                seed=current_seed + 1,
                strategy=args.subset_strategy,
            ),
            "val",
        )
        test_sample_ids: list[int] = []
        if args.evaluate_test:
            test_sample_ids = require_limited_subset(
                select_subset_sample_ids(
                    args.index,
                    splits["test"],
                    args.limit_test_samples,
                    seed=current_seed + 2,
                    strategy=args.subset_strategy,
                ),
                "test",
            )

        split_sample_ids = train_sample_ids + val_sample_ids + test_sample_ids
        for sample_id in split_sample_ids:
            normalized_id = int(sample_id)
            if normalized_id in seen_sample_ids:
                continue
            seen_sample_ids.add(normalized_id)
            combined_sample_ids.append(normalized_id)

        print(
            f"split_seed={current_seed} "
            f"train_samples={len(train_sample_ids)} "
            f"val_samples={len(val_sample_ids)} "
            f"test_samples={len(test_sample_ids)} "
            f"cumulative_unique_samples={len(combined_sample_ids)}"
        )

    summary = prefetch_subset_assets(
        index_csv_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        sample_ids=combined_sample_ids,
        cache_dir=args.cache_dir,
    )
    print(f"prefetch_summary={summary}")


if __name__ == "__main__":
    main()
