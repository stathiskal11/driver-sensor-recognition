from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import HDBDPaperWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check the paper baseline dataset loader."
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
        help="Which precomputed heatmap archive to use.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=8,
        help="Limit the dataset size for a quick sanity check.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the DataLoader sanity check.",
    )
    return parser.parse_args()


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def main() -> None:
    args = parse_args()
    dataset = HDBDPaperWindowDataset(
        index_csv_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        limit_samples=args.limit_samples,
    )

    print(f"dataset_len={len(dataset)}")
    sample = dataset[0]
    print(f"sample_id={sample['sample_id']}")
    print(f"participant_id={sample['participant_id']}")
    print(f"csv_member={sample['csv_member']}")
    print_tensor_info("scene_gaze", sample["scene_gaze"])
    print_tensor_info("signals", sample["signals"])
    print_tensor_info("hmi", sample["hmi"])
    print_tensor_info("label", sample["label"])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print_tensor_info("batch.scene_gaze", batch["scene_gaze"])
    print_tensor_info("batch.signals", batch["signals"])
    print_tensor_info("batch.hmi", batch["hmi"])
    print_tensor_info("batch.label", batch["label"])


if __name__ == "__main__":
    main()
