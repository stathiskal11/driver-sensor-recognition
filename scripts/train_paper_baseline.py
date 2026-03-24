from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import HDBDPaperWindowDataset
from src.models import PaperTakeoverBaselineModel
from src.training import make_participant_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or sanity-check the paper baseline model."
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
        "--split-seed",
        type=int,
        default=0,
        help="Participant split seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=1,
        help="Optional cap on train batches for sanity checks.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=1,
        help="Optional cap on validation batches for sanity checks.",
    )
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=8,
        help="Optional cap on train samples for quick runs.",
    )
    parser.add_argument(
        "--limit-val-samples",
        type=int,
        default=4,
        help="Optional cap on validation samples for quick runs.",
    )
    return parser.parse_args()


def make_loader(
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    participant_ids: list[str],
    batch_size: int,
    limit_samples: int | None,
) -> DataLoader:
    dataset = HDBDPaperWindowDataset(
        index_csv_path=index_path,
        bundle_path=bundle_path,
        heatmap_variant=heatmap_variant,
        participant_ids=participant_ids,
        limit_samples=limit_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def run_epoch(
    model: PaperTakeoverBaselineModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    device: torch.device,
) -> tuple[float, int]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        scene_gaze = batch["scene_gaze"].to(device)
        signals = batch["signals"].to(device)
        hmi = batch["hmi"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(scene_gaze=scene_gaze, signals=signals, hmi=hmi)
            loss = criterion(logits, labels)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / total_batches if total_batches else 0.0
    return average_loss, total_batches


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = make_participant_split(args.index, seed=args.split_seed)
    print(f"split_seed={args.split_seed}")
    print(f"train_participants={splits['train']}")
    print(f"val_participants={splits['val']}")
    print(f"test_participants={splits['test']}")

    train_loader = make_loader(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["train"],
        batch_size=args.batch_size,
        limit_samples=args.limit_train_samples,
    )
    val_loader = make_loader(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["val"],
        batch_size=args.batch_size,
        limit_samples=args.limit_val_samples,
    )

    model = PaperTakeoverBaselineModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"device={device}")
    for epoch in range(args.epochs):
        train_loss, train_batches = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            device=device,
        )
        val_loss, val_batches = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            max_batches=args.max_val_batches,
            device=device,
        )
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} train_batches={train_batches} "
            f"val_loss={val_loss:.6f} val_batches={val_batches}"
        )


if __name__ == "__main__":
    main()
