from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import HDBDPaperWindowDataset
from src.evaluation import BinaryPredictionMetrics, summarize_binary_predictions
from src.models import PaperTakeoverBaselineModel
from src.training import (
    SplitIndexSummary,
    make_participant_split,
    select_subset_sample_ids,
    summarize_participant_slices,
)


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
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=1,
        help="Optional cap on test batches.",
    )
    parser.add_argument(
        "--limit-test-samples",
        type=int,
        default=4,
        help="Optional cap on test samples for quick runs.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Run a held-out test evaluation after the final epoch.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print split statistics and exit without training.",
    )
    parser.add_argument(
        "--subset-strategy",
        choices=["head", "random", "balanced"],
        default="balanced",
        help="How to choose limited debug subsets when limit_*_samples is set.",
    )
    return parser.parse_args()


def make_dataset(
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    participant_ids: list[str],
    sample_ids: list[int] | None,
    limit_samples: int | None,
) -> HDBDPaperWindowDataset:
    return HDBDPaperWindowDataset(
        index_csv_path=index_path,
        bundle_path=bundle_path,
        heatmap_variant=heatmap_variant,
        participant_ids=participant_ids,
        sample_ids=sample_ids,
        limit_samples=limit_samples,
    )


def make_loader(
    dataset: HDBDPaperWindowDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(
    model: PaperTakeoverBaselineModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    device: torch.device,
) -> BinaryPredictionMetrics:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0
    labels_seen: list[float] = []
    probabilities_seen: list[float] = []

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
            probabilities = torch.sigmoid(logits)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities_seen.extend(probabilities.detach().cpu().tolist())

    average_loss = total_loss / total_batches if total_batches else 0.0
    return summarize_binary_predictions(
        loss=average_loss,
        batch_count=total_batches,
        labels=labels_seen,
        probabilities=probabilities_seen,
    )


def format_epoch_metrics(prefix: str, metrics: BinaryPredictionMetrics) -> str:
    roc_auc_text = "n/a" if metrics.roc_auc is None else f"{metrics.roc_auc:.6f}"
    return (
        f"{prefix}_loss={metrics.loss:.6f} "
        f"{prefix}_batches={metrics.batch_count} "
        f"{prefix}_examples={metrics.example_count} "
        f"{prefix}_positive_rate={metrics.positive_rate:.6f} "
        f"{prefix}_mean_probability={metrics.mean_probability:.6f} "
        f"{prefix}_roc_auc={roc_auc_text}"
    )


def summarize_loaded_dataset(dataset: HDBDPaperWindowDataset) -> SplitIndexSummary:
    index_frame = dataset.index
    sample_count = int(len(index_frame))
    positive_count = int(index_frame["label"].sum()) if sample_count else 0
    positive_rate = float(positive_count / sample_count) if sample_count else 0.0
    participant_count = int(index_frame["participant_id"].nunique())
    return SplitIndexSummary(
        participant_count=participant_count,
        sample_count=sample_count,
        positive_count=positive_count,
        positive_rate=positive_rate,
    )


def format_split_summary(
    split_name: str,
    summary: SplitIndexSummary,
    *,
    scope: str,
) -> str:
    return (
        f"{scope}_{split_name}: "
        f"participants={summary.participant_count} "
        f"samples={summary.sample_count} "
        f"positives={summary.positive_count} "
        f"positive_rate={summary.positive_rate:.6f}"
    )


def maybe_warn_for_zero_positives(
    split_name: str,
    summary: SplitIndexSummary,
    *,
    scope: str,
) -> None:
    if summary.sample_count == 0:
        print(
            f"warning={scope}_{split_name} has zero samples. "
            "The run cannot provide meaningful metrics for this split."
        )
        return
    if summary.positive_count == 0:
        print(
            f"warning={scope}_{split_name} has zero positive samples. "
            "ROC AUC will be unavailable unless you increase the sample limit or use a larger run."
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = make_participant_split(args.index, seed=args.split_seed)
    print(f"split_seed={args.split_seed}")
    print(f"subset_strategy={args.subset_strategy}")
    print(f"train_participants={splits['train']}")
    print(f"val_participants={splits['val']}")
    print(f"test_participants={splits['test']}")

    full_summaries = summarize_participant_slices(args.index, splits)
    for split_name in ["train", "val", "test"]:
        print(
            format_split_summary(
                split_name, full_summaries[split_name], scope="full_split"
            )
        )

    train_sample_ids = select_subset_sample_ids(
        args.index,
        splits["train"],
        args.limit_train_samples,
        seed=args.split_seed,
        strategy=args.subset_strategy,
    )
    val_sample_ids = select_subset_sample_ids(
        args.index,
        splits["val"],
        args.limit_val_samples,
        seed=args.split_seed + 1,
        strategy=args.subset_strategy,
    )
    test_sample_ids = None
    if args.evaluate_test:
        test_sample_ids = select_subset_sample_ids(
            args.index,
            splits["test"],
            args.limit_test_samples,
            seed=args.split_seed + 2,
            strategy=args.subset_strategy,
        )

    train_dataset = make_dataset(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["train"],
        sample_ids=train_sample_ids,
        limit_samples=None,
    )
    val_dataset = make_dataset(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["val"],
        sample_ids=val_sample_ids,
        limit_samples=None,
    )
    test_dataset = None
    if args.evaluate_test:
        test_dataset = make_dataset(
            index_path=args.index,
            bundle_path=args.bundle,
            heatmap_variant=args.heatmap_variant,
            participant_ids=splits["test"],
            sample_ids=test_sample_ids,
            limit_samples=None,
        )

    active_datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }
    if test_dataset is not None:
        active_datasets["test"] = test_dataset

    for split_name, dataset in active_datasets.items():
        loaded_summary = summarize_loaded_dataset(dataset)
        print(
            format_split_summary(
                split_name, loaded_summary, scope="loaded_subset"
            )
        )
        maybe_warn_for_zero_positives(
            split_name, loaded_summary, scope="loaded_subset"
        )

    if args.report_only:
        return

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = make_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = PaperTakeoverBaselineModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"device={device}")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            device=device,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            max_batches=args.max_val_batches,
            device=device,
        )
        print(
            f"epoch={epoch + 1} "
            f"{format_epoch_metrics('train', train_metrics)} "
            f"{format_epoch_metrics('val', val_metrics)}"
        )

    if test_loader is not None:
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            max_batches=args.max_test_batches,
            device=device,
        )
        print(f"test_metrics={asdict(test_metrics)}")


if __name__ == "__main__":
    main()
