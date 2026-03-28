from __future__ import annotations

"""Training entrypoint για το baseline reproduction του paper.

Το script αυτό ενώνει όλα τα προηγούμενα κομμάτια:
- παίρνει το generated window index
- φτιάχνει participant-independent splits
- φορτώνει Dataset / DataLoader
- εκπαιδεύει το baseline model
- μετρά validation / test metrics
- γράφει experiment logs και checkpoints
"""

import argparse
import sys
from dataclasses import asdict
from statistics import mean
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Allows `python scripts/...` without packaging the repository first.
    sys.path.insert(0, str(REPO_ROOT))

from src.data import HDBDPaperWindowDataset, prefetch_subset_assets
from src.evaluation import BinaryPredictionMetrics, summarize_binary_predictions
from src.models import PaperTakeoverBaselineModel
from src.training import (
    ExperimentRecorder,
    SplitIndexSummary,
    default_experiment_root,
    make_participant_split,
    select_subset_sample_ids,
    summarize_participant_slices,
)


def parse_args() -> argparse.Namespace:
    """Ορίζει όλα τα training, debug και logging arguments του run."""
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
    parser.add_argument(
        "--num-split-groups",
        type=int,
        default=1,
        help="How many shuffled participant-independent partition groups to run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional descriptive name for the experiment directory.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=default_experiment_root(REPO_ROOT),
        help="Directory where experiment logs and checkpoints are saved.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local cache directory for extracted inner HDBD archives.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Values > 0 help most when assets are prefetched to local files.",
    )
    parser.add_argument(
        "--prefetch-subset-assets",
        action="store_true",
        help="Extract the exact limited subset assets needed by the run into cache_dir/prefetched_assets before training.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_roc_auc", "val_loss"],
        default="val_roc_auc",
        help="Validation metric used to choose the best checkpoint.",
    )
    return parser.parse_args()


def make_dataset(
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    participant_ids: list[str],
    sample_ids: list[int] | None,
    limit_samples: int | None,
    cache_dir: Path | None,
) -> HDBDPaperWindowDataset:
    """Φτιάχνει ένα dataset object για το συγκεκριμένο split/subset."""
    return HDBDPaperWindowDataset(
        index_csv_path=index_path,
        bundle_path=bundle_path,
        heatmap_variant=heatmap_variant,
        participant_ids=participant_ids,
        sample_ids=sample_ids,
        limit_samples=limit_samples,
        cache_dir=cache_dir,
    )


def make_loader(
    dataset: HDBDPaperWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Φτιάχνει DataLoader με τις ελάχιστες ρυθμίσεις που χρειαζόμαστε τώρα."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def maybe_prefetch_active_assets(
    *,
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    cache_dir: Path | None,
    sample_id_lists: list[list[int] | None],
) -> None:
    combined_sample_ids: list[int] = []
    seen_sample_ids: set[int] = set()
    for sample_ids in sample_id_lists:
        if sample_ids is None:
            continue
        for sample_id in sample_ids:
            normalized_id = int(sample_id)
            if normalized_id in seen_sample_ids:
                continue
            seen_sample_ids.add(normalized_id)
            combined_sample_ids.append(normalized_id)

    if not combined_sample_ids:
        print(
            "prefetch_subset_assets=skipped "
            "(no limited subset sample ids were available)"
        )
        return

    prefetch_summary = prefetch_subset_assets(
        index_csv_path=index_path,
        bundle_path=bundle_path,
        heatmap_variant=heatmap_variant,
        sample_ids=combined_sample_ids,
        cache_dir=cache_dir,
    )
    print(f"prefetch_subset_assets={prefetch_summary}")


def run_epoch(
    model: PaperTakeoverBaselineModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    device: torch.device,
) -> BinaryPredictionMetrics:
    """Τρέχει ένα epoch είτε σε training είτε σε evaluation mode.

    Αν δοθεί optimizer:
    - το μοντέλο μπαίνει σε train mode
    - γίνεται backward + optimizer step

    Αν δεν δοθεί optimizer:
    - το μοντέλο μπαίνει σε eval mode
    - γίνεται μόνο forward και metric collection
    """
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

        # Το batch έχει ήδη το format που ορίζει ο HDBDPaperWindowDataset:
        # scene_gaze, signals, hmi, label.
        scene_gaze = batch["scene_gaze"].to(device)
        signals = batch["signals"].to(device)
        hmi = batch["hmi"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            # Το model επιστρέφει logits. Οι probabilities βγαίνουν μόνο για
            # logging/metrics με sigmoid πάνω στα logits.
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

    # Στο τέλος κάθε epoch κρατάμε όχι μόνο loss, αλλά και τα βασικά binary
    # metrics που μας ενδιαφέρουν για το paper baseline.
    average_loss = total_loss / total_batches if total_batches else 0.0
    return summarize_binary_predictions(
        loss=average_loss,
        batch_count=total_batches,
        labels=labels_seen,
        probabilities=probabilities_seen,
    )


def format_epoch_metrics(prefix: str, metrics: BinaryPredictionMetrics) -> str:
    """Μετατρέπει τα metrics ενός epoch σε μία compact γραμμή για terminal output."""
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
    """Μετράει το πραγματικό subset που φορτώθηκε σε ένα split."""
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
    """Επιστρέφει σταθερό text format για split statistics."""
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
    """Προειδοποιεί όταν ένα subset είναι πολύ μικρό για χρήσιμα metrics."""
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


def format_aggregate_metrics(
    prefix: str,
    metrics_list: list[BinaryPredictionMetrics],
) -> str:
    """Συνοψίζει metrics από πολλά split groups σε μία γραμμή."""
    if not metrics_list:
        return f"{prefix}_aggregate=n/a"

    roc_auc_values = [metrics.roc_auc for metrics in metrics_list if metrics.roc_auc is not None]
    roc_auc_text = "n/a" if not roc_auc_values else f"{mean(roc_auc_values):.6f}"
    return (
        f"{prefix}_aggregate: "
        f"splits={len(metrics_list)} "
        f"mean_loss={mean([metrics.loss for metrics in metrics_list]):.6f} "
        f"mean_examples={mean([metrics.example_count for metrics in metrics_list]):.2f} "
        f"mean_positive_rate={mean([metrics.positive_rate for metrics in metrics_list]):.6f} "
        f"mean_probability={mean([metrics.mean_probability for metrics in metrics_list]):.6f} "
        f"mean_roc_auc={roc_auc_text}"
    )


def namespace_to_serializable_dict(args: argparse.Namespace) -> dict[str, object]:
    """Μετατρέπει το argparse namespace σε JSON-friendly dict για logging."""
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def checkpoint_metric_payload(
    metric_name: str,
    val_metrics: BinaryPredictionMetrics,
) -> tuple[float, str]:
    """Επιστρέφει το score με το οποίο συγκρίνουμε checkpoints.

    Επειδή σε μικρά debug runs το ROC AUC μπορεί να είναι undefined, το helper
    αυτό δίνει και ασφαλές fallback σε validation loss.
    """
    if metric_name == "val_roc_auc":
        # Small debug subsets may not have both classes, so we fall back to
        # validation loss instead of skipping checkpoint selection.
        if val_metrics.roc_auc is not None:
            return val_metrics.roc_auc, "val_roc_auc"
        return -val_metrics.loss, "neg_val_loss_fallback"
    if metric_name == "val_loss":
        return -val_metrics.loss, "neg_val_loss"
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def run_single_split(
    args: argparse.Namespace,
    *,
    split_seed: int,
    device: torch.device,
    recorder: ExperimentRecorder | None,
) -> dict[str, object]:
    """Τρέχει ολόκληρο το pipeline για ένα participant split seed.

    Δηλαδή:
    - δημιουργία train/val/test participant split
    - επιλογή subset ids αν έχουμε debug limits
    - dataset / dataloader creation
    - training / validation / test
    - logging και checkpoints
    """
    splits = make_participant_split(args.index, seed=split_seed)
    print(f"split_seed={split_seed}")
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
        seed=split_seed,
        strategy=args.subset_strategy,
    )
    val_sample_ids = select_subset_sample_ids(
        args.index,
        splits["val"],
        args.limit_val_samples,
        seed=split_seed + 1,
        strategy=args.subset_strategy,
    )
    test_sample_ids = None
    if args.evaluate_test:
        test_sample_ids = select_subset_sample_ids(
            args.index,
            splits["test"],
            args.limit_test_samples,
            seed=split_seed + 2,
            strategy=args.subset_strategy,
        )

    if args.prefetch_subset_assets:
        maybe_prefetch_active_assets(
            index_path=args.index,
            bundle_path=args.bundle,
            heatmap_variant=args.heatmap_variant,
            cache_dir=args.cache_dir,
            sample_id_lists=[train_sample_ids, val_sample_ids, test_sample_ids],
        )

    train_dataset = make_dataset(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["train"],
        sample_ids=train_sample_ids,
        limit_samples=None,
        cache_dir=args.cache_dir,
    )
    val_dataset = make_dataset(
        index_path=args.index,
        bundle_path=args.bundle,
        heatmap_variant=args.heatmap_variant,
        participant_ids=splits["val"],
        sample_ids=val_sample_ids,
        limit_samples=None,
        cache_dir=args.cache_dir,
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
            cache_dir=args.cache_dir,
        )

    active_datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }
    if test_dataset is not None:
        active_datasets["test"] = test_dataset

    loaded_summaries: dict[str, SplitIndexSummary] = {}
    for split_name, dataset in active_datasets.items():
        loaded_summary = summarize_loaded_dataset(dataset)
        loaded_summaries[split_name] = loaded_summary
        print(
            format_split_summary(
                split_name, loaded_summary, scope="loaded_subset"
            )
        )
        maybe_warn_for_zero_positives(
            split_name, loaded_summary, scope="loaded_subset"
        )

    if recorder is not None:
        recorder.record_split_setup(
            split_seed=split_seed,
            participant_splits=splits,
            full_summaries=full_summaries,
            loaded_summaries=loaded_summaries,
        )

    if args.report_only:
        # Σε report-only mode σταματάμε εδώ, γιατί ο στόχος είναι μόνο να δούμε
        # πώς είναι τα splits και όχι να τρέξουμε training.
        return {
            "split_seed": split_seed,
            "train_metrics": None,
            "val_metrics": None,
            "test_metrics": None,
        }

    pin_memory = device.type == "cuda"
    print(f"loader_num_workers={args.num_workers}")
    print(f"loader_pin_memory={pin_memory}")

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = make_loader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = PaperTakeoverBaselineModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"device={device}")
    train_metrics: BinaryPredictionMetrics | None = None
    val_metrics: BinaryPredictionMetrics | None = None
    best_checkpoint_score = float("-inf")
    best_checkpoint_info: dict[str, object] = {}
    for epoch in range(args.epochs):
        # Κάθε epoch έχει ένα train pass και ένα validation pass με τον ίδιο
        # ακριβώς metric collection μηχανισμό.
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

        if recorder is not None:
            recorder.record_epoch(
                split_seed=split_seed,
                epoch=epoch + 1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            checkpoint_score, checkpoint_metric_used = checkpoint_metric_payload(
                args.checkpoint_metric,
                val_metrics,
            )
            last_checkpoint_path = recorder.save_checkpoint(
                split_seed=split_seed,
                epoch=epoch + 1,
                checkpoint_tag="last",
                checkpoint_metric_name=checkpoint_metric_used,
                checkpoint_metric_value=checkpoint_score,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            print(f"last_checkpoint={last_checkpoint_path}")
            if checkpoint_score > best_checkpoint_score:
                # Κρατάμε ξεχωριστό "best" checkpoint για το metric που έχει
                # οριστεί από τα arguments του run.
                best_checkpoint_score = checkpoint_score
                best_checkpoint_path = recorder.save_checkpoint(
                    split_seed=split_seed,
                    epoch=epoch + 1,
                    checkpoint_tag="best",
                    checkpoint_metric_name=checkpoint_metric_used,
                    checkpoint_metric_value=checkpoint_score,
                    model=model,
                    optimizer=optimizer,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
                best_checkpoint_info = {
                    "path": str(best_checkpoint_path),
                    "epoch": epoch + 1,
                    "metric_name": checkpoint_metric_used,
                    "metric_value": checkpoint_score,
                }
                print(f"best_checkpoint={best_checkpoint_path}")

    test_metrics = None
    if test_loader is not None:
        # Το test τρέχει μόνο στο τέλος, ώστε να λειτουργεί ως πραγματικό
        # held-out evaluation και όχι ως μέρος του training loop.
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            max_batches=args.max_test_batches,
            device=device,
        )
        print(f"test_metrics={asdict(test_metrics)}")

    if recorder is not None:
        recorder.record_split_result(
            split_seed=split_seed,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_checkpoint=best_checkpoint_info,
        )

    return {
        "split_seed": split_seed,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": best_checkpoint_info,
    }


def main() -> None:
    """Κύριο entrypoint του training script.

    Εδώ:
    - δημιουργείται το experiment recorder
    - τρέχουν 1 ή περισσότερα split groups
    - γράφονται aggregate summaries στο τέλος
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recorder = ExperimentRecorder(
        experiment_root=args.experiment_root,
        run_name=args.run_name,
        args=namespace_to_serializable_dict(args),
        report_only=args.report_only,
    )
    print(f"run_dir={recorder.run_dir}")

    split_results: list[dict[str, object]] = []
    for split_offset in range(args.num_split_groups):
        current_split_seed = args.split_seed + split_offset
        if args.num_split_groups > 1:
            print(
                f"===== split_group={split_offset + 1}/{args.num_split_groups} ====="
            )
        # Repeating the run over shuffled participant groups makes evaluation
        # closer to the multi-partition protocol described in the paper.
        split_results.append(
            run_single_split(
                args,
                split_seed=current_split_seed,
                device=device,
                recorder=recorder,
            )
        )

    aggregate: dict[str, object] = {
        "num_split_groups": args.num_split_groups,
    }
    if args.report_only:
        # Σε report-only mode σώζουμε μόνο summary των splits και τελειώνουμε.
        summary_path = recorder.finalize(aggregate=aggregate)
        print(f"summary_path={summary_path}")
        return

    val_metric_list: list[BinaryPredictionMetrics] = [
        result["val_metrics"]
        for result in split_results
        if result["val_metrics"] is not None
    ]
    test_metric_list: list[BinaryPredictionMetrics] = [
        result["test_metrics"]
        for result in split_results
        if result["test_metrics"] is not None
    ]
    aggregate["val"] = {
        "text": format_aggregate_metrics("val", val_metric_list),
        "metrics": [asdict(metrics) for metrics in val_metric_list],
    }
    print(aggregate["val"]["text"])
    if test_metric_list:
        aggregate["test"] = {
            "text": format_aggregate_metrics("test", test_metric_list),
            "metrics": [asdict(metrics) for metrics in test_metric_list],
        }
        print(aggregate["test"]["text"])

    if args.num_split_groups <= 1:
        aggregate["single_split"] = True

    summary_path = recorder.finalize(aggregate=aggregate)
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
