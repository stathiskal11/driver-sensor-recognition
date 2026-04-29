from __future__ import annotations

import argparse
import json
import random
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
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
PAPER_REFERENCE_SETTINGS = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_split_groups": 5,
    "heatmap_variant": "sigma64",
}
PATH_ARG_NAMES = [
    "bundle",
    "index",
    "experiment_root",
    "cache_dir",
    "config",
]


def make_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. CLI flags override config values.",
    )
    return parser


def load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Config file must contain a JSON object, got {type(payload).__name__}."
        )
    return payload


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    config_defaults: dict[str, Any],
) -> None:
    if not config_defaults:
        return
    valid_keys = {action.dest for action in parser._actions}
    unexpected_keys = sorted(set(config_defaults) - valid_keys)
    if unexpected_keys:
        parser.error(
            "Unsupported keys in --config: " + ", ".join(unexpected_keys)
        )
    parser.set_defaults(**config_defaults)


def normalize_path_namespace(args: Namespace) -> None:
    for name in PATH_ARG_NAMES:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None or isinstance(value, Path):
            continue
        setattr(args, name, Path(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(argv) if argv is not None else sys.argv[1:]
    config_parser = make_config_parser()
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_config_defaults(config_args.config)
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Train or sanity-check the paper baseline model.",
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
        "--prefetch-active-splits",
        action="store_true",
        help="Prefetch all assets for the active train/val/test participant splits into local files before training.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_roc_auc", "val_loss"],
        default="val_roc_auc",
        help="Validation metric used to choose the best checkpoint.",
    )
    parser.add_argument(
        "--test-checkpoint",
        choices=["last", "best"],
        default="best",
        help="Which checkpoint to evaluate on the held-out test split.",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=0,
        help="Global random seed for reproducible training and loader shuffling.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    apply_config_defaults(parser, config_defaults)
    args = parser.parse_args(argv)
    normalize_path_namespace(args)
    return args


def make_dataset(
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    participant_ids: list[str],
    sample_ids: list[int] | None,
    limit_samples: int | None,
    cache_dir: Path | None,
) -> HDBDPaperWindowDataset:
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
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def maybe_prefetch_active_assets(
    *,
    index_path: Path,
    bundle_path: Path | None,
    heatmap_variant: str,
    cache_dir: Path | None,
    sample_id_lists: list[list[int] | None],
    participant_id_lists: list[list[str] | None],
    allow_full_split_prefetch: bool,
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
    if combined_sample_ids:
        prefetch_summary = prefetch_subset_assets(
            index_csv_path=index_path,
            bundle_path=bundle_path,
            heatmap_variant=heatmap_variant,
            sample_ids=combined_sample_ids,
            cache_dir=cache_dir,
        )
        print(f"prefetch_subset_assets={prefetch_summary}")
        return
    if not allow_full_split_prefetch:
        print(
            "prefetch_subset_assets=skipped "
            "(no limited subset sample ids were available)"
        )
        return
    combined_participant_ids: list[str] = []
    seen_participant_ids: set[str] = set()
    for participant_ids in participant_id_lists:
        if participant_ids is None:
            continue
        for participant_id in participant_ids:
            normalized_id = str(participant_id)
            if normalized_id in seen_participant_ids:
                continue
            seen_participant_ids.add(normalized_id)
            combined_participant_ids.append(normalized_id)
    if not combined_participant_ids:
        print(
            "prefetch_active_splits=skipped "
            "(no participant ids were available for the active splits)"
        )
        return
    prefetch_summary = prefetch_subset_assets(
        index_csv_path=index_path,
        bundle_path=bundle_path,
        heatmap_variant=heatmap_variant,
        participant_ids=combined_participant_ids,
        cache_dir=cache_dir,
    )
    print(f"prefetch_active_splits={prefetch_summary}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly, but it is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_train_loader_generator(global_seed: int, split_seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(global_seed + split_seed)
    return generator


def collect_run_warnings(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    sample_limits = [
        args.limit_train_samples,
        args.limit_val_samples,
        args.limit_test_samples if args.evaluate_test else None,
    ]
    batch_caps = [
        args.max_train_batches,
        args.max_val_batches,
        args.max_test_batches if args.evaluate_test else None,
    ]
    has_sample_limits = any(value is not None for value in sample_limits)
    has_batch_caps = any(value is not None for value in batch_caps)
    if args.report_only:
        warnings.append(
            "report-only mode skips model training and does not produce comparable AUC metrics."
        )
    if has_sample_limits:
        warnings.append(
            "sample limits are active, so the run only uses a subset of each split."
        )
    if has_sample_limits and args.subset_strategy == "balanced":
        warnings.append(
            "balanced subset sampling shifts the class prior toward 50/50 and should only be used for debugging."
        )
    if not args.report_only and has_batch_caps:
        warnings.append(
            "batch caps are active, so each epoch only sees part of the selected dataset."
        )
    if args.num_split_groups != PAPER_REFERENCE_SETTINGS["num_split_groups"]:
        warnings.append(
            "paper-faithful evaluation expects five shuffled split groups."
        )
    if not args.report_only and args.epochs != PAPER_REFERENCE_SETTINGS["epochs"]:
        warnings.append(
            "paper-faithful training expects 20 epochs."
        )
    if not args.report_only and args.batch_size != PAPER_REFERENCE_SETTINGS["batch_size"]:
        warnings.append(
            "paper-faithful training expects batch size 32."
        )
    if (
        not args.report_only
        and args.learning_rate != PAPER_REFERENCE_SETTINGS["learning_rate"]
    ):
        warnings.append(
            "paper-faithful training expects learning rate 0.001."
        )
    if args.heatmap_variant != PAPER_REFERENCE_SETTINGS["heatmap_variant"]:
        warnings.append(
            "paper-faithful preprocessing currently assumes sigma64 Gaussian heatmaps."
        )
    if not args.report_only and not args.evaluate_test:
        warnings.append(
            "held-out test evaluation is disabled for this run."
        )
    if args.evaluate_test and args.test_checkpoint != "best":
        warnings.append(
            "test evaluation is using the last checkpoint instead of the best validation checkpoint."
        )
    return warnings


def print_run_warnings(args: argparse.Namespace) -> None:
    for warning_text in collect_run_warnings(args):
        print(f"warning={warning_text}")


def load_checkpoint_into_model(
    model: PaperTakeoverBaselineModel,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


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


def format_aggregate_metrics(
    prefix: str,
    metrics_list: list[BinaryPredictionMetrics],
) -> str:
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
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def checkpoint_metric_payload(
    metric_name: str,
    val_metrics: BinaryPredictionMetrics,
) -> tuple[float, str]:
    if metric_name == "val_roc_auc":
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
    set_global_seed(args.global_seed + split_seed, device)
    splits = make_participant_split(args.index, seed=split_seed)
    print(f"split_seed={split_seed}")
    print(f"split_run_seed={args.global_seed + split_seed}")
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
            participant_id_lists=[
                splits["train"],
                splits["val"],
                splits["test"] if args.evaluate_test else None,
            ],
            allow_full_split_prefetch=args.prefetch_active_splits,
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
        return {
            "split_seed": split_seed,
            "train_metrics": None,
            "val_metrics": None,
            "test_metrics": None,
            "best_checkpoint": {},
            "test_evaluation_checkpoint": {},
        }
    pin_memory = device.type == "cuda"
    print(f"loader_num_workers={args.num_workers}")
    print(f"loader_pin_memory={pin_memory}")
    train_loader_generator = make_train_loader_generator(
        args.global_seed,
        split_seed,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=train_loader_generator,
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
    last_checkpoint_info: dict[str, object] = {}
    best_checkpoint_info: dict[str, object] = {}
    test_evaluation_checkpoint: dict[str, object] = {}
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
            last_checkpoint_info = {
                "path": str(last_checkpoint_path),
                "epoch": epoch + 1,
                "metric_name": checkpoint_metric_used,
                "metric_value": checkpoint_score,
            }
            print(f"last_checkpoint={last_checkpoint_path}")
            if checkpoint_score > best_checkpoint_score:
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
        if args.test_checkpoint == "best":
            if best_checkpoint_info.get("path"):
                checkpoint_path = Path(str(best_checkpoint_info["path"]))
                load_checkpoint_into_model(model, checkpoint_path, device)
                test_evaluation_checkpoint = {
                    "selector": "best",
                    **best_checkpoint_info,
                }
                print(f"test_checkpoint={checkpoint_path}")
            else:
                print(
                    "warning=best checkpoint was requested for test evaluation, "
                    "but no best checkpoint was recorded. Using the current model state."
                )
                test_evaluation_checkpoint = {
                    "selector": "current_model_fallback",
                }
        else:
            test_evaluation_checkpoint = {
                "selector": "last",
                **last_checkpoint_info,
            }
            if last_checkpoint_info.get("path"):
                print(f"test_checkpoint={last_checkpoint_info['path']}")
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
            test_evaluation_checkpoint=test_evaluation_checkpoint,
        )
    return {
        "split_seed": split_seed,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": best_checkpoint_info,
        "test_evaluation_checkpoint": test_evaluation_checkpoint,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_global_seed(args.global_seed, device)
    recorder = ExperimentRecorder(
        experiment_root=args.experiment_root,
        run_name=args.run_name,
        args=namespace_to_serializable_dict(args),
        report_only=args.report_only,
    )
    print(f"run_dir={recorder.run_dir}")
    print(f"global_seed={args.global_seed}")
    print_run_warnings(args)
    split_results: list[dict[str, object]] = []
    for split_offset in range(args.num_split_groups):
        current_split_seed = args.split_seed + split_offset
        if args.num_split_groups > 1:
            print(
                f"===== split_group={split_offset + 1}/{args.num_split_groups} ====="
            )
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
