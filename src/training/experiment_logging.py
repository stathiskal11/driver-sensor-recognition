from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import torch


def default_experiment_root(repo_root: Path) -> Path:
    return repo_root / "experiments"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    slug = slug.strip("-._")
    return slug or "run"


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    return value


class ExperimentRecorder:
    def __init__(
        self,
        *,
        experiment_root: Path,
        run_name: str | None,
        args: dict[str, Any],
        report_only: bool,
    ) -> None:
        timestamp = datetime.now().astimezone()
        base_name = run_name or "paper-baseline"
        run_id = f"{timestamp:%Y%m%d_%H%M%S}_{slugify(base_name)}"
        self.run_id = run_id
        self.created_at = timestamp.isoformat(timespec="seconds")
        self.run_dir = experiment_root / run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.report_only = report_only
        self.config_path = self.run_dir / "config.json"
        self.history_path = self.run_dir / "history.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._summary: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "report_only": report_only,
            "args": sanitize_for_json(args),
            "splits": [],
            "aggregate": {},
        }
        with self.config_path.open("w", encoding="utf-8") as config_file:
            json.dump(self._summary["args"], config_file, indent=2)

    def record_split_setup(
        self,
        *,
        split_seed: int,
        participant_splits: dict[str, list[str]],
        full_summaries: dict[str, Any],
        loaded_summaries: dict[str, Any],
    ) -> None:
        self._summary["splits"].append(
            {
                "split_seed": split_seed,
                "participant_splits": sanitize_for_json(participant_splits),
                "full_summaries": sanitize_for_json(full_summaries),
                "loaded_summaries": sanitize_for_json(loaded_summaries),
                "final_metrics": {},
                "best_checkpoint": {},
                "test_evaluation_checkpoint": {},
            }
        )

    def record_epoch(
        self,
        *,
        split_seed: int,
        epoch: int,
        train_metrics: Any,
        val_metrics: Any,
    ) -> None:
        record = {
            "split_seed": split_seed,
            "epoch": epoch,
            "train_metrics": sanitize_for_json(train_metrics),
            "val_metrics": sanitize_for_json(val_metrics),
        }
        with self.history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(record) + "\n")

    def save_checkpoint(
        self,
        *,
        split_seed: int,
        epoch: int,
        checkpoint_tag: str,
        checkpoint_metric_name: str,
        checkpoint_metric_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_metrics: Any,
        val_metrics: Any,
    ) -> Path:
        checkpoint_path = (
            self.checkpoints_dir
            / f"split_seed_{split_seed}_{checkpoint_tag}.pt"
        )
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "split_seed": split_seed,
            "epoch": epoch,
            "checkpoint_tag": checkpoint_tag,
            "checkpoint_metric_name": checkpoint_metric_name,
            "checkpoint_metric_value": checkpoint_metric_value,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": sanitize_for_json(train_metrics),
            "val_metrics": sanitize_for_json(val_metrics),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def record_split_result(
        self,
        *,
        split_seed: int,
        train_metrics: Any,
        val_metrics: Any,
        test_metrics: Any,
        best_checkpoint: dict[str, Any],
        test_evaluation_checkpoint: dict[str, Any],
    ) -> None:
        split_record = self._find_split_record(split_seed)
        split_record["final_metrics"] = {
            "train": sanitize_for_json(train_metrics),
            "val": sanitize_for_json(val_metrics),
            "test": sanitize_for_json(test_metrics),
        }
        split_record["best_checkpoint"] = sanitize_for_json(best_checkpoint)
        split_record["test_evaluation_checkpoint"] = sanitize_for_json(
            test_evaluation_checkpoint
        )

    def finalize(self, *, aggregate: dict[str, Any]) -> Path:
        self._summary["aggregate"] = sanitize_for_json(aggregate)
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(self._summary, summary_file, indent=2)
        return self.summary_path

    def _find_split_record(self, split_seed: int) -> dict[str, Any]:
        for split_record in self._summary["splits"]:
            if split_record["split_seed"] == split_seed:
                return split_record
        raise KeyError(f"Missing split record for split_seed={split_seed}")
