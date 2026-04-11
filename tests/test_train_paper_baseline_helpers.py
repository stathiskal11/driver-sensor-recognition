from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_paper_baseline.py"
TEST_TMP_ROOT = REPO_ROOT / ".tmp" / "unit_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location("train_paper_baseline", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
train_paper_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_paper_baseline)


class ParseArgsTests(unittest.TestCase):
    def test_parse_args_uses_config_defaults_and_cli_overrides(self) -> None:
        case_dir = TEST_TMP_ROOT / "parse_args_defaults"
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "batch_size": 8,
                    "epochs": 3,
                    "evaluate_test": True,
                    "max_train_batches": None,
                    "index": "custom/index.csv",
                }
            ),
            encoding="utf-8",
        )

        args = train_paper_baseline.parse_args(
            ["--config", str(config_path), "--epochs", "5"]
        )

        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.epochs, 5)
        self.assertTrue(args.evaluate_test)
        self.assertIsNone(args.max_train_batches)
        self.assertEqual(args.index, Path("custom/index.csv"))

    def test_parse_args_rejects_unknown_config_keys(self) -> None:
        case_dir = TEST_TMP_ROOT / "parse_args_bad_config"
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "bad_config.json"
        config_path.write_text(
            json.dumps({"unknown_option": 123}),
            encoding="utf-8",
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train_paper_baseline.parse_args(["--config", str(config_path)])


class WarningTests(unittest.TestCase):
    def test_collect_run_warnings_flags_debug_settings(self) -> None:
        args = train_paper_baseline.parse_args(
            [
                "--evaluate-test",
                "--limit-train-samples",
                "16",
                "--limit-val-samples",
                "16",
                "--limit-test-samples",
                "16",
                "--max-train-batches",
                "4",
                "--max-val-batches",
                "4",
                "--max-test-batches",
                "4",
                "--subset-strategy",
                "balanced",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--num-split-groups",
                "1",
                "--test-checkpoint",
                "last",
            ]
        )

        warnings = train_paper_baseline.collect_run_warnings(args)
        combined = "\n".join(warnings)

        self.assertIn("balanced subset sampling", combined)
        self.assertIn("batch caps are active", combined)
        self.assertIn("five shuffled split groups", combined)
        self.assertIn("last checkpoint", combined)


class CheckpointMetricPayloadTests(unittest.TestCase):
    def test_checkpoint_metric_payload_falls_back_to_loss(self) -> None:
        metrics = train_paper_baseline.BinaryPredictionMetrics(
            loss=0.4,
            batch_count=2,
            example_count=8,
            positive_count=1,
            positive_rate=0.125,
            mean_probability=0.2,
            roc_auc=None,
        )

        metric_value, metric_name = train_paper_baseline.checkpoint_metric_payload(
            "val_roc_auc",
            metrics,
        )

        self.assertEqual(metric_name, "neg_val_loss_fallback")
        self.assertEqual(metric_value, -0.4)


if __name__ == "__main__":
    unittest.main()
