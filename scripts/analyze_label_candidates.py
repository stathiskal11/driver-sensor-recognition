from __future__ import annotations

import argparse
import csv
import tarfile
from collections import Counter
from pathlib import Path


CSV_ARCHIVE = "./hdbd_data/Synced_csv_files-participant_level.tar.gz"
DEFAULT_HORIZONS = [0, 10, 30, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate label definitions for the paper baseline."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to hdbd.tar.gz. If omitted, the script searches common local paths.",
    )
    parser.add_argument(
        "--lookback-steps",
        type=int,
        default=30,
        help="Window length in timesteps.",
    )
    parser.add_argument(
        "--throttle-threshold",
        type=float,
        default=0.01,
        help="Threshold for declaring a throttle onset event.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def find_default_bundle() -> Path:
    repo_root = repo_root_from_script()
    candidates = [
        repo_root / "data" / "raw" / "hdbd.tar.gz",
        repo_root.parent / "hdbd.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find hdbd.tar.gz automatically. Use --bundle to provide it explicitly."
    )


def iter_csv_members(nested_tar: tarfile.TarFile):
    for member in nested_tar:
        if not member.isfile():
            continue
        if not member.name.endswith(".csv"):
            continue
        if "/.~lock." in member.name:
            continue
        yield member


def build_next_positive_index(event_flags: list[bool]) -> list[int | None]:
    next_positive: list[int | None] = [None] * len(event_flags)
    next_seen: int | None = None
    for idx in range(len(event_flags) - 1, -1, -1):
        if event_flags[idx]:
            next_seen = idx
        next_positive[idx] = next_seen
    return next_positive


def parse_throttle(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def main() -> None:
    args = parse_args()
    bundle_path = (args.bundle or find_default_bundle()).resolve()

    total_rows = 0
    total_windows = 0
    keydown_rows = 0
    throttle_onset_rows = 0

    keydown_positive_by_horizon = Counter()
    throttle_positive_by_horizon = Counter()

    with tarfile.open(bundle_path, "r:gz") as outer_tar:
        extracted = outer_tar.extractfile(CSV_ARCHIVE)
        if extracted is None:
            raise FileNotFoundError(f"Missing nested archive: {CSV_ARCHIVE}")

        with tarfile.open(fileobj=extracted, mode="r|gz") as csv_tar:
            for member in iter_csv_members(csv_tar):
                raw_bytes = csv_tar.extractfile(member)
                if raw_bytes is None:
                    continue

                rows = list(
                    csv.DictReader(
                        raw_bytes.read().decode("utf-8", errors="replace").splitlines()
                    )
                )
                total_rows += len(rows)

                keydown_flags = [
                    row.get("KeyEvent", "O") == "main_keydown" for row in rows
                ]
                keydown_rows += sum(keydown_flags)

                throttle_flags: list[bool] = []
                previous_throttle = 0.0
                for row in rows:
                    current_throttle = parse_throttle(row.get("Throttle"))
                    is_onset = (
                        current_throttle > args.throttle_threshold
                        and previous_throttle <= args.throttle_threshold
                    )
                    throttle_flags.append(is_onset)
                    previous_throttle = current_throttle
                throttle_onset_rows += sum(throttle_flags)

                keydown_next = build_next_positive_index(keydown_flags)
                throttle_next = build_next_positive_index(throttle_flags)

                for end_idx in range(args.lookback_steps - 1, len(rows)):
                    total_windows += 1
                    for horizon in DEFAULT_HORIZONS:
                        keydown_next_idx = keydown_next[end_idx]
                        if (
                            keydown_next_idx is not None
                            and keydown_next_idx <= end_idx + horizon
                        ):
                            keydown_positive_by_horizon[horizon] += 1

                        throttle_next_idx = throttle_next[end_idx]
                        if (
                            throttle_next_idx is not None
                            and throttle_next_idx <= end_idx + horizon
                        ):
                            throttle_positive_by_horizon[horizon] += 1

    print(f"bundle_path={bundle_path}")
    print(f"lookback_steps={args.lookback_steps}")
    print(f"total_rows={total_rows}")
    print(f"total_windows={total_windows}")
    print(f"keydown_rows={keydown_rows}")
    print(f"throttle_onset_rows={throttle_onset_rows}")

    print()
    print("keydown-based candidates")
    print("-----------------------")
    for horizon in DEFAULT_HORIZONS:
        positive_count = keydown_positive_by_horizon[horizon]
        positive_rate = (positive_count / total_windows) if total_windows else 0.0
        print(
            f"horizon_steps={horizon} "
            f"positive_windows={positive_count} "
            f"positive_rate={positive_rate:.6f}"
        )

    print()
    print("throttle-onset candidates")
    print("-------------------------")
    for horizon in DEFAULT_HORIZONS:
        positive_count = throttle_positive_by_horizon[horizon]
        positive_rate = (positive_count / total_windows) if total_windows else 0.0
        print(
            f"horizon_steps={horizon} "
            f"positive_windows={positive_count} "
            f"positive_rate={positive_rate:.6f}"
        )


if __name__ == "__main__":
    main()
