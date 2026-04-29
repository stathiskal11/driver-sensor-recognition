from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


CSV_ARCHIVE = "./hdbd_data/Synced_csv_files-participant_level.tar.gz"


@dataclass
class BuildStats:
    bundle_path: str
    output_csv: str
    output_summary_json: str
    lookback_steps: int
    stride: int
    label_mode: str
    prediction_horizon_steps: int
    csv_files: int = 0
    participants: int = 0
    total_rows: int = 0
    total_windows: int = 0
    positive_windows: int = 0
    positive_rate_percent: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style sliding-window index from HDBD participant-level CSV files."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to hdbd.tar.gz. If omitted, the script searches common local paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for the sample index.",
    )
    parser.add_argument(
        "--lookback-steps",
        type=int,
        default=30,
        help="Number of timesteps per sample window. Paper default is 30 for 3 seconds at 10 Hz.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride in timesteps.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["final_keydown", "final_non_o", "future_keydown", "future_non_o"],
        default="future_keydown",
        help="Current default uses a future-keydown rule because it better matches the paper's reported positive-rate scale.",
    )
    parser.add_argument(
        "--prediction-horizon-steps",
        type=int,
        default=10,
        help="How many future timesteps are searched for a positive event. At 10 Hz, 10 steps is 1 second.",
    )
    parser.add_argument(
        "--limit-csv-files",
        type=int,
        default=None,
        help="Optional debugging limit on the number of participant-level CSV files processed.",
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


def default_output_path() -> Path:
    return repo_root_from_script() / "data" / "interim" / "paper_window_index.csv"


def event_matches(key_event: str, label_mode: str) -> bool:
    if label_mode in {"final_keydown", "future_keydown"}:
        return key_event == "main_keydown"
    if label_mode in {"final_non_o", "future_non_o"}:
        return key_event != "O"
    raise ValueError(f"Unsupported label mode: {label_mode}")


def effective_horizon_steps(label_mode: str, requested_horizon: int) -> int:
    if label_mode.startswith("final_"):
        return 0
    return requested_horizon


def build_next_positive_index(rows: list[dict[str, str]], label_mode: str) -> list[int | None]:
    next_positive: list[int | None] = [None] * len(rows)
    next_seen: int | None = None
    for idx in range(len(rows) - 1, -1, -1):
        if event_matches(rows[idx].get("KeyEvent", "O"), label_mode):
            next_seen = idx
        next_positive[idx] = next_seen
    return next_positive


def iter_csv_members(nested_tar: tarfile.TarFile):
    for member in nested_tar:
        if not member.isfile():
            continue
        if not member.name.endswith(".csv"):
            continue
        if "/.~lock." in member.name:
            continue
        yield member


def main() -> None:
    args = parse_args()
    bundle_path = (args.bundle or find_default_bundle()).resolve()
    output_path = (args.output or default_output_path()).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    horizon_steps = effective_horizon_steps(
        args.label_mode, args.prediction_horizon_steps
    )
    participants: set[str] = set()
    label_counter: Counter[int] = Counter()
    total_rows = 0
    total_windows = 0
    processed_csvs = 0
    with tarfile.open(bundle_path, "r:gz") as outer_tar:
        extracted = outer_tar.extractfile(CSV_ARCHIVE)
        if extracted is None:
            raise FileNotFoundError(f"Missing nested archive: {CSV_ARCHIVE}")
        with tarfile.open(fileobj=extracted, mode="r|gz") as csv_tar, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as out_file:
            fieldnames = [
                "sample_id",
                "participant_id",
                "session_id",
                "csv_member",
                "window_start_idx",
                "window_end_idx",
                "lookback_steps",
                "stride",
                "label_mode",
                "prediction_horizon_steps",
                "label",
                "next_positive_offset_steps",
                "window_start_timestamp",
                "window_end_timestamp",
                "window_duration_ms",
                "window_end_vid_ts",
                "final_keyevent",
                "final_image_file",
                "final_heatmap_file",
                "navigation",
                "transparency",
                "weather",
            ]
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            sample_id = 0
            for member in iter_csv_members(csv_tar):
                processed_csvs += 1
                if args.limit_csv_files is not None and processed_csvs > args.limit_csv_files:
                    break
                participant_id = PurePosixPath(member.name).parts[1]
                participants.add(participant_id)
                session_id = PurePosixPath(member.name).name
                raw_bytes = csv_tar.extractfile(member)
                if raw_bytes is None:
                    continue
                rows = list(
                    csv.DictReader(
                        raw_bytes.read().decode("utf-8", errors="replace").splitlines()
                    )
                )
                total_rows += len(rows)
                next_positive = build_next_positive_index(rows, args.label_mode)
                for end_idx in range(args.lookback_steps - 1, len(rows), args.stride):
                    start_idx = end_idx - args.lookback_steps + 1
                    next_idx = next_positive[end_idx]
                    label = 0
                    offset_steps = ""
                    if next_idx is not None and next_idx <= end_idx + horizon_steps:
                        label = 1
                        offset_steps = str(next_idx - end_idx)
                    end_row = rows[end_idx]
                    start_row = rows[start_idx]
                    start_ts = int(start_row["TimeStamp"])
                    end_ts = int(end_row["TimeStamp"])
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "participant_id": participant_id,
                            "session_id": session_id,
                            "csv_member": member.name,
                            "window_start_idx": start_idx,
                            "window_end_idx": end_idx,
                            "lookback_steps": args.lookback_steps,
                            "stride": args.stride,
                            "label_mode": args.label_mode,
                            "prediction_horizon_steps": horizon_steps,
                            "label": label,
                            "next_positive_offset_steps": offset_steps,
                            "window_start_timestamp": start_ts,
                            "window_end_timestamp": end_ts,
                            "window_duration_ms": end_ts - start_ts,
                            "window_end_vid_ts": end_row["vid_ts"],
                            "final_keyevent": end_row["KeyEvent"],
                            "final_image_file": end_row["ImageFile"],
                            "final_heatmap_file": f"{end_row['TimeStamp']}.png",
                            "navigation": end_row["navigation"],
                            "transparency": end_row["transparency"],
                            "weather": end_row["weather"],
                        }
                    )
                    label_counter[label] += 1
                    total_windows += 1
                    sample_id += 1
    positive_windows = label_counter[1]
    positive_rate_percent = (100.0 * positive_windows / total_windows) if total_windows else 0.0
    stats = BuildStats(
        bundle_path=str(bundle_path),
        output_csv=str(output_path),
        output_summary_json=str(summary_path),
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        label_mode=args.label_mode,
        prediction_horizon_steps=horizon_steps,
        csv_files=processed_csvs if args.limit_csv_files is None else min(processed_csvs, args.limit_csv_files),
        participants=len(participants),
        total_rows=total_rows,
        total_windows=total_windows,
        positive_windows=positive_windows,
        positive_rate_percent=round(positive_rate_percent, 4),
    )
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(asdict(stats), summary_file, indent=2)
    print("Paper window index built successfully.")
    print(f"bundle_path={bundle_path}")
    print(f"output_csv={output_path}")
    print(f"output_summary_json={summary_path}")
    print(f"csv_files={stats.csv_files}")
    print(f"participants={stats.participants}")
    print(f"total_rows={stats.total_rows}")
    print(f"total_windows={stats.total_windows}")
    print(f"positive_windows={stats.positive_windows}")
    print(f"positive_rate_percent={stats.positive_rate_percent}")
    print(f"label_mode={stats.label_mode}")
    print(f"prediction_horizon_steps={stats.prediction_horizon_steps}")
if __name__ == "__main__":
    main()
