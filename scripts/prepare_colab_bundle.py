from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy hdbd.tar.gz from Google Drive to a local Colab path."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source path of hdbd.tar.gz on Google Drive.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/content/hdbd.tar.gz"),
        help="Local destination path on the Colab runtime disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the local copy even if a matching file already exists.",
    )
    return parser.parse_args()


def human_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source bundle not found: {source}")
    if dest.exists() and not args.force:
        if dest.stat().st_size == source.stat().st_size:
            print("Local bundle already exists with matching size.")
            print(f"source={source}")
            print(f"dest={dest}")
            print(f"size={human_gb(source.stat().st_size)}")
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying bundle from Drive to local runtime disk...")
    print(f"source={source}")
    print(f"dest={dest}")
    print(f"size={human_gb(source.stat().st_size)}")
    with source.open("rb") as src_file, dest.open("wb") as dst_file:
        shutil.copyfileobj(src_file, dst_file, length=16 * 1024 * 1024)
    print("Copy complete.")
    print(f"dest={dest}")
    print(f"size={human_gb(dest.stat().st_size)}")
if __name__ == "__main__":
    main()
