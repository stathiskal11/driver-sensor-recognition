from __future__ import annotations

import csv
import io
import shutil
import tarfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CSV_ARCHIVE = "./hdbd_data/Synced_csv_files-participant_level.tar.gz"
SEG_ARCHIVE = "./hdbd_data/seg_img_90_160_new_dash.tar.gz"
HEATMAP_ARCHIVES = {
    "sigma16": "./hdbd_data/Heat_maps_90_160_sigma_16.tar.gz",
    "sigma32": "./hdbd_data/Heat_maps_90_160_sigma_32.tar.gz",
    "sigma64": "./hdbd_data/Heat_maps_90_160_sigma_64.tar.gz",
    "laplace": "./hdbd_data/Heat_maps_90_160_laplace.tar.gz",
}

DEFAULT_SIGNAL_COLUMNS = [
    "ECGtoHR",
    "GSR",
    "Steering",
    "Brake",
    "RPM",
    "Speed",
]
NAVIGATION_CATEGORIES = ["left", "right", "straight", "unknown"]
TRANSPARENCY_CATEGORIES = ["0", "1", "2"]
WEATHER_CATEGORIES = ["0", "1"]
EXPECTED_IMAGE_SIZE = (90, 160)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_bundle_path() -> Path:
    root = repo_root()
    candidates = [
        root / "data" / "raw" / "hdbd.tar.gz",
        root.parent / "hdbd.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find hdbd.tar.gz automatically. Pass bundle_path explicitly."
    )


def default_index_path() -> Path:
    return repo_root() / "data" / "interim" / "paper_window_index.csv"


def default_cache_dir() -> Path:
    return repo_root() / "data" / "raw" / "hdbd_archives"


def ensure_inner_archive_cached(
    bundle_path: Path, inner_member_name: str, target_path: Path
) -> Path:
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as outer_tar:
        extracted = outer_tar.extractfile(inner_member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing inner archive: {inner_member_name}")
        with target_path.open("wb") as out_file:
            shutil.copyfileobj(extracted, out_file, length=1024 * 1024)
    return target_path


class _ArrayCache:
    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str) -> torch.Tensor | None:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


class _TarImageStore:
    def __init__(
        self,
        archive_path: Path,
        expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
        cache_size: int = 512,
    ) -> None:
        self.archive_path = archive_path
        self.expected_size = expected_size
        self.cache = _ArrayCache(max_items=cache_size)
        self._member_name_by_basename: dict[str, str] | None = None
        self._tar: tarfile.TarFile | None = None

    def _ensure_member_map(self) -> None:
        if self._member_name_by_basename is not None:
            return
        mapping: dict[str, str] = {}
        with tarfile.open(self.archive_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                mapping[PurePosixPath(member.name).name] = member.name
        self._member_name_by_basename = mapping

    def _ensure_tar_open(self) -> None:
        if self._tar is None:
            self._tar = tarfile.open(self.archive_path, "r:gz")

    def load(self, basename: str) -> torch.Tensor:
        cached = self.cache.get(basename)
        if cached is not None:
            return cached.clone()

        self._ensure_member_map()
        assert self._member_name_by_basename is not None

        member_name = self._member_name_by_basename.get(basename)
        if member_name is None:
            tensor = torch.zeros(self.expected_size, dtype=torch.float32)
            self.cache.put(basename, tensor)
            return tensor.clone()

        self._ensure_tar_open()
        assert self._tar is not None

        extracted = self._tar.extractfile(member_name)
        if extracted is None:
            tensor = torch.zeros(self.expected_size, dtype=torch.float32)
            self.cache.put(basename, tensor)
            return tensor.clone()

        image_bytes = extracted.read()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("L")
            if image.size != (self.expected_size[1], self.expected_size[0]):
                image = image.resize(
                    (self.expected_size[1], self.expected_size[0]), Image.NEAREST
                )
            array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array)
        self.cache.put(basename, tensor)
        return tensor.clone()


class _CsvSequenceStore:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path
        self._tar: tarfile.TarFile | None = None
        self._cache: dict[str, list[dict[str, str]]] = {}

    def _ensure_tar_open(self) -> None:
        if self._tar is None:
            self._tar = tarfile.open(self.archive_path, "r:gz")

    def get_rows(self, member_name: str) -> list[dict[str, str]]:
        cached = self._cache.get(member_name)
        if cached is not None:
            return cached

        self._ensure_tar_open()
        assert self._tar is not None

        extracted = self._tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing CSV member: {member_name}")

        rows = list(
            csv.DictReader(
                extracted.read().decode("utf-8", errors="replace").splitlines()
            )
        )
        self._cache[member_name] = rows
        return rows


def _one_hot(value: str, categories: Iterable[str]) -> list[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def _safe_float(value: str) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


class HDBDPaperWindowDataset(Dataset):
    def __init__(
        self,
        index_csv_path: str | Path | None = None,
        bundle_path: str | Path | None = None,
        heatmap_variant: str = "sigma64",
        signal_columns: list[str] | None = None,
        participant_ids: Iterable[str] | None = None,
        sample_ids: Iterable[int] | None = None,
        limit_samples: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if heatmap_variant not in HEATMAP_ARCHIVES:
            raise ValueError(
                f"Unsupported heatmap_variant={heatmap_variant!r}. "
                f"Expected one of {sorted(HEATMAP_ARCHIVES)}."
            )

        self.bundle_path = Path(bundle_path) if bundle_path else default_bundle_path()
        self.index_csv_path = (
            Path(index_csv_path) if index_csv_path else default_index_path()
        )
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.signal_columns = signal_columns or list(DEFAULT_SIGNAL_COLUMNS)
        self.heatmap_variant = heatmap_variant

        self.csv_archive_path = ensure_inner_archive_cached(
            self.bundle_path,
            CSV_ARCHIVE,
            self.cache_dir / "Synced_csv_files-participant_level.tar.gz",
        )
        self.seg_archive_path = ensure_inner_archive_cached(
            self.bundle_path,
            SEG_ARCHIVE,
            self.cache_dir / "seg_img_90_160_new_dash.tar.gz",
        )
        self.heatmap_archive_path = ensure_inner_archive_cached(
            self.bundle_path,
            HEATMAP_ARCHIVES[self.heatmap_variant],
            self.cache_dir / f"Heat_maps_90_160_{self.heatmap_variant}.tar.gz",
        )

        usecols = [
            "sample_id",
            "participant_id",
            "session_id",
            "csv_member",
            "window_start_idx",
            "window_end_idx",
            "lookback_steps",
            "label",
            "final_image_file",
            "final_heatmap_file",
            "navigation",
            "transparency",
            "weather",
        ]
        dtype_map = {
            "sample_id": "int64",
            "participant_id": "string",
            "session_id": "string",
            "csv_member": "string",
            "window_start_idx": "int32",
            "window_end_idx": "int32",
            "lookback_steps": "int16",
            "label": "int8",
            "final_image_file": "string",
            "final_heatmap_file": "string",
            "navigation": "string",
            "transparency": "string",
            "weather": "string",
        }
        index_frame = pd.read_csv(
            self.index_csv_path,
            usecols=usecols,
            dtype=dtype_map,
        )

        if participant_ids is not None:
            participant_set = {str(participant_id) for participant_id in participant_ids}
            index_frame = index_frame[
                index_frame["participant_id"].isin(participant_set)
            ]

        if sample_ids is not None:
            ordered_sample_ids = [int(sample_id) for sample_id in sample_ids]
            sample_id_set = set(ordered_sample_ids)
            index_frame = index_frame[index_frame["sample_id"].isin(sample_id_set)]
            sample_order = {
                sample_id: order for order, sample_id in enumerate(ordered_sample_ids)
            }
            index_frame["_sample_order"] = index_frame["sample_id"].map(sample_order)
            index_frame = index_frame.sort_values("_sample_order").drop(
                columns="_sample_order"
            )

        if limit_samples is not None:
            index_frame = index_frame.head(limit_samples)

        for column in [
            "participant_id",
            "session_id",
            "csv_member",
            "final_image_file",
            "final_heatmap_file",
            "navigation",
            "transparency",
            "weather",
        ]:
            index_frame[column] = index_frame[column].astype("category")

        self.index = index_frame.reset_index(drop=True)
        if len(self.index) == 0:
            raise ValueError("The dataset index is empty after applying filters.")

        self.csv_store = _CsvSequenceStore(self.csv_archive_path)
        self.seg_store = _TarImageStore(self.seg_archive_path)
        self.heatmap_store = _TarImageStore(self.heatmap_archive_path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.index.iloc[index]
        csv_member = str(row["csv_member"])
        start_idx = int(row["window_start_idx"])
        end_idx = int(row["window_end_idx"])
        lookback_steps = int(row["lookback_steps"])

        rows = self.csv_store.get_rows(csv_member)
        window_rows = rows[start_idx : end_idx + 1]
        if len(window_rows) != lookback_steps:
            raise ValueError(
                f"Expected {lookback_steps} rows for {csv_member}[{start_idx}:{end_idx}], "
                f"but got {len(window_rows)}."
            )

        segmentation_frames = []
        heatmap_frames = []
        signal_sequence = []

        for window_row in window_rows:
            segmentation_frames.append(
                self.seg_store.load(window_row["ImageFile"]).unsqueeze(0)
            )
            heatmap_frames.append(
                self.heatmap_store.load(f"{window_row['TimeStamp']}.png").unsqueeze(0)
            )
            signal_sequence.append(
                [_safe_float(window_row.get(column, "")) for column in self.signal_columns]
            )

        segmentation_tensor = torch.cat(segmentation_frames, dim=0)
        heatmap_tensor = torch.cat(heatmap_frames, dim=0)
        scene_gaze = torch.stack([segmentation_tensor, heatmap_tensor], dim=0)

        signals = torch.tensor(signal_sequence, dtype=torch.float32)
        final_row = window_rows[-1]
        hmi_vector = torch.tensor(
            _one_hot(final_row["navigation"], NAVIGATION_CATEGORIES)
            + _one_hot(final_row["transparency"], TRANSPARENCY_CATEGORIES)
            + _one_hot(final_row["weather"], WEATHER_CATEGORIES),
            dtype=torch.float32,
        )

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "scene_gaze": scene_gaze,
            "signals": signals,
            "hmi": hmi_vector,
            "label": label,
            "sample_id": int(row["sample_id"]),
            "participant_id": str(row["participant_id"]),
            "session_id": str(row["session_id"]),
            "csv_member": csv_member,
            "window_start_idx": start_idx,
            "window_end_idx": end_idx,
        }
