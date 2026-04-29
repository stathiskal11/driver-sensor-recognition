from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
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
PHYSIOLOGY_SIGNAL_COLUMNS = ["ECGtoHR", "GSR"]
CAN_BUS_SIGNAL_COLUMNS = ["Throttle", "RPM", "Steering", "Speed"]
DEFAULT_SIGNAL_COLUMNS = PHYSIOLOGY_SIGNAL_COLUMNS + CAN_BUS_SIGNAL_COLUMNS
NAVIGATION_CATEGORIES = ["left", "right", "straight", "unknown"]
TRANSPARENCY_CATEGORIES = ["0", "1", "2"]
WEATHER_CATEGORIES = ["0", "1"]
EXPECTED_IMAGE_SIZE = (90, 160)
INDEX_USECOLS = [
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
INDEX_DTYPE_MAP = {
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


def default_signal_stats_path() -> Path:
    return repo_root() / "data" / "interim" / "paper_signal_stats.json"


def default_prefetched_asset_root(cache_dir: Path) -> Path:
    return cache_dir / "prefetched_assets"


def normalize_member_name(member_name: str) -> str:
    normalized = member_name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def member_name_to_path(root: Path, member_name: str) -> Path:
    normalized = normalize_member_name(member_name)
    return root.joinpath(*PurePosixPath(normalized).parts)


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


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _valid_physiology_value(column: str, value: float | None) -> bool:
    if value is None:
        return False
    if column == "ECGtoHR":
        return value >= 0.0
    if column == "GSR":
        return value >= 0.0
    return True


def compute_signal_stats(csv_archive_path: Path) -> dict[str, object]:
    physiology_accumulators: dict[str, dict[str, dict[str, float]]] = {}
    can_bus_min_max = {
        column: {"min": float("inf"), "max": float("-inf")}
        for column in CAN_BUS_SIGNAL_COLUMNS
    }
    with tarfile.open(csv_archive_path, "r:gz") as csv_tar:
        for member in csv_tar:
            if not member.isfile():
                continue
            if not member.name.endswith(".csv"):
                continue
            if "/.~lock." in member.name:
                continue
            participant_id = PurePosixPath(member.name).parts[1]
            participant_acc = physiology_accumulators.setdefault(
                participant_id,
                {
                    column: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
                    for column in PHYSIOLOGY_SIGNAL_COLUMNS
                },
            )
            extracted = csv_tar.extractfile(member)
            if extracted is None:
                continue
            rows = csv.DictReader(
                extracted.read().decode("utf-8", errors="replace").splitlines()
            )
            for row in rows:
                for column in PHYSIOLOGY_SIGNAL_COLUMNS:
                    value = _parse_float(row.get(column))
                    if not _valid_physiology_value(column, value):
                        continue
                    assert value is not None
                    column_acc = participant_acc[column]
                    column_acc["count"] += 1.0
                    column_acc["sum"] += value
                    column_acc["sum_sq"] += value * value
                for column in CAN_BUS_SIGNAL_COLUMNS:
                    value = _parse_float(row.get(column))
                    if value is None:
                        continue
                    can_bus_min_max[column]["min"] = min(
                        can_bus_min_max[column]["min"], value
                    )
                    can_bus_min_max[column]["max"] = max(
                        can_bus_min_max[column]["max"], value
                    )
    physiology_stats: dict[str, dict[str, dict[str, float]]] = {}
    for participant_id, participant_acc in physiology_accumulators.items():
        physiology_stats[participant_id] = {}
        for column, column_acc in participant_acc.items():
            count = column_acc["count"]
            if count <= 0:
                physiology_stats[participant_id][column] = {
                    "mean": 0.0,
                    "std": 1.0,
                }
                continue
            mean = column_acc["sum"] / count
            variance = max(column_acc["sum_sq"] / count - mean * mean, 0.0)
            std = variance**0.5
            physiology_stats[participant_id][column] = {
                "mean": mean,
                "std": std if std > 0 else 1.0,
            }
    can_bus_stats: dict[str, dict[str, float]] = {}
    for column, min_max in can_bus_min_max.items():
        min_value = min_max["min"]
        max_value = min_max["max"]
        if min_value == float("inf") or max_value == float("-inf"):
            min_value = 0.0
            max_value = 1.0
        if max_value <= min_value:
            max_value = min_value + 1.0
        can_bus_stats[column] = {
            "min": min_value,
            "max": max_value,
        }
    return {
        "physiology": physiology_stats,
        "can_bus": can_bus_stats,
    }


def ensure_signal_stats_cached(
    csv_archive_path: Path,
    stats_path: Path,
) -> dict[str, object]:
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as stats_file:
            return json.load(stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_signal_stats(csv_archive_path)
    with stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)
    return stats


def load_filtered_index_frame(
    index_csv_path: str | Path,
    *,
    participant_ids: Iterable[str] | None = None,
    sample_ids: Iterable[int] | None = None,
    limit_samples: int | None = None,
) -> pd.DataFrame:
    index_frame = pd.read_csv(
        index_csv_path,
        usecols=INDEX_USECOLS,
        dtype=INDEX_DTYPE_MAP,
    )
    if participant_ids is not None:
        participant_set = {str(participant_id) for participant_id in participant_ids}
        index_frame = index_frame[index_frame["participant_id"].isin(participant_set)]
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
    return index_frame.reset_index(drop=True)


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


@dataclass
class _SessionSequence:
    image_files: list[str]
    heatmap_files: list[str]
    hmi_vectors: np.ndarray
    normalized_signals: np.ndarray


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
        self._member_info_by_name: dict[str, tarfile.TarInfo] | None = None

    def _member_index_path(self) -> Path:
        return self.archive_path.with_name(
            self.archive_path.name + ".basename_index.json"
        )

    def _ensure_member_map(self) -> None:
        if self._member_name_by_basename is not None:
            return
        index_path = self._member_index_path()
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as index_file:
                self._member_name_by_basename = json.load(index_file)
            return
        mapping: dict[str, str] = {}
        with tarfile.open(self.archive_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                mapping[PurePosixPath(member.name).name] = member.name
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_index_path = index_path.with_name(index_path.name + ".tmp")
        with temp_index_path.open("w", encoding="utf-8") as index_file:
            json.dump(mapping, index_file)
        temp_index_path.replace(index_path)
        self._member_name_by_basename = mapping

    def _ensure_tar_open(self) -> None:
        if self._tar is None:
            self._tar = tarfile.open(self.archive_path, "r:gz")
        if self._member_info_by_name is None:
            assert self._tar is not None
            self._member_info_by_name = {
                member.name: member
                for member in self._tar.getmembers()
                if member.isfile()
            }

    def load(self, basename: str) -> torch.Tensor:
        cached = self.cache.get(basename)
        if cached is not None:
            return cached
        self._ensure_member_map()
        assert self._member_name_by_basename is not None
        member_name = self._member_name_by_basename.get(basename)
        if member_name is None:
            tensor = torch.zeros(self.expected_size, dtype=torch.float32)
            self.cache.put(basename, tensor)
            return tensor
        self._ensure_tar_open()
        assert self._tar is not None
        assert self._member_info_by_name is not None
        member_info = self._member_info_by_name.get(member_name)
        if member_info is None:
            tensor = torch.zeros(self.expected_size, dtype=torch.float32)
            self.cache.put(basename, tensor)
            return tensor
        extracted = self._tar.extractfile(member_info)
        if extracted is None:
            tensor = torch.zeros(self.expected_size, dtype=torch.float32)
            self.cache.put(basename, tensor)
            return tensor
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
        return tensor


class _LocalImageStore:
    def __init__(
        self,
        root: Path,
        expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
        cache_size: int = 512,
    ) -> None:
        self.root = root
        self.expected_size = expected_size
        self.cache = _ArrayCache(max_items=cache_size)

    def load(self, basename: str) -> torch.Tensor | None:
        cached = self.cache.get(basename)
        if cached is not None:
            return cached
        image_path = self.root / basename
        if not image_path.exists():
            return None
        with Image.open(image_path) as image:
            image = image.convert("L")
            if image.size != (self.expected_size[1], self.expected_size[0]):
                image = image.resize(
                    (self.expected_size[1], self.expected_size[0]), Image.NEAREST
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        self.cache.put(basename, tensor)
        return tensor


class _HybridImageStore:
    def __init__(
        self,
        *,
        archive_path: Path,
        local_root: Path | None = None,
        expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
        cache_size: int = 512,
    ) -> None:
        self.local_store = None
        if local_root is not None:
            self.local_store = _LocalImageStore(
                local_root,
                expected_size=expected_size,
                cache_size=cache_size,
            )
        self.tar_store = _TarImageStore(
            archive_path,
            expected_size=expected_size,
            cache_size=cache_size,
        )

    def load(self, basename: str) -> torch.Tensor:
        if self.local_store is not None:
            tensor = self.local_store.load(basename)
            if tensor is not None:
                return tensor
        return self.tar_store.load(basename)


def _participant_id_from_member_name(member_name: str) -> str:
    normalized_name = normalize_member_name(member_name)
    parts = PurePosixPath(normalized_name).parts
    if len(parts) < 2:
        raise ValueError(f"Could not infer participant id from csv member: {member_name}")
    return str(parts[1])


def _build_normalized_signal_column(
    rows: list[dict[str, str]],
    *,
    participant_id: str,
    column: str,
    signal_stats: dict[str, object],
) -> np.ndarray:
    parsed_values = np.asarray(
        [
            np.nan if (value := _parse_float(row.get(column))) is None else value
            for row in rows
        ],
        dtype=np.float32,
    )
    normalized = np.zeros(len(rows), dtype=np.float32)
    finite_mask = np.isfinite(parsed_values)
    if column in PHYSIOLOGY_SIGNAL_COLUMNS:
        valid_mask = finite_mask & (parsed_values >= 0.0)
        if not np.any(valid_mask):
            return normalized
        participant_stats = signal_stats.get("physiology", {}).get(participant_id, {})
        column_stats = participant_stats.get(column)
        if not column_stats:
            normalized[valid_mask] = parsed_values[valid_mask]
            return normalized
        std = float(column_stats.get("std", 1.0))
        mean = float(column_stats.get("mean", 0.0))
        if std <= 0.0:
            return normalized
        normalized[valid_mask] = (parsed_values[valid_mask] - mean) / std
        return normalized
    valid_mask = finite_mask
    if not np.any(valid_mask):
        return normalized
    if column in CAN_BUS_SIGNAL_COLUMNS:
        column_stats = signal_stats.get("can_bus", {}).get(column)
        if not column_stats:
            normalized[valid_mask] = parsed_values[valid_mask]
            return normalized
        min_value = float(column_stats.get("min", 0.0))
        max_value = float(column_stats.get("max", 1.0))
        if max_value <= min_value:
            return normalized
        scaled = (parsed_values[valid_mask] - min_value) / (max_value - min_value)
        normalized[valid_mask] = np.clip(scaled, 0.0, 1.0)
        return normalized
    normalized[valid_mask] = parsed_values[valid_mask]
    return normalized


def _build_session_sequence(
    rows: list[dict[str, str]],
    *,
    member_name: str,
    signal_columns: list[str],
    signal_stats: dict[str, object],
) -> _SessionSequence:
    participant_id = _participant_id_from_member_name(member_name)
    image_files = [str(row.get("ImageFile", "")) for row in rows]
    heatmap_files = [
        f"{timestamp}.png" if timestamp else ""
        for timestamp in (row.get("TimeStamp", "") for row in rows)
    ]
    hmi_vectors = np.asarray(
        [
            _one_hot(str(row.get("navigation", "")), NAVIGATION_CATEGORIES)
            + _one_hot(str(row.get("transparency", "")), TRANSPARENCY_CATEGORIES)
            + _one_hot(str(row.get("weather", "")), WEATHER_CATEGORIES)
            for row in rows
        ],
        dtype=np.float32,
    )
    signal_columns_data = [
        _build_normalized_signal_column(
            rows,
            participant_id=participant_id,
            column=column,
            signal_stats=signal_stats,
        )
        for column in signal_columns
    ]
    normalized_signals = (
        np.stack(signal_columns_data, axis=1)
        if signal_columns_data
        else np.zeros((len(rows), 0), dtype=np.float32)
    )
    return _SessionSequence(
        image_files=image_files,
        heatmap_files=heatmap_files,
        hmi_vectors=hmi_vectors,
        normalized_signals=normalized_signals,
    )


class _TarSessionStore:
    def __init__(
        self,
        archive_path: Path,
        *,
        signal_columns: list[str],
        signal_stats: dict[str, object],
    ) -> None:
        self.archive_path = archive_path
        self.signal_columns = signal_columns
        self.signal_stats = signal_stats
        self._tar: tarfile.TarFile | None = None
        self._cache: dict[str, _SessionSequence] = {}

    def _ensure_tar_open(self) -> None:
        if self._tar is None:
            self._tar = tarfile.open(self.archive_path, "r:gz")

    def get_session(self, member_name: str) -> _SessionSequence:
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
        session = _build_session_sequence(
            rows,
            member_name=member_name,
            signal_columns=self.signal_columns,
            signal_stats=self.signal_stats,
        )
        self._cache[member_name] = session
        return session


class _LocalSessionStore:
    def __init__(
        self,
        root: Path,
        *,
        signal_columns: list[str],
        signal_stats: dict[str, object],
    ) -> None:
        self.root = root
        self.signal_columns = signal_columns
        self.signal_stats = signal_stats
        self._cache: dict[str, _SessionSequence] = {}

    def get_session(self, member_name: str) -> _SessionSequence | None:
        normalized_name = normalize_member_name(member_name)
        cached = self._cache.get(normalized_name)
        if cached is not None:
            return cached
        csv_path = member_name_to_path(self.root, normalized_name)
        if not csv_path.exists():
            return None
        rows = list(
            csv.DictReader(
                csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        )
        session = _build_session_sequence(
            rows,
            member_name=normalized_name,
            signal_columns=self.signal_columns,
            signal_stats=self.signal_stats,
        )
        self._cache[normalized_name] = session
        return session


class _HybridSessionStore:
    def __init__(
        self,
        archive_path: Path,
        *,
        signal_columns: list[str],
        signal_stats: dict[str, object],
        local_root: Path | None = None,
    ) -> None:
        self.local_store = None
        if local_root is not None:
            self.local_store = _LocalSessionStore(
                local_root,
                signal_columns=signal_columns,
                signal_stats=signal_stats,
            )
        self.tar_store = _TarSessionStore(
            archive_path,
            signal_columns=signal_columns,
            signal_stats=signal_stats,
        )

    def get_session(self, member_name: str) -> _SessionSequence:
        if self.local_store is not None:
            session = self.local_store.get_session(member_name)
            if session is not None:
                return session
        return self.tar_store.get_session(member_name)


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


class _LocalCsvStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, list[dict[str, str]]] = {}

    def get_rows(self, member_name: str) -> list[dict[str, str]] | None:
        normalized_name = normalize_member_name(member_name)
        cached = self._cache.get(normalized_name)
        if cached is not None:
            return cached
        csv_path = member_name_to_path(self.root, normalized_name)
        if not csv_path.exists():
            return None
        rows = list(
            csv.DictReader(csv_path.read_text(encoding="utf-8", errors="replace").splitlines())
        )
        self._cache[normalized_name] = rows
        return rows


class _HybridCsvStore:
    def __init__(self, archive_path: Path, local_root: Path | None = None) -> None:
        self.local_store = _LocalCsvStore(local_root) if local_root is not None else None
        self.tar_store = _CsvSequenceStore(archive_path)

    def get_rows(self, member_name: str) -> list[dict[str, str]]:
        if self.local_store is not None:
            rows = self.local_store.get_rows(member_name)
            if rows is not None:
                return rows
        return self.tar_store.get_rows(member_name)


def _extract_selected_csv_members(
    archive_path: Path,
    target_root: Path,
    member_names: Iterable[str],
) -> dict[str, int]:
    normalized_targets = {normalize_member_name(name) for name in member_names}
    if not normalized_targets:
        return {"requested": 0, "extracted": 0, "reused": 0, "missing": 0}
    missing_targets = {
        normalized_name
        for normalized_name in normalized_targets
        if not member_name_to_path(target_root, normalized_name).exists()
    }
    pending = set(missing_targets)
    extracted_count = 0
    reused_count = len(normalized_targets) - len(missing_targets)
    target_root.mkdir(parents=True, exist_ok=True)
    if not pending:
        return {
            "requested": len(normalized_targets),
            "extracted": 0,
            "reused": reused_count,
            "missing": 0,
        }
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not pending:
                break
            if not member.isfile():
                continue
            normalized_member_name = normalize_member_name(member.name)
            if normalized_member_name not in pending:
                continue
            target_path = member_name_to_path(target_root, normalized_member_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as out_file:
                shutil.copyfileobj(extracted, out_file, length=1024 * 1024)
            extracted_count += 1
            pending.remove(normalized_member_name)
    return {
        "requested": len(normalized_targets),
        "extracted": extracted_count,
        "reused": reused_count,
        "missing": len(pending),
    }


def _extract_selected_image_basenames(
    archive_path: Path,
    target_root: Path,
    basenames: Iterable[str],
) -> dict[str, int]:
    target_set = {str(name) for name in basenames if str(name)}
    if not target_set:
        return {"requested": 0, "extracted": 0, "reused": 0, "missing": 0}
    missing_targets = {
        basename for basename in target_set if not (target_root / basename).exists()
    }
    pending = set(missing_targets)
    extracted_count = 0
    reused_count = len(target_set) - len(missing_targets)
    target_root.mkdir(parents=True, exist_ok=True)
    if not pending:
        return {
            "requested": len(target_set),
            "extracted": 0,
            "reused": reused_count,
            "missing": 0,
        }
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not pending:
                break
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            if basename not in pending:
                continue
            target_path = target_root / basename
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with target_path.open("wb") as out_file:
                shutil.copyfileobj(extracted, out_file, length=1024 * 1024)
            extracted_count += 1
            pending.remove(basename)
    return {
        "requested": len(target_set),
        "extracted": extracted_count,
        "reused": reused_count,
        "missing": len(pending),
    }


def prefetch_subset_assets(
    *,
    index_csv_path: str | Path,
    bundle_path: str | Path | None = None,
    heatmap_variant: str = "sigma64",
    participant_ids: Iterable[str] | None = None,
    sample_ids: Iterable[int] | None = None,
    limit_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    if heatmap_variant not in HEATMAP_ARCHIVES:
        raise ValueError(
            f"Unsupported heatmap_variant={heatmap_variant!r}. "
            f"Expected one of {sorted(HEATMAP_ARCHIVES)}."
        )
    bundle_path_obj = Path(bundle_path) if bundle_path else default_bundle_path()
    cache_dir_path = Path(cache_dir) if cache_dir else default_cache_dir()
    prefetched_root = default_prefetched_asset_root(cache_dir_path)
    csv_target_root = prefetched_root / "csv"
    seg_target_root = prefetched_root / "seg"
    heatmap_target_root = prefetched_root / f"heatmaps_{heatmap_variant}"
    index_frame = load_filtered_index_frame(
        index_csv_path,
        participant_ids=participant_ids,
        sample_ids=sample_ids,
        limit_samples=limit_samples,
    )
    if len(index_frame) == 0:
        raise ValueError("Cannot prefetch assets for an empty dataset subset.")
    csv_archive_path = ensure_inner_archive_cached(
        bundle_path_obj,
        CSV_ARCHIVE,
        cache_dir_path / "Synced_csv_files-participant_level.tar.gz",
    )
    seg_archive_path = ensure_inner_archive_cached(
        bundle_path_obj,
        SEG_ARCHIVE,
        cache_dir_path / "seg_img_90_160_new_dash.tar.gz",
    )
    heatmap_archive_path = ensure_inner_archive_cached(
        bundle_path_obj,
        HEATMAP_ARCHIVES[heatmap_variant],
        cache_dir_path / f"Heat_maps_90_160_{heatmap_variant}.tar.gz",
    )
    csv_member_names = index_frame["csv_member"].astype(str).tolist()
    csv_stats = _extract_selected_csv_members(
        csv_archive_path,
        csv_target_root,
        csv_member_names,
    )
    csv_store = _LocalCsvStore(csv_target_root)
    segmentation_basenames: set[str] = set()
    heatmap_basenames: set[str] = set()
    csv_member_count = 0
    sample_count = 0
    grouped_rows = index_frame.groupby("csv_member", sort=False)
    for csv_member, member_frame in grouped_rows:
        csv_member_count += 1
        rows = csv_store.get_rows(str(csv_member))
        if rows is None:
            raise FileNotFoundError(
                f"Expected prefetched CSV member is missing: {csv_member}"
            )
        for sample in member_frame.itertuples(index=False):
            sample_count += 1
            start_idx = int(sample.window_start_idx)
            end_idx = int(sample.window_end_idx)
            lookback_steps = int(sample.lookback_steps)
            window_rows = rows[start_idx : end_idx + 1]
            if len(window_rows) != lookback_steps:
                raise ValueError(
                    f"Expected {lookback_steps} rows for {csv_member}[{start_idx}:{end_idx}], "
                    f"but got {len(window_rows)} during asset prefetch."
                )
            for window_row in window_rows:
                segmentation_basenames.add(str(window_row["ImageFile"]))
                heatmap_basenames.add(f"{window_row['TimeStamp']}.png")
    seg_stats = _extract_selected_image_basenames(
        seg_archive_path,
        seg_target_root,
        segmentation_basenames,
    )
    heatmap_stats = _extract_selected_image_basenames(
        heatmap_archive_path,
        heatmap_target_root,
        heatmap_basenames,
    )
    summary = {
        "sample_count": sample_count,
        "csv_member_count": csv_member_count,
        "segmentation_file_count": len(segmentation_basenames),
        "heatmap_file_count": len(heatmap_basenames),
        "csv": csv_stats,
        "segmentation": seg_stats,
        "heatmaps": heatmap_stats,
        "prefetched_root": str(prefetched_root),
    }
    summary_path = prefetched_root / f"prefetch_summary_{heatmap_variant}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


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
        self.signal_stats_path = default_signal_stats_path()
        self.signal_columns = signal_columns or list(DEFAULT_SIGNAL_COLUMNS)
        self.heatmap_variant = heatmap_variant
        self.prefetched_root = default_prefetched_asset_root(self.cache_dir)
        self.prefetched_csv_root = self.prefetched_root / "csv"
        self.prefetched_seg_root = self.prefetched_root / "seg"
        self.prefetched_heatmap_root = (
            self.prefetched_root / f"heatmaps_{self.heatmap_variant}"
        )
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
        self.signal_stats = ensure_signal_stats_cached(
            self.csv_archive_path,
            self.signal_stats_path,
        )
        index_frame = load_filtered_index_frame(
            self.index_csv_path,
            participant_ids=participant_ids,
            sample_ids=sample_ids,
            limit_samples=limit_samples,
        )
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
        self.records = self.index.to_dict(orient="records")
        csv_local_root = self.prefetched_csv_root if self.prefetched_csv_root.exists() else None
        seg_local_root = self.prefetched_seg_root if self.prefetched_seg_root.exists() else None
        heatmap_local_root = (
            self.prefetched_heatmap_root if self.prefetched_heatmap_root.exists() else None
        )
        self.session_store = _HybridSessionStore(
            self.csv_archive_path,
            signal_columns=self.signal_columns,
            signal_stats=self.signal_stats,
            local_root=csv_local_root,
        )
        self.seg_store = _HybridImageStore(
            archive_path=self.seg_archive_path,
            local_root=seg_local_root,
        )
        self.heatmap_store = _HybridImageStore(
            archive_path=self.heatmap_archive_path,
            local_root=heatmap_local_root,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records[index]
        csv_member = str(row["csv_member"])
        start_idx = int(row["window_start_idx"])
        end_idx = int(row["window_end_idx"])
        lookback_steps = int(row["lookback_steps"])
        session = self.session_store.get_session(csv_member)
        window_image_files = session.image_files[start_idx : end_idx + 1]
        window_heatmap_files = session.heatmap_files[start_idx : end_idx + 1]
        if len(window_image_files) != lookback_steps:
            raise ValueError(
                f"Expected {lookback_steps} rows for {csv_member}[{start_idx}:{end_idx}], "
                f"but got {len(window_image_files)}."
            )
        segmentation_frames = []
        heatmap_frames = []
        for image_file, heatmap_file in zip(window_image_files, window_heatmap_files):
            segmentation_frames.append(self.seg_store.load(image_file).unsqueeze(0))
            heatmap_frames.append(self.heatmap_store.load(heatmap_file).unsqueeze(0))
        segmentation_tensor = torch.cat(segmentation_frames, dim=0)
        heatmap_tensor = torch.cat(heatmap_frames, dim=0)
        scene_gaze = torch.stack([segmentation_tensor, heatmap_tensor], dim=0)
        signals = torch.from_numpy(
            np.ascontiguousarray(session.normalized_signals[start_idx : end_idx + 1])
        )
        hmi_vector = torch.from_numpy(session.hmi_vectors[end_idx])
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
