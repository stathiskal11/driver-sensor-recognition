from __future__ import annotations

"""Γρήγορος έλεγχος του HDBD archive χωρίς extraction στο disk.

Το script αυτό είναι το πρώτο βήμα του project:
- ανοίγει το εξωτερικό `hdbd.tar.gz`
- κοιτάζει τι nested archives υπάρχουν μέσα
- μετρά participants / CSVs / rows
- ελέγχει αν τα references προς segmentation images και heatmaps βρίσκονται όντως
  μέσα στα αντίστοιχα archives

Στόχος του δεν είναι να εκπαιδεύσει κάτι, αλλά να μας πει αν το dataset που
έχουμε στα χέρια μας είναι αυτό που νομίζουμε ότι είναι.
"""

import argparse
import csv
import io
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable


OUTER_README = "./hdbd_data/README.md"
CSV_ARCHIVE = "./hdbd_data/Synced_csv_files-participant_level.tar.gz"
SEG_ARCHIVE = "./hdbd_data/seg_img_90_160_new_dash.tar.gz"
HEATMAP_ARCHIVES = {
    "sigma16": "./hdbd_data/Heat_maps_90_160_sigma_16.tar.gz",
    "sigma32": "./hdbd_data/Heat_maps_90_160_sigma_32.tar.gz",
    "sigma64": "./hdbd_data/Heat_maps_90_160_sigma_64.tar.gz",
    "laplace": "./hdbd_data/Heat_maps_90_160_laplace.tar.gz",
}


@dataclass
class CsvSummary:
    """Συγκεντρωτικά στοιχεία από τα participant-level CSV αρχεία."""

    participants: set[str] = field(default_factory=set)
    csv_files: int = 0
    total_rows: int = 0
    fieldnames: list[str] = field(default_factory=list)
    sample_csvs: list[str] = field(default_factory=list)
    keyevent_counts: Counter[str] = field(default_factory=Counter)
    navigation_counts: Counter[str] = field(default_factory=Counter)
    transparency_counts: Counter[str] = field(default_factory=Counter)
    weather_counts: Counter[str] = field(default_factory=Counter)
    sampled_seg_targets: Counter[str] = field(default_factory=Counter)
    sampled_heatmap_targets: Counter[str] = field(default_factory=Counter)


@dataclass
class ArchiveCoverage:
    """Σύνοψη του πόσα sampled targets βρέθηκαν μέσα σε ένα file archive."""

    file_count: int = 0
    matched_rows: int = 0
    matched_unique_targets: set[str] = field(default_factory=set)
    first_files: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Διαβάζει τα command-line arguments του inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect the HDBD archive without extracting it to disk."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to hdbd.tar.gz. If omitted, the script searches common local paths.",
    )
    parser.add_argument(
        "--heatmap-variant",
        choices=sorted(HEATMAP_ARCHIVES),
        default="sigma64",
        help="Which precomputed heatmap archive to validate against.",
    )
    parser.add_argument(
        "--coverage-sample-rows",
        type=int,
        default=10_000,
        help="How many CSV rows to sample for image and heatmap coverage checks.",
    )
    return parser.parse_args()


def find_default_bundle(script_path: Path) -> Path:
    """Ψάχνει το hdbd.tar.gz στα πιο συνηθισμένα local paths του project."""
    repo_root = script_path.resolve().parents[1]
    candidates = [
        repo_root / "data" / "raw" / "hdbd.tar.gz",
        repo_root.parent / "hdbd.tar.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find hdbd.tar.gz automatically. Use --bundle to provide the path."
    )


def open_nested_tar(outer_tar: tarfile.TarFile, member_name: str) -> tarfile.TarFile:
    """Ανοίγει ένα inner tar.gz που βρίσκεται μέσα στο εξωτερικό bundle."""
    extracted = outer_tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not open nested archive member: {member_name}")
    return tarfile.open(fileobj=extracted, mode="r|gz")


def read_outer_readme(outer_tar: tarfile.TarFile) -> str:
    """Διαβάζει το dataset README από το outer archive αν υπάρχει."""
    extracted = outer_tar.extractfile(OUTER_README)
    if extracted is None:
        return ""
    return extracted.read().decode("utf-8", errors="replace").strip()


def scan_csv_archive(
    outer_tar: tarfile.TarFile,
    member_name: str,
    coverage_sample_rows: int,
) -> CsvSummary:
    """Σαρώνει τα participant-level CSVs και συγκεντρώνει βασικά στατιστικά.

    Εδώ χτίζουμε την πρώτη πραγματική εικόνα του dataset:
    - πόσοι participants υπάρχουν
    - πόσα CSV sessions υπάρχουν
    - ποια πεδία έχουν τα rows
    - πόσο sparse είναι τα key events
    """
    summary = CsvSummary()
    sampled_rows = 0

    with open_nested_tar(outer_tar, member_name) as nested_tar:
        for member in nested_tar:
            if not member.isfile():
                continue
            if not member.name.endswith(".csv"):
                continue
            if "/.~lock." in member.name:
                continue

            summary.csv_files += 1
            participant = PurePosixPath(member.name).parts[1]
            summary.participants.add(participant)
            if len(summary.sample_csvs) < 5:
                summary.sample_csvs.append(member.name)

            extracted = nested_tar.extractfile(member)
            if extracted is None:
                continue

            # Διαβάζουμε όλο το CSV member στη μνήμη γιατί κάθε participant/session
            # file είναι σχετικά μικρό και αυτό απλοποιεί το inspection logic.
            decoded_lines = extracted.read().decode("utf-8", errors="replace").splitlines()
            reader = csv.DictReader(decoded_lines)
            if not summary.fieldnames:
                summary.fieldnames = reader.fieldnames or []

            for row in reader:
                summary.total_rows += 1
                summary.keyevent_counts[row.get("KeyEvent", "<missing>")] += 1
                summary.navigation_counts[row.get("navigation", "<missing>")] += 1
                summary.transparency_counts[row.get("transparency", "<missing>")] += 1
                summary.weather_counts[row.get("weather", "<missing>")] += 1

                if sampled_rows < coverage_sample_rows:
                    # Sampled coverage keeps this script fast while still giving
                    # us a strong sanity check for image and heatmap targets.
                    image_file = row.get("ImageFile")
                    timestamp = row.get("TimeStamp")
                    if image_file:
                        summary.sampled_seg_targets[image_file] += 1
                    if timestamp:
                        summary.sampled_heatmap_targets[f"{timestamp}.png"] += 1
                    sampled_rows += 1

    return summary


def scan_file_archive(
    outer_tar: tarfile.TarFile, member_name: str, sampled_targets: Counter[str]
) -> ArchiveCoverage:
    """Ελέγχει αν sampled image/heatmap targets όντως υπάρχουν στο archive."""
    coverage = ArchiveCoverage()
    target_names = set(sampled_targets)

    with open_nested_tar(outer_tar, member_name) as nested_tar:
        for member in nested_tar:
            if not member.isfile():
                continue

            coverage.file_count += 1
            basename = PurePosixPath(member.name).name
            if len(coverage.first_files) < 3:
                coverage.first_files.append(member.name)

            if basename in target_names:
                coverage.matched_rows += sampled_targets[basename]
                coverage.matched_unique_targets.add(basename)

    return coverage


def format_counter(counter: Counter[str]) -> str:
    """Τυπώνει ένα Counter σε σταθερή και εύκολα αναγνώσιμη μορφή."""
    parts = [f"{key}={value}" for key, value in sorted(counter.items())]
    return ", ".join(parts)


def format_missing(sampled_targets: Counter[str], matched_targets: set[str]) -> str:
    """Δείχνει ένα μικρό preview από targets που δεν βρέθηκαν."""
    missing = sorted(set(sampled_targets) - matched_targets)
    if not missing:
        return "none"
    preview = ", ".join(missing[:5])
    if len(missing) > 5:
        preview += ", ..."
    return preview


def positive_rate(keyevent_counts: Counter[str], total_rows: int) -> str:
    """Υπολογίζει πόσο συχνό είναι το `main_keydown` στο raw row level."""
    if total_rows == 0:
        return "0.00%"
    positives = keyevent_counts.get("main_keydown", 0)
    return f"{(100.0 * positives / total_rows):.2f}%"


def print_section(title: str) -> None:
    """Μικρό helper για πιο καθαρό terminal output."""
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    """Κύρια ροή του inspection.

    Η σειρά είναι:
    1. εντοπισμός του bundle
    2. άνοιγμα outer archive
    3. inspection των CSVs
    4. coverage check για segmentation και heatmaps
    5. εκτύπωση των συμπερασμάτων
    """
    args = parse_args()
    bundle_path = args.bundle or find_default_bundle(Path(__file__))
    heatmap_member = HEATMAP_ARCHIVES[args.heatmap_variant]

    with tarfile.open(bundle_path, "r:gz") as outer_tar:
        # Παίρνουμε πρώτα μια γενική εικόνα του outer archive και μετά μπαίνουμε
        # στα nested αρχεία που μας ενδιαφέρουν για το baseline.
        outer_members = outer_tar.getmembers()
        member_names = [member.name for member in outer_members]
        readme_text = read_outer_readme(outer_tar)

        csv_summary = scan_csv_archive(
            outer_tar, CSV_ARCHIVE, args.coverage_sample_rows
        )
        seg_coverage = scan_file_archive(
            outer_tar, SEG_ARCHIVE, csv_summary.sampled_seg_targets
        )
        heatmap_coverage = scan_file_archive(
            outer_tar, heatmap_member, csv_summary.sampled_heatmap_targets
        )

    print(f"HDBD bundle: {bundle_path}")

    print_section("Outer Bundle Members")
    print(f"member_count={len(member_names)}")
    for name in member_names:
        print(f"- {name}")

    if readme_text:
        print_section("Dataset README Snippet")
        snippet = readme_text.splitlines()[:8]
        for line in snippet:
            print(line)

    print_section("Participant-Level CSV Summary")
    print(f"participants={len(csv_summary.participants)}")
    print(f"csv_files={csv_summary.csv_files}")
    print(f"total_rows={csv_summary.total_rows}")
    print(f"positive_rate_main_keydown={positive_rate(csv_summary.keyevent_counts, csv_summary.total_rows)}")
    print(f"fieldnames={csv_summary.fieldnames}")
    print(f"sample_csvs={csv_summary.sample_csvs}")
    print(f"keyevent_counts={format_counter(csv_summary.keyevent_counts)}")
    print(f"navigation_counts={format_counter(csv_summary.navigation_counts)}")
    print(f"transparency_counts={format_counter(csv_summary.transparency_counts)}")
    print(f"weather_counts={format_counter(csv_summary.weather_counts)}")

    print_section("Segmentation Archive Check")
    print(f"archive_member={SEG_ARCHIVE}")
    print(f"file_count={seg_coverage.file_count}")
    print(f"first_files={seg_coverage.first_files}")
    print(
        f"sample_row_coverage={seg_coverage.matched_rows}/{sum(csv_summary.sampled_seg_targets.values())}"
    )
    print(
        f"missing_sample_targets={format_missing(csv_summary.sampled_seg_targets, seg_coverage.matched_unique_targets)}"
    )

    print_section("Heatmap Archive Check")
    print(f"archive_member={heatmap_member}")
    print(f"file_count={heatmap_coverage.file_count}")
    print(f"first_files={heatmap_coverage.first_files}")
    print(
        f"sample_row_coverage={heatmap_coverage.matched_rows}/{sum(csv_summary.sampled_heatmap_targets.values())}"
    )
    print(
        f"missing_sample_targets={format_missing(csv_summary.sampled_heatmap_targets, heatmap_coverage.matched_unique_targets)}"
    )

    print_section("Interpretation")
    print(
        "- The archive already contains precomputed 90x160 segmentation images and heatmaps, so full extraction is not required for the first baseline."
    )
    print(
        "- The raw key events are extremely sparse, so label generation must be defined carefully instead of using a naive row-level positive class."
    )
    print(
        "- Participant-independent splits are required for a faithful reproduction."
    )


if __name__ == "__main__":
    main()
