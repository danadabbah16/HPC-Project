"""Sanity checks on the Sleep-EDF data pipeline for docs/eeg_review.md."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import xlrd
from scipy.stats import spearmanr

from src.data.preprocess import EPOCH_SEC, STAGE_MAP, STAGE_NAMES, find_pairs

SPLIT_NAMES = ["train", "val", "test"]
N3_INDEX = STAGE_NAMES.index("N3")
# Label times are stored as floats, so allow a tiny rounding error.
GRID_TOLERANCE_S = 0.01
# Sleep-cassette names are SC4ssN..., where ss is the subject and N the night.
SLEEP_CASSETTE_PREFIX = "SC4"
SUBJECT_ID_LENGTH = 2
SHARE_DECIMALS = 4


def get_subject_id(recording_name: str) -> str:
    """Return the two-digit subject ID from a recording name like SC4012."""
    subject_start = len(SLEEP_CASSETTE_PREFIX)
    subject_end = subject_start + SUBJECT_ID_LENGTH
    if not recording_name.startswith(SLEEP_CASSETTE_PREFIX):
        raise ValueError(f"Unexpected recording name: {recording_name}")
    if len(recording_name) <= subject_end:
        raise ValueError(f"Recording name too short: {recording_name}")
    return recording_name[subject_start:subject_end]


def load_json(json_path: Path) -> dict:
    """Return the contents of a JSON file."""
    try:
        with open(json_path) as json_file:
            return json.load(json_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {json_path}: {error}") from error


def write_csv(output_path: Path, header: list[str], rows: list[list]) -> None:
    """Write rows to a CSV file with a header line."""
    try:
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(rows)
    except OSError as error:
        raise RuntimeError(f"Could not write {output_path}: {error}") from error
    print(f"Wrote {output_path}")


def read_psg_start(psg_path: Path) -> datetime | None:
    """Return the recording start time from a PSG file header."""
    try:
        # preload=False reads only the header, not the signals.
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    except Exception as error:
        raise RuntimeError(f"Could not read PSG header {psg_path}: {error}") from error
    return raw.info["meas_date"]


def read_hypnogram(hypnogram_path: Path) -> mne.Annotations:
    """Return the expert sleep-stage annotations of one night."""
    try:
        return mne.read_annotations(hypnogram_path)
    except Exception as error:
        raise RuntimeError(
            f"Could not read hypnogram {hypnogram_path}: {error}"
        ) from error


def compute_start_offset_s(
    psg_start: datetime | None, hypnogram_start: datetime | None
) -> float:
    """Return how many seconds the hypnogram starts after the PSG."""
    if psg_start is None or hypnogram_start is None:
        raise ValueError("Start time missing in the PSG or hypnogram header")
    offset = hypnogram_start - psg_start
    return offset.total_seconds()


def count_off_grid_times(times_s: np.ndarray) -> int:
    """Return how many times are not a whole number of 30 s epochs."""
    times_in_epochs = times_s / EPOCH_SEC
    distance_s = np.abs(times_in_epochs - np.round(times_in_epochs)) * EPOCH_SEC
    return int(np.sum(distance_s > GRID_TOLERANCE_S))


def check_label_alignment(
    psg_start: datetime | None, annotations: mne.Annotations
) -> dict:
    """Return the start offset and the number of off-grid stage labels for one night."""
    # Only stage labels become training labels, so only they must sit on the 30 s grid.
    stage_onsets_s = []
    stage_durations_s = []
    for annotation in annotations:
        if annotation["description"] in STAGE_MAP:
            stage_onsets_s.append(annotation["onset"])
            stage_durations_s.append(annotation["duration"])

    return {
        "start_offset_s": compute_start_offset_s(psg_start, annotations.orig_time),
        "off_grid_onsets": count_off_grid_times(np.array(stage_onsets_s)),
        "off_grid_durations": count_off_grid_times(np.array(stage_durations_s)),
    }


def has_alignment_problem(alignment: dict) -> bool:
    """Return True if the labels of a night are shifted or off the 30 s grid."""
    if alignment["start_offset_s"] != 0:
        return True
    return alignment["off_grid_onsets"] > 0 or alignment["off_grid_durations"] > 0


def count_annotation_epochs(annotations: mne.Annotations) -> dict[str, int]:
    """Return the number of 30 s epochs per original R&K annotation label."""
    epoch_counts = {}
    for annotation in annotations:
        description = annotation["description"]
        n_epochs = round(annotation["duration"] / EPOCH_SEC)
        epoch_counts[description] = epoch_counts.get(description, 0) + n_epochs
    return epoch_counts


def find_shared_subjects(splits: dict[str, list[str]]) -> list[str]:
    """Return subject IDs that appear in more than one split."""
    split_of_subject = {}
    shared_subjects = []
    for split_name, recording_names in splits.items():
        for recording_name in recording_names:
            subject_id = get_subject_id(recording_name)
            first_split = split_of_subject.get(subject_id)
            if first_split is None:
                split_of_subject[subject_id] = split_name
            elif first_split != split_name and subject_id not in shared_subjects:
                shared_subjects.append(subject_id)
    return sorted(shared_subjects)


def count_stages_per_split(
    meta: dict, splits: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Return epoch counts per sleep stage for each split, using meta.json."""
    counts_per_split = {}
    for split_name in SPLIT_NAMES:
        if split_name not in splits:
            raise ValueError(f"Split '{split_name}' missing from splits.json")
        split_counts = np.zeros(len(STAGE_NAMES), dtype=np.int64)
        for recording_name in splits[split_name]:
            if recording_name not in meta["recordings"]:
                raise ValueError(f"Recording {recording_name} missing from meta.json")
            split_counts += np.array(meta["recordings"][recording_name]["counts"])
        counts_per_split[split_name] = split_counts
    return counts_per_split


def load_subject_ages(subjects_xls_path: Path) -> dict[str, int]:
    """Return the age of each subject from PhysioNet's SC-subjects.xls."""
    try:
        sheet = xlrd.open_workbook(subjects_xls_path).sheet_by_index(0)
    except Exception as error:
        raise RuntimeError(f"Could not read {subjects_xls_path}: {error}") from error

    header = sheet.row_values(0)
    if "subject" not in header or "age" not in header:
        raise ValueError(
            f"Expected 'subject' and 'age' columns in {subjects_xls_path}, got {header}"
        )
    subject_column = header.index("subject")
    age_column = header.index("age")

    # The sheet has one row per night, so each subject appears once or twice.
    ages = {}
    for row_index in range(1, sheet.nrows):
        row = sheet.row_values(row_index)
        # Excel stores numbers as floats, e.g. 1.0 for subject "01".
        subject_id = str(int(row[subject_column])).zfill(SUBJECT_ID_LENGTH)
        ages[subject_id] = int(row[age_column])
    return ages


def compute_n3_share_per_subject(meta: dict) -> dict[str, float]:
    """Return each subject's fraction of N3 epochs, pooled over their nights."""
    n3_epochs = {}
    total_epochs = {}
    for recording_name in meta["recordings"]:
        recording_info = meta["recordings"][recording_name]
        subject_id = get_subject_id(recording_name)
        n3_epochs[subject_id] = (
            n3_epochs.get(subject_id, 0) + recording_info["counts"][N3_INDEX]
        )
        total_epochs[subject_id] = (
            total_epochs.get(subject_id, 0) + recording_info["n_epochs"]
        )

    n3_shares = {}
    for subject_id, subject_n3_epochs in n3_epochs.items():
        n3_shares[subject_id] = subject_n3_epochs / total_epochs[subject_id]
    return n3_shares


def find_split_of_subjects(splits: dict[str, list[str]]) -> dict[str, str]:
    """Return the split name of every subject."""
    split_of_subject = {}
    for split_name, recording_names in splits.items():
        for recording_name in recording_names:
            split_of_subject[get_subject_id(recording_name)] = split_name
    return split_of_subject


def summarise_ages_per_split(
    splits: dict[str, list[str]], ages: dict[str, int]
) -> list[list]:
    """Return one row per split: name, subject count, mean, median, min, max age."""
    split_of_subject = find_split_of_subjects(splits)
    rows = []
    for split_name in SPLIT_NAMES:
        split_ages = []
        for subject_id in split_of_subject:
            if split_of_subject[subject_id] != split_name:
                continue
            if subject_id not in ages:
                raise ValueError(f"No age found for subject {subject_id}")
            split_ages.append(ages[subject_id])
        if len(split_ages) == 0:
            raise ValueError(f"Split '{split_name}' has no subjects")
        mean_age = round(float(np.mean(split_ages)), 1)
        median_age = float(np.median(split_ages))
        rows.append(
            [
                split_name,
                len(split_ages),
                mean_age,
                median_age,
                min(split_ages),
                max(split_ages),
            ]
        )
    return rows


def run_alignment_checks(raw_dir: Path, out_dir: Path) -> None:
    """Check label timing and count original labels for every night."""
    recording_pairs = find_pairs(raw_dir)
    print(f"Checking label alignment in {len(recording_pairs)} nights")

    alignment_rows = []
    n_problem_nights = 0
    total_label_epochs = {}
    for recording_name, psg_path, hypnogram_path in recording_pairs:
        annotations = read_hypnogram(hypnogram_path)
        alignment = check_label_alignment(read_psg_start(psg_path), annotations)
        alignment_rows.append(
            [
                recording_name,
                alignment["start_offset_s"],
                alignment["off_grid_onsets"],
                alignment["off_grid_durations"],
            ]
        )
        if has_alignment_problem(alignment):
            n_problem_nights += 1
            print(f"PROBLEM in {recording_name}: {alignment}")

        night_label_epochs = count_annotation_epochs(annotations)
        for description, n_epochs in night_label_epochs.items():
            previous_count = total_label_epochs.get(description, 0)
            total_label_epochs[description] = previous_count + n_epochs

    n_nights = len(alignment_rows)
    print(f"Nights with misaligned labels: {n_problem_nights} of {n_nights}")
    write_csv(
        out_dir / "label_alignment.csv",
        ["recording", "start_offset_s", "off_grid_onsets", "off_grid_durations"],
        alignment_rows,
    )

    label_rows = []
    print("Original labels over whole recordings, before wake trimming:")
    for description in sorted(total_label_epochs):
        n_epochs = total_label_epochs[description]
        label_rows.append([description, n_epochs])
        print(f"{description}: {n_epochs} epochs")
    write_csv(
        out_dir / "raw_label_counts.csv", ["original_label", "n_epochs"], label_rows
    )


def run_split_checks(processed_dir: Path, out_dir: Path) -> None:
    """Check the subject split and recount stages per split."""
    meta = load_json(processed_dir / "meta.json")
    splits = load_json(processed_dir / "splits.json")

    shared_subjects = find_shared_subjects(splits)
    if shared_subjects:
        print(f"LEAKAGE: subjects in more than one split: {shared_subjects}")
    else:
        print("No subject appears in more than one split")

    counts_per_split = count_stages_per_split(meta, splits)
    stage_rows = []
    for split_name in SPLIT_NAMES:
        split_counts = counts_per_split[split_name]
        split_total = int(split_counts.sum())
        for stage_index in range(len(STAGE_NAMES)):
            stage_count = int(split_counts[stage_index])
            stage_share = round(stage_count / split_total, SHARE_DECIMALS)
            stage_rows.append(
                [split_name, STAGE_NAMES[stage_index], stage_count, stage_share]
            )
        n3_share = round(int(split_counts[N3_INDEX]) / split_total, SHARE_DECIMALS)
        print(f"{split_name}: {split_total} epochs, N3 share {n3_share}")
    write_csv(
        out_dir / "stage_counts_per_split.csv",
        ["split", "stage", "n_epochs", "share"],
        stage_rows,
    )


def run_age_checks(processed_dir: Path, subjects_xls_path: Path, out_dir: Path) -> None:
    """Compare ages between splits and relate age to the amount of N3."""
    meta = load_json(processed_dir / "meta.json")
    splits = load_json(processed_dir / "splits.json")
    ages = load_subject_ages(subjects_xls_path)

    age_rows = summarise_ages_per_split(splits, ages)
    for row in age_rows:
        print(f"{row[0]}: {row[1]} subjects, mean age {row[2]}, median {row[3]}")
        print(f"{row[0]}: youngest {row[4]}, oldest {row[5]}")
    write_csv(
        out_dir / "ages_per_split.csv",
        ["split", "n_subjects", "mean_age", "median_age", "min_age", "max_age"],
        age_rows,
    )

    n3_shares = compute_n3_share_per_subject(meta)
    split_of_subject = find_split_of_subjects(splits)
    subject_rows = []
    subject_ages = []
    subject_n3_shares = []
    for subject_id in sorted(n3_shares):
        subject_rows.append(
            [
                subject_id,
                split_of_subject[subject_id],
                ages[subject_id],
                round(n3_shares[subject_id], SHARE_DECIMALS),
            ]
        )
        subject_ages.append(ages[subject_id])
        subject_n3_shares.append(n3_shares[subject_id])
    write_csv(
        out_dir / "age_vs_n3.csv", ["subject", "split", "age", "n3_share"], subject_rows
    )

    # Spearman uses ranks, so it does not assume N3 falls linearly with age.
    correlation = spearmanr(subject_ages, subject_n3_shares)
    rho = round(float(correlation.statistic), 2)
    p_value = float(correlation.pvalue)
    print(f"Age vs N3 share over {len(subject_ages)} subjects:")
    print(f"Spearman rho {rho}, p-value {p_value:.2g}")


def main() -> None:
    """Run all review checks and save the results as CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default="data/raw/sleep-cassette",
        help="folder with the Sleep-EDF EDF files",
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help="folder with meta.json and splits.json",
    )
    parser.add_argument(
        "--subjects-xls",
        default="data/raw/SC-subjects.xls",
        help="PhysioNet subject table with ages",
    )
    parser.add_argument(
        "--out", default="results/jinin/review", help="folder for the result CSVs"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Command: python -m src.eeg_review {' '.join(sys.argv[1:])}")
    print(f"MNE {mne.__version__}, NumPy {np.__version__}, xlrd {xlrd.__version__}")

    run_alignment_checks(Path(args.raw), out_dir)
    run_split_checks(Path(args.processed), out_dir)
    run_age_checks(Path(args.processed), Path(args.subjects_xls), out_dir)


if __name__ == "__main__":
    main()
