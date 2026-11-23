"""Unit tests for src/eeg_review.py on small synthetic data."""

from datetime import datetime

import mne
import numpy as np
import pytest

from src.data.preprocess import STAGE_MAP, STAGE_NAMES
from src.eeg_review import (
    check_label_alignment,
    compute_n3_share_per_subject,
    compute_start_offset_s,
    count_annotation_epochs,
    count_off_grid_times,
    count_stages_per_split,
    find_shared_subjects,
    get_subject_id,
    has_alignment_problem,
    read_edf_start,
    summarise_ages_per_split,
)

RECORDING_START = datetime(1989, 4, 24, 16, 13)
EDF_HEADER_LENGTH = 256


def write_fake_edf_header(edf_path, start_text: str) -> None:
    """Write a blank 256-byte EDF header with the given start date and time."""
    header = bytearray(b" " * EDF_HEADER_LENGTH)
    header[168:184] = start_text.encode("ascii")
    with open(edf_path, "wb") as edf_file:
        edf_file.write(header)


def test_stage_map_merges_s3_and_s4_into_n3():
    n3_label = STAGE_NAMES.index("N3")
    assert STAGE_MAP["Sleep stage 3"] == n3_label
    assert STAGE_MAP["Sleep stage 4"] == n3_label


def test_stage_map_drops_unscored_epochs():
    assert "Sleep stage ?" not in STAGE_MAP
    assert "Movement time" not in STAGE_MAP


def test_get_subject_id_reads_subject_not_night():
    assert get_subject_id("SC4012") == "01"
    assert get_subject_id("SC4821") == "82"


def test_get_subject_id_rejects_unexpected_name():
    with pytest.raises(ValueError):
        get_subject_id("ST7011")
    with pytest.raises(ValueError):
        get_subject_id("SC40")


def test_find_shared_subjects_accepts_clean_split():
    splits = {
        "train": ["SC4001", "SC4002"],
        "val": ["SC4011"],
        "test": ["SC4021", "SC4022"],
    }
    assert find_shared_subjects(splits) == []


def test_find_shared_subjects_detects_nights_in_different_splits():
    splits = {"train": ["SC4001", "SC4011"], "val": ["SC4012"], "test": ["SC4002"]}
    assert find_shared_subjects(splits) == ["00", "01"]


def test_count_off_grid_times_accepts_whole_epochs():
    assert count_off_grid_times(np.array([0.0, 30.0, 3600.0, 29.999])) == 0


def test_count_off_grid_times_finds_shifted_labels():
    assert count_off_grid_times(np.array([0.0, 45.0, 30.5])) == 2


def test_read_edf_start_reads_header(tmp_path):
    # Same start as the real SC4001E0-PSG.edf header.
    edf_path = tmp_path / "fake.edf"
    write_fake_edf_header(edf_path, "24.04.8916.13.00")
    assert read_edf_start(edf_path) == RECORDING_START


def test_read_edf_start_rejects_broken_header(tmp_path):
    edf_path = tmp_path / "broken.edf"
    write_fake_edf_header(edf_path, "not a date here")
    with pytest.raises(ValueError):
        read_edf_start(edf_path)


def test_compute_start_offset_s():
    hypnogram_start = datetime(1989, 4, 24, 16, 13, 10)
    assert compute_start_offset_s(RECORDING_START, hypnogram_start) == 10.0


def test_check_label_alignment_ignores_non_stage_labels():
    annotations = mne.Annotations(
        onset=[0.0, 60.0, 75.0],
        duration=[60.0, 30.0, 10.0],
        description=["Sleep stage W", "Sleep stage 1", "Movement time"],
    )
    alignment = check_label_alignment(RECORDING_START, RECORDING_START, annotations)
    assert alignment == {
        "start_offset_s": 0.0,
        "off_grid_onsets": 0,
        "off_grid_durations": 0,
    }
    assert not has_alignment_problem(alignment)


def test_check_label_alignment_finds_shifted_stage_label():
    annotations = mne.Annotations(
        onset=[0.0, 45.0],
        duration=[45.0, 30.0],
        description=["Sleep stage W", "Sleep stage 2"],
    )
    alignment = check_label_alignment(RECORDING_START, RECORDING_START, annotations)
    assert alignment["off_grid_onsets"] == 1
    assert alignment["off_grid_durations"] == 1
    assert has_alignment_problem(alignment)


def test_check_label_alignment_finds_late_hypnogram():
    annotations = mne.Annotations(
        onset=[0.0], duration=[30.0], description=["Sleep stage W"]
    )
    hypnogram_start = datetime(1989, 4, 24, 16, 13, 30)
    alignment = check_label_alignment(RECORDING_START, hypnogram_start, annotations)
    assert alignment["start_offset_s"] == 30.0
    assert has_alignment_problem(alignment)


def test_count_annotation_epochs():
    annotations = mne.Annotations(
        onset=[0.0, 60.0, 90.0],
        duration=[60.0, 30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 1", "Sleep stage W"],
    )
    assert count_annotation_epochs(annotations) == {
        "Sleep stage W": 3,
        "Sleep stage 1": 1,
    }


def make_fake_meta() -> dict:
    """Return a tiny meta.json with three subjects, one of them with two nights."""
    return {
        "recordings": {
            "SC4001": {"n_epochs": 10, "counts": [2, 1, 4, 2, 1]},
            "SC4002": {"n_epochs": 10, "counts": [4, 1, 4, 0, 1]},
            "SC4011": {"n_epochs": 5, "counts": [1, 1, 1, 1, 1]},
            "SC4021": {"n_epochs": 4, "counts": [1, 0, 1, 2, 0]},
        }
    }


FAKE_SPLITS = {"train": ["SC4001", "SC4002"], "val": ["SC4011"], "test": ["SC4021"]}


def test_count_stages_per_split_adds_up_recordings():
    counts_per_split = count_stages_per_split(make_fake_meta(), FAKE_SPLITS)
    assert counts_per_split["train"].tolist() == [6, 2, 8, 2, 2]
    assert counts_per_split["test"].tolist() == [1, 0, 1, 2, 0]


def test_count_stages_per_split_rejects_unknown_recording():
    splits = {"train": ["SC4091"], "val": [], "test": []}
    with pytest.raises(ValueError):
        count_stages_per_split(make_fake_meta(), splits)


def test_compute_n3_share_per_subject_pools_nights():
    n3_shares = compute_n3_share_per_subject(make_fake_meta())
    assert n3_shares["00"] == pytest.approx(2 / 20)
    assert n3_shares["02"] == pytest.approx(2 / 4)


def test_summarise_ages_per_split_counts_each_subject_once():
    ages = {"00": 30, "01": 60, "02": 90}
    rows = summarise_ages_per_split(FAKE_SPLITS, ages)
    assert rows[0] == ["train", 1, 30.0, 30.0, 30, 30]
    assert rows[2] == ["test", 1, 90.0, 90.0, 90, 90]
