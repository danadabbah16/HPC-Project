"""Sleep-EDF preprocessing: EDF -> filtered, labelled 30-second epochs.

For every night it reads one EEG channel, band-pass filters it, cuts it into
30 s epochs, attaches the expert sleep-stage label to each epoch and saves
one compressed .npz file per recording:

    x : float32, shape (n_epochs, 3000)   # 30 s at 100 Hz, z-scored per night
    y : int64,   shape (n_epochs,)        # 0=W 1=N1 2=N2 3=N3 4=REM

Usage:
    python -m src.data.preprocess --raw data/raw/sleep-cassette --out data/processed
"""
import argparse
import json
import time
from pathlib import Path

import mne
import numpy as np

SFREQ = 100                  # Hz, EEG sampling rate in Sleep-EDF
EPOCH_SEC = 30
EPOCH_LEN = SFREQ * EPOCH_SEC  # 3000 samples
CHANNEL = "EEG Fpz-Cz"
BANDPASS = (0.3, 35.0)       # Hz
WAKE_MARGIN = 60             # epochs (30 min) of wake kept before/after sleep

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
# R&K stages 3 and 4 are merged into N3 (AASM standard).
# "Movement time" and "?" are unscored and get dropped.
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def find_pairs(raw_dir):
    """Return [(record_id, psg_path, hypnogram_path)] for every night found."""
    raw_dir = Path(raw_dir)
    pairs = []
    for psg in sorted(raw_dir.glob("*-PSG.edf")):
        rec = psg.name[:6]                      # e.g. SC4001 = subject 00, night 1
        hyp = sorted(raw_dir.glob(f"{rec}*-Hypnogram.edf"))
        if hyp:
            pairs.append((rec, psg, hyp[0]))
        else:
            print(f"WARNING: no hypnogram for {psg.name}, skipped")
    return pairs


def load_recording(psg_path, hyp_path, channel=CHANNEL, trim_wake=True):
    """Read one night and return (x, y) as described in the module docstring."""
    raw = mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose="ERROR")
    assert int(raw.info["sfreq"]) == SFREQ, f"unexpected sampling rate in {psg_path}"
    raw.filter(*BANDPASS, verbose="ERROR")
    signal = raw.get_data()[0] * 1e6            # volts -> microvolts

    # One label per 30 s epoch; -1 = unscored.
    n_epochs = len(signal) // EPOCH_LEN
    labels = np.full(n_epochs, -1, dtype=np.int64)
    for ann in mne.read_annotations(hyp_path):
        stage = STAGE_MAP.get(ann["description"])
        if stage is None:
            continue
        start = int(round(ann["onset"] / EPOCH_SEC))
        stop = start + int(round(ann["duration"] / EPOCH_SEC))
        labels[start:min(stop, n_epochs)] = stage

    x = signal[: n_epochs * EPOCH_LEN].reshape(n_epochs, EPOCH_LEN)

    # Recordings span ~20 h, mostly daytime wake. Keep only the sleep period
    # plus 30 min of wake on each side, otherwise Wake dominates the dataset.
    if trim_wake:
        asleep = np.where(labels > 0)[0]
        if len(asleep):
            lo = max(0, asleep[0] - WAKE_MARGIN)
            hi = min(n_epochs, asleep[-1] + WAKE_MARGIN + 1)
            x, labels = x[lo:hi], labels[lo:hi]

    keep = labels >= 0
    x, labels = x[keep], labels[keep]

    # Per-recording z-score: removes differences in amplitude between subjects.
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32), labels


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", default="data/raw/sleep-cassette")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--channel", default=CHANNEL)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.raw)
    print(f"Found {len(pairs)} recordings in {args.raw}")

    t0 = time.time()
    summary = {}
    for rec, psg, hyp in pairs:
        x, y = load_recording(psg, hyp, args.channel)
        np.savez_compressed(out / f"{rec}.npz", x=x, y=y)
        counts = np.bincount(y, minlength=5).tolist()
        summary[rec] = {"subject": rec[3:5], "n_epochs": len(y), "counts": counts}
        print(f"{rec}: {len(y):5d} epochs  {dict(zip(STAGE_NAMES, counts))}")

    total = np.sum([s["counts"] for s in summary.values()], axis=0)
    meta = {
        "channel": args.channel, "sfreq": SFREQ, "epoch_sec": EPOCH_SEC,
        "bandpass_hz": BANDPASS, "stage_names": STAGE_NAMES,
        "n_recordings": len(summary),
        "n_subjects": len({s["subject"] for s in summary.values()}),
        "n_epochs": int(total.sum()),
        "class_counts": dict(zip(STAGE_NAMES, total.tolist())),
        "preprocess_seconds": round(time.time() - t0, 1),
        "recordings": summary,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTotal: {meta['n_epochs']} epochs from {meta['n_subjects']} subjects "
          f"in {meta['preprocess_seconds']} s")
    print("Class counts:", meta["class_counts"])


if __name__ == "__main__":
    main()
