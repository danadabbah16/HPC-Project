# EEG review of the data pipeline

Author: Jinin (EEG & Neurophysiology Lead).

This note checks whether Dana's data pipeline (`scripts/download_data.sh`, `src/data/`, `docs/DATA.md`, group commit `1c3559f`) turns the raw sleep recordings into correct training data. The checks are in [`src/eeg_review.py`](../src/eeg_review.py). They ran as Slurm job 36130 (commit `4519cf3`), and their results are in `results/jinin/review/`. The terms are explained in the next section.

## Summary

**The data pipeline is correct. Nothing needs to change, and the baseline results stand.**

- Every expert label sits on the right piece of signal, in all 153 nights.
- The old sleep-stage labels were converted to the modern ones without losing or mixing any.
- Each person's recordings are in exactly one of train, validation or test.

There are two main findings. First, the test set contains much more deep sleep than the training set (section 2). Second, a few removed segments leave gaps in each night, which matters once the model looks at neighbouring segments (section 3). Section 4 lists small suggestions.

## Terms used in this note

| Term | Meaning |
|---|---|
| EEG | Electroencephalography: the brain's electrical activity, recorded with electrodes on the scalp. |
| Channel Fpz-Cz | The EEG signal we use. It is the voltage between an electrode on the forehead (Fpz) and one on the top of the head (Cz). It is sampled 100 times per second (100 Hz). |
| Hz, µV | Hertz (waves per second) measures how fast a brain wave is. Microvolts (millionths of a volt) measure how large it is. |
| PSG file | Polysomnography: the full recording of one night (EEG plus eye, muscle and breathing signals). |
| Hypnogram file | The sleep expert's labels for one night: one sleep stage for every 30 seconds. |
| Epoch | One 30-second piece of a night, the unit that experts label and our model classifies. This is not the same as a training epoch. |
| Sleep stages | W: awake. N1: very light sleep, while drifting off. N2: light sleep, most of the night. N3: deep sleep. REM: the stage with rapid eye movements, when most vivid dreams happen. |
| Slow waves | Large, slow brain waves (about 0.5 to 2 per second) that define deep sleep (N3). |
| Spindles | Short bursts of faster waves (11 to 16 Hz), typical of N2. |
| Delta, theta | Frequency bands: delta is 0.5 to 4 Hz (deep sleep), theta is 4 to 8 Hz (light sleep and REM). |
| R&K and AASM | Two rulebooks for labelling sleep. Sleep-EDF was labelled with the older R&K rules (1968). These split deep sleep into stages S3 and S4. The current AASM rules merge them into one stage, N3. |
| Sleep-EDF, Sleep Cassette | Sleep-EDF is the public dataset we use. Its Sleep Cassette part has 153 nights from 78 healthy people, aged 25 to 101, mostly two nights each. |
| Subject | One person in the dataset. |
| Split | Dividing the people into training (train), validation (val) and test sets. |
| Leakage | When information from the test set reaches training. It makes test scores look better than they really are. |
| Band-pass filter | Removes frequencies below and above a chosen range, here very slow drift and fast noise. |
| z-score | Rescales a signal to mean 0 and standard deviation 1, so recordings with different amplitudes become comparable. |
| Spearman ρ (rho) | How strongly two quantities rise or fall together, from −1 to +1. A negative value means one goes down when the other goes up. |
| p-value | The chance of seeing a relationship this strong if there were really none. Below 0.05 is usually called significant. |
| κ (kappa) | Agreement between the model and the expert, corrected for agreement by chance. |
| Context window | Giving the model one epoch together with its neighbouring epochs, as a human expert would look at them (planned in Task 2). |

## What was checked

| Check | Why it matters | Result |
|---|---|---|
| Each person's nights are in only one split | Two nights of the same person look very similar. If one is in train and the other in test, the model is tested on someone it already knows (leakage). | Checked for every recording listed in `splits.json`. No person is in two splits (54 / 12 / 12 people) |
| Old labels converted to new ones | The model must learn the five modern stages. | S1→N1, S2→N2, S3 and S4→N3, R→REM, W→W. The counts match exactly |
| Unusable epochs removed | Epochs labelled `Sleep stage ?` (not scored) or `Movement time` (the person moved) have no sleep stage. | Both removed (25,047 and 128 epochs) |
| Wake trimmed | Each recording lasts about 20 hours, mostly daytime wake. | 30 minutes of wake kept before and after sleep. No sleep epoch lost |
| Labels line up with the signal | If labels are shifted in time, the model learns from wrong examples without any error message. | Read from the header of every PSG and hypnogram file: signal and labels start at the same second in all 153 nights. Every label starts and ends on a 30-second boundary |
| Channel stated | The report must say which signal was used. | Fpz-Cz at 100 Hz, Sleep Cassette only. Stated in DATA.md |
| Filter keeps the important waves | Too narrow a filter would delete the waves that identify N2 and N3. | 0.3 to 35 Hz (MNE FIR filter, zero-phase), which keeps delta waves and spindles. The filter runs on the whole night before it is cut into epochs, and zero-phase means it does not shift the signal in time. In the sample data, spindle-band power is highest in N2 and theta in REM, as expected |
| No leakage through rescaling | Statistics from the test set must not shape training data. | Each night is z-scored using only its own values (section 4) |
| Stage counts per split | Very different splits make test results hard to compare. | Recounted from `meta.json`. Counts match DATA.md, but N3 differs between splits (section 2) |

## 1. Effect of the label conversion and wake trimming

| Original label | Epochs before processing | After processing |
|---|---|---|
| W | 285,937 | 65,941 W (after wake trimming) |
| S1 | 21,522 | 21,522 N1 |
| S2 | 69,132 | 69,132 N2 |
| S3 | 8,793 | 13,039 N3 |
| S4 | 4,246 | (merged with S3) |
| R | 25,835 | 25,835 REM |
| ? | 25,047 | removed |
| Movement time | 128 | removed |

Without trimming, wake would make up about 67% of all epochs. A model could then score high accuracy by always answering "awake". Trimming brings wake down to 33.7%.

S4 is a third of all deep sleep. Merging S3 and S4 keeps N3 as one class, which is already the rarest (6.7% of epochs).

## 2. The test set has more deep sleep, and age explains part of it

The test set has 1.7 times as much deep sleep (N3) as the training set: 9.9% of its epochs against 5.8%, with validation at 7.5%. The split is still correct. It is made by person, and with only 12 people in the test set, a few people can change the share.

Older people have much less deep sleep in this dataset. Across all 78 people, Spearman ρ is −0.48 with p < 0.0001, a clear relationship that is very unlikely to be chance. The people fall into four age groups, with no one in between:

| Age group | People | Median share of N3 | Train | Val | Test |
|---|---|---|---|---|---|
| 25 to 34 | 20 | 13.4% | 15 | 3 | 2 |
| 50 to 60 | 22 | 4.8% | 11 | 6 | 5 |
| 66 to 74 | 20 | 4.1% | 15 | 2 | 3 |
| 85 to 101 | 16 | 1.1% | 13 | 1 | 2 |

Sleep research has shown this decline of deep sleep with age before (Ohayon et al., 2004). Six people, aged 54 to 97, have no N3 at all.

Age alone does not explain the test set. The training set has more of the oldest people (13 of 54, against 2 of 12 in test), but most of the extra deep sleep in test comes from a few individuals:

- Subjects 18 and 00 (ages 28 and 33) spend 27% and 26% of the time in N3, among the highest values in the dataset.
- Three older test subjects have far more N3 than others their age: subject 60 (age 89, 14.3%), subject 43 (age 73, 13.6%) and subject 41 (age 66, 13.6%).

Inside the test set, age and deep sleep are only weakly related (ρ = −0.20, against −0.54 in train and −0.56 in validation).

The scoring rules may exaggerate the age effect. Both rulebooks count a slow wave only if it is larger than 75 µV, and slow waves get smaller with age. So some older people may be labelled with less N3 partly because their waves fall under this fixed size limit, in addition to sleeping less deeply.

What this means for the project:

- Keep the split. Everyone uses seed 42, and changing it would mean redoing every result.
- Treat N3 results on the test set with care, because they depend mostly on a handful of people. Per-stage F1 and κ say more than overall accuracy.
- Report results per person (Task 3), so that differences between people, including age, are visible.

## 3. Removed epochs leave hidden gaps in each night

Dana's code removes the `?` and `Movement time` epochs after trimming, and it does not save where each epoch originally was. Movement happens during the night, so wherever a movement epoch was removed, two epochs that are next to each other in the saved data were not next to each other in time.

The baseline model is not affected, because it looks at one epoch at a time. Context windows would be affected, because some would put together epochs that were not neighbours. At most about 128 of the 195,469 epochs are involved. The fix is to save each epoch's original position in the `.npz` files, so that windows can skip the gaps. The alternative is to accept the gaps and say so in the report.

## 4. Small suggestions

- The download script keeps going when a file fails to download, and only the final checksum test notices. This happened in one of our downloads. The script should stop with an error instead.
- `data/sample` contains 150 randomly chosen epochs per night, so its epochs are not in time order. That is fine for quick tests of the baseline, but not for testing context windows.
- Each test night is z-scored with its own mean and standard deviation. This uses no labels and nothing from other people, so it is not leakage, but the report should mention it.
- Package versions are not recorded yet, and the MNE version changes how the filter behaves. `src/eeg_review.py` also needs the `xlrd` package, which should be added to the pinned requirements.

## How to reproduce

From the repo root on the cluster, after Dana's three data steps:

```bash
wget -P data/raw https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls
mkdir -p results/jinin/review
sbatch slurm/jinin_eeg_review.sbatch
```

The job uses 1 CPU and 2 GB of memory and finishes in under 10 minutes. It involves no randomness, so it needs no seed. The unit tests run with `python -m pytest tests/test_eeg_review.py`.

## References

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Trans Biomed Eng. 2000;47(9):1185–1194. (Sleep-EDF dataset)
- Rechtschaffen A, Kales A. A manual of standardized terminology, techniques and scoring system for sleep stages of human subjects. 1968. (R&K rules)
- Berry RB, et al. The AASM Manual for the Scoring of Sleep and Associated Events. American Academy of Sleep Medicine. (AASM rules)
- Ohayon MM, Carskadon MA, Guilleminault C, Vitiello MV. Meta-analysis of quantitative sleep parameters from childhood to old age in healthy individuals. Sleep. 2004;27(7):1255–1273.
- Supratak A, Dong H, Wu C, Guo Y. DeepSleepNet: a model for automatic sleep stage scoring based on raw single-channel EEG. IEEE Trans Neural Syst Rehabil Eng. 2017;25(11):1998–2008. (30-minute wake margin)
