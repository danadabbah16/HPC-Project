#!/usr/bin/env bash
# Download Sleep-EDF Expanded (sleep-cassette study) from PhysioNet.
# Run on the LOGIN node (compute nodes usually have no internet).
# The data is NOT committed to the repo: keep data/ in .gitignore.
#
# Usage:
#   bash scripts/download_data.sh [DEST_DIR] [N_SUBJECTS]
#   bash scripts/download_data.sh data/raw        # everything (~8 GB, 153 nights)
#   bash scripts/download_data.sh data/raw 10     # first 10 subjects only
set -euo pipefail

DEST="${1:-data/raw}"
NSUBJ="${2:-all}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0"

mkdir -p "$DEST"
cd "$DEST"

# The checksum file doubles as the list of every file in the dataset.
wget -q -N "$BASE/SHA256SUMS.txt"

# Keep only sleep-cassette EDF files; optionally restrict to the first N subjects.
# File names look like SC4ssNE0-PSG.edf, where ss = subject, N = night.
grep "sleep-cassette/.*\.edf$" SHA256SUMS.txt | awk '{print $2}' | sort > all_files.txt
if [ "$NSUBJ" = "all" ]; then
    cp all_files.txt files.txt
else
    subjects=$(sed -E 's#.*/SC4([0-9]{2}).*#\1#' all_files.txt | sort -u | head -n "$NSUBJ" | paste -sd'|')
    grep -E "/SC4($subjects)" all_files.txt > files.txt
fi
echo "Downloading $(wc -l < files.txt) files into $(pwd)"

# -c resumes partial downloads, -N skips files that are already up to date.
while read -r f; do
    wget -q -c -N -P sleep-cassette "$BASE/$f" && echo "ok  $f"
done < files.txt

# Verify integrity of what we downloaded.
grep -F -f files.txt SHA256SUMS.txt | sha256sum -c --quiet && echo "All checksums OK"
