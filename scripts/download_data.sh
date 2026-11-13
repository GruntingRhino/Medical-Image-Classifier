#!/usr/bin/env bash
# Download and stage all datasets for the proxy-sampling study.
# Sources and licenses are documented in src/proxystudy/data/*.py docstrings.
#
# Usage: scripts/download_data.sh
# Requires: python env with proxystudy installed (.venv), kaggle CLI creds at
# ~/.kaggle/kaggle.json (for the Fitzpatrick17k image mirror), curl.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

mkdir -p data/raw data/processed

echo "=== 1/4 HAM10000 (Harvard Dataverse, CC BY-NC-SA 4.0) ==="
if [ ! -f data/raw/ham10000/metadata.tab ]; then
  echo "Place the HAM10000 release under data/raw/ham10000/"
  echo "Required: metadata.tab + 10,015 ISIC*.jpg (flat or in ham10000_images_part_1/2)"
  echo "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
else
  echo "metadata.tab present; $(find data/raw/ham10000 -name 'ISIC_*.jpg' | wc -l) images"
fi

echo "=== 2/4 Fitzpatrick17k metadata (Zenodo 12739457, quality re-host) ==="
cd data/raw/fitzpatrick17k 2>/dev/null || { mkdir -p data/raw/fitzpatrick17k && cd data/raw/fitzpatrick17k; }
for f in Fitzpatrick17k-C.csv Fitzpatrick17k_DiagnosisMapping.xlsx; do
  [ -f "$f" ] || curl -sL -o "$f" "https://zenodo.org/api/records/12739457/files/$f/content"
done
cd ../..

echo "=== 3/4 Fitzpatrick17k images (original Groh release via Kaggle mirror, CC BY-NC-SA 3.0) ==="
if [ ! -d data/raw/fitzpatrick17k_original/finalfitz17k ]; then
  .venv/bin/kaggle datasets download mobaswiralfarabi/fitzpatrick17k-original \
    -p data/raw/fitzpatrick17k_original --unzip
fi

echo "=== 4/4 Adult + German credit (UCI mirrors via the loaders) ==="
$PY -c "from proxystudy.data.adult import load_adult; d=load_adult(); print('adult ok', len(d['train']))" 
$PY -c "from proxystudy.data.german import load_german; print('german ok', len(load_german()))"

echo "=== checksums + split sealing ==="
$PY - <<'EOF'
from proxystudy.data import ham10000, fitzpatrick17k
from proxystudy.exp.ledger import file_checksum
from pathlib import Path
meta = ham10000.lesion_split(ham10000.load_metadata(), seed=20261111)
print("ham10000 test-split checksum:", ham10000.checksum_of_test_split(meta))
print("ham10000 metadata.tab sha256:", file_checksum(ham10000.RAW / "metadata.tab"))
print("ham10000 cache sha256:", file_checksum(ham10000.PROC / "images_224.npy"))
print("fitz17k C.csv sha256:", file_checksum(fitzpatrick17k.RAW / "Fitzpatrick17k-C.csv"))
print("fitz17k eval checksum:", fitzpatrick17k.checksum_eval())
print("fitz17k cache sha256:", file_checksum(fitzpatrick17k.PROC / "eval_malignant_benign_224.npy"))
EOF
echo "Done."
