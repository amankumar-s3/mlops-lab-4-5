#!/usr/bin/env bash
# Download the Kaggle dataset and place it at data/dataset.csv
# Needs a Kaggle API token: Kaggle -> Settings -> API -> "Create New Token" -> save as ~/.kaggle/kaggle.json
#   chmod 600 ~/.kaggle/kaggle.json     (NEVER commit this file)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if kaggle datasets download -d camnugent/california-housing-prices -p data/raw --unzip; then
  cp data/raw/housing.csv data/dataset.csv
else
  echo "Kaggle CLI failed. Manual alternative:"
  echo "  1) open https://www.kaggle.com/datasets/camnugent/california-housing-prices and click Download"
  echo "  2) unzip it and copy housing.csv to data/dataset.csv"
  exit 1
fi
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/dataset.csv")
print(df.shape); print(df.dtypes); print("missing values per column:"); print(df.isna().sum())
PY
echo "Check the dataset's license on its Kaggle page before publishing it anywhere (DVC keeps it out of Git anyway)."
