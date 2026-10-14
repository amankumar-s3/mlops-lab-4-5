#!/usr/bin/env bash
# B2 + B3 + B4: init DVC, configure a local remote, track dataset v1, push.
#
# NOTE ON ORDER: the remote is configured and committed BEFORE the first dataset version is committed
# and tagged. .dvc/config lives in Git, so an older commit that predates the remote setting would not
# know where to `dvc pull` from when you check it out later (recovery would fail with
# "No remote provided and no default remote set").
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f data/dataset.csv ] || { echo "data/dataset.csv missing - run scripts/01_get_data.sh"; exit 1; }

# ---- B2: initialise DVC
[ -d .dvc ] || dvc init
dvc config core.analytics false
git status --short | grep -v '^??' || true
ls -a .dvc
git commit -m "Initialize DVC" || true

# ---- B4 (part 1): remote storage OUTSIDE the repository (use an approved shared/cloud remote for teams)
mkdir -p ../dvc-storage
dvc remote list | grep -q localremote || dvc remote add -d localremote ../dvc-storage
git add .dvc/config
git commit -m "Configure DVC remote" || true

# ---- B3: track the dataset
dvc add data/dataset.csv
echo "---- data/dataset.csv.dvc ----"; cat data/dataset.csv.dvc
git status --short | grep -v '^??' || true
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC (dataset v1)"
git tag -f dataset-v1

# ---- B4 (part 2): push the data to the remote
dvc push
dvc status -c

python src/dataset_tools.py info --tag dataset-v1 --note "Baseline: raw Kaggle CSV"
git log --oneline --decorate
