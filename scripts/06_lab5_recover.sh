#!/usr/bin/env bash
# B5 (steps 25-28): recover dataset v1 safely on a temporary branch, then return to the latest version.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Working tree has uncommitted changes. Commit or stash them first."; exit 1
fi
LATEST=$(git rev-parse --abbrev-ref HEAD)

echo "== Git history of the DVC metadata =="
git log --oneline --decorate -- data/dataset.csv.dvc
echo; echo "== Latest (current) dataset =="; python src/dataset_tools.py info

dvc push   # make sure the remote really holds every version before we delete the local cache
echo; echo "== Simulate a fresh clone: delete the file AND the local DVC cache =="
rm -f data/dataset.csv; rm -rf .dvc/cache

echo; echo "== Recover dataset v1 on a temporary branch =="
git switch -c recover-dataset-v1 dataset-v1
cat data/dataset.csv.dvc
dvc pull
python src/dataset_tools.py info
python src/dataset_tools.py verify
md5sum data/dataset.csv; wc -l data/dataset.csv; ls -l data/dataset.csv

echo; echo "== Return to the latest commit and recover the latest data too =="
git switch "$LATEST"
dvc pull
python src/dataset_tools.py info
python src/dataset_tools.py verify
git branch -D recover-dataset-v1
