#!/usr/bin/env bash
# A2 / B2: create the virtual environment, install tools, initialise Git.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mlflow scikit-learn pandas matplotlib pyyaml tabulate dvc kaggle
python -m pip freeze > requirements.txt

echo "---- tool versions (put these in the report) ----"
python --version
mlflow --version
dvc --version
git --version
python -c "import sklearn, pandas; print('scikit-learn', sklearn.__version__, '| pandas', pandas.__version__)"

[ -d .git ] || git init -b main
if [ -z "$(git config user.email || true)" ]; then
  echo "!! Set your Git identity first:  git config --global user.name 'Your Name'; git config --global user.email 'you@example.com'"
fi
echo
echo "Done. In EVERY new terminal run:  source .venv/bin/activate"
