"""Part C: from a registered model back to code + data (what a teammate does to investigate a prediction).

    python src/trace_model.py --alias production
    python src/trace_model.py --version 4
Writes reports/trace_<...>.md
"""
import argparse

import subprocess

import mlflow

from common import ROOT, dvc_md5, git_info, load_config, setup_mlflow

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--alias")
g.add_argument("--version")
g.add_argument("--latest", action="store_true", help="highest registered version")
a = ap.parse_args()

cfg = load_config()
setup_mlflow(cfg)
client = mlflow.MlflowClient()
name = cfg["mlflow"]["registered_model_name"]
if a.latest:
    a.version = str(max(int(v.version) for v in client.search_model_versions(f"name='{name}'")))
mv = client.get_model_version_by_alias(name, a.alias) if a.alias else client.get_model_version(name, a.version)
run = client.get_run(mv.run_id)
t, m = run.data.tags, run.data.metrics
label = a.alias or f"v{a.version}"
cur = git_info()
md5 = t.get("dataset_md5", "")
# Which Git commit(s) recorded this exact dataset hash in the .dvc metadata? (pickaxe search)
def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


meta_path = cfg["data"]["path"] + ".dvc"
candidates = [c.split()[0] for c in _git("log", "--format=%h", f"-S{md5}", "--", meta_path).splitlines()] if md5 else []
# keep only commits whose .dvc file CONTAINS the hash (the pickaxe also lists commits that removed it)
found = [c for c in candidates if md5 in _git("show", f"{c}:{meta_path}")]
data_commit = found[-1] if found else None   # oldest commit that introduced this hash
cur_md5 = dvc_md5(cfg)
lines = [
    f"# Trace of {name} ({label})", "",
    f"- Registered model version : **v{mv.version}**  (aliases: {list(client.get_model_version(name, mv.version).aliases) or 'none'})",
    f"- Lifecycle state tag      : {mv.tags.get('lifecycle_state', '-')}",
    f"- Source MLflow run        : `{mv.run_id}`  (run name: {run.info.run_name})",
    f"- Algorithm / candidate    : {t.get('candidate')} - {t.get('candidate_description')}",
    f"- Metrics                  : val_rmse={m.get('val_rmse', float('nan')):.0f}, "
    f"test_rmse={m.get('test_rmse', float('nan')):.0f}, test_r2={m.get('test_r2', float('nan')):.3f}",
    f"- Git commit (code)        : `{t.get('git_commit')}`  (uncommitted changes at training time: {t.get('git_dirty')})",
    f"- Dataset version tag      : {t.get('dataset_version')}",
    f"- Dataset md5 (DVC hash)   : `{md5}`  rows={t.get('dataset_rows')} cols={t.get('dataset_cols')}",
    f"- Git commit(s) whose .dvc file recorded that hash : {', '.join(found) or 'none found'}",
    "",
    "## Is my current checkout the same code and data?", "",
    f"- Current Git commit : `{cur['git_commit']}`  -> {'SAME as training commit' if cur['git_commit'] == t.get('git_commit') else 'different'}",
    f"- Current .dvc md5   : `{cur_md5}`  -> {'SAME dataset' if cur_md5 == md5 else 'different dataset (or no .dvc file in this checkout)'}",
    "", "## Commands to recover exactly what produced this model", "", "```bash",
    f"git switch -c investigate-v{mv.version} {t.get('git_commit')}      # training code at that time",
]
if data_commit:
    lines += [f"git checkout {data_commit} -- {cfg['data']['path']}.dvc   # restore the dataset metadata (md5 {md5[:8]}...)",
              "dvc pull                                                  # fetch that dataset from the DVC remote"]
else:
    lines += ["# no commit recorded this dataset hash yet: commit the .dvc file (dvc add + git commit) first"]
lines += ["python src/dataset_tools.py verify                        # file on disk == hash in .dvc ?",
          f"mlflow ui --backend-store-uri sqlite:///mlflow.db         # inspect run {mv.run_id[:8]}... (params, plots)",
          "```"]
out = ROOT / "reports" / f"trace_{label}.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
