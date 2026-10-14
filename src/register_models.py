"""A5: register the 3 best candidate runs as versions 1, 2, 3 of one registered model.

Selection criterion: lowest validation RMSE. Versions are registered worst -> best so that
the best model ends up as the highest version number (v3).

    python src/register_models.py
Writes reports/model_versions_table.md / .csv
"""
import mlflow
import pandas as pd

from common import ROOT, load_config, md_table, setup_mlflow

cfg = load_config()
setup_mlflow(cfg)
client = mlflow.MlflowClient()
name = cfg["mlflow"]["registered_model_name"]

if client.search_model_versions(f"name='{name}'"):
    raise SystemExit(f"{name} already has versions. Delete the registry DB (rm mlflow.db) to start over.")

exp = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
runs = client.search_runs([exp.experiment_id], "tags.purpose = 'candidate_training'",
                          order_by=["attributes.start_time DESC"])
latest = {}                                    # keep the most recent run per candidate
for r in runs:
    latest.setdefault(r.data.tags["candidate"], r)
top3 = sorted(latest.values(), key=lambda r: r.data.metrics["val_rmse"])[:3]
top3 = top3[::-1]                              # worst -> best => v1 .. v3

try:
    client.create_registered_model(name, description=(
        "House-price regression model (Kaggle California Housing). "
        "Versions differ by algorithm/hyper-parameters; lifecycle is tracked with aliases + tags."))
except mlflow.exceptions.MlflowException:
    pass

rows = []
for r in top3:
    cand = r.data.tags["candidate"]
    mv = mlflow.register_model(f"runs:/{r.info.run_id}/model", name)
    m, p = r.data.metrics, r.data.params
    change = f"{r.data.tags['candidate_description']} ({', '.join(f'{k}={v}' for k, v in p.items() if k in cfg['candidates'][cand]['params'])})"
    tags = {"source_run_id": r.info.run_id, "candidate": cand, "val_rmse": f"{m['val_rmse']:.2f}",
            "val_r2": f"{m['val_r2']:.4f}", "test_rmse": f"{m['test_rmse']:.2f}",
            "test_r2": f"{m['test_r2']:.4f}", "git_commit": r.data.tags["git_commit"],
            "dataset_md5": r.data.tags["dataset_md5"], "dataset_version": r.data.tags["dataset_version"],
            "lifecycle_state": "registered",
            "intended_use": "Batch/online house-price estimates after passing acceptance rule"}
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.update_model_version(name, mv.version, (
        f"{change}. val RMSE={m['val_rmse']:.0f}, val R2={m['val_r2']:.3f}. "
        f"Source run {r.info.run_id}. Dataset md5 {r.data.tags['dataset_md5'][:8]}."))
    rows.append({"Version": f"v{mv.version}", "Source run ID": r.info.run_id, "Candidate": cand,
                 "val RMSE": round(m["val_rmse"]), "val R2": round(m["val_r2"], 3),
                 "test RMSE": round(m["test_rmse"]), "Change/description": change})
    print(f"registered {name} v{mv.version} <- {cand} ({r.info.run_id[:8]})")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "model_versions_table.csv", index=False)
(ROOT / "reports" / "model_versions_table.md").write_text(md_table(df))
print(df.to_string(index=False))
