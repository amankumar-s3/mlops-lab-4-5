"""Train candidate models and log everything to MLflow.

Examples
  python src/train.py                         # all candidates in configs/config.yaml
  python src/train.py --candidate gbr         # one candidate
  python src/train.py --candidate rf_large --use-processed --save-local --purpose pipeline
  python src/train.py --candidate gbr --register --run-name gbr_on_new_data
"""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from common import (ROOT, build_pipeline, dataset_summary, dvc_md5, git_info, load_config,
                    load_data, load_splits, regression_metrics, save_plots, setup_mlflow)

mlflow.sklearn.autolog(disable=True)  # we log explicitly here; see autolog_demo.py for autolog


def log_model_compat(pipe, signature, example):
    """MLflow 3.x uses `name=`, MLflow 2.x uses `artifact_path=`."""
    try:
        return mlflow.sklearn.log_model(pipe, name="model", signature=signature, input_example=example,
                                        serialization_format="cloudpickle")
    except TypeError:
        return mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature,
                                        input_example=example, serialization_format="cloudpickle")


def run_candidate(cfg, name, cand, splits, args):
    X_tr, y_tr = splits["train"]
    pipe = build_pipeline(cfg, cand["model"], cand["params"])
    run_name = args.run_name or name

    with mlflow.start_run(run_name=run_name) as run:
        # ---- tags: traceability (Git + DVC + purpose)
        g, ds = git_info(), dataset_summary(cfg)
        mlflow.set_tags({**g, "purpose": args.purpose, "candidate": name,
                         "candidate_description": cand["description"],
                         "dataset_md5": ds["dataset_md5"],
                         "dvc_metadata_md5": str(dvc_md5(cfg)),
                         "dataset_rows": str(ds["dataset_rows"]),
                         "dataset_cols": str(ds["dataset_cols"])})
        # ---- dataset lineage in the MLflow UI ("Datasets" section of the run)
        try:
            df = load_data(cfg)
            mlflow.log_input(mlflow.data.from_pandas(df, source=(ROOT / cfg["data"]["path"]).as_uri(),
                                                     name="house_price_data",
                                                     targets=cfg["data"]["target"]), context="training")
        except Exception as e:  # lineage is nice-to-have, never block training
            print("dataset lineage not logged:", e)

        # ---- parameters
        mlflow.log_params({"model_type": cand["model"], **cand["params"],
                           "test_size": cfg["data"]["test_size"], "val_size": cfg["data"]["val_size"],
                           "random_state": cfg["data"]["random_state"],
                           "n_features": len(X_tr.columns)})

        # ---- fit + metrics for every split
        pipe.fit(X_tr, y_tr)
        report = {}
        for split, (X, y) in splits.items():
            m = regression_metrics(y, pipe.predict(X))
            report[split] = m
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})

        # ---- artifacts (>= 2 required): plots, config snapshot, evaluation report
        X_val, y_val = splits["val"]
        for p in save_plots(pipe, y_val, pipe.predict(X_val), run_name):
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.log_dict(report, "evaluation_report.json")
        mlflow.log_dict({"candidate": name, **cand, "data": cfg["data"]}, "config_snapshot.yaml")
        lock = ROOT / "dvc.lock"
        if lock.exists():
            mlflow.log_artifact(str(lock), artifact_path="dvc")

        # ---- model + signature + input example
        example = X_tr.dropna().head(5)
        signature = infer_signature(example, pipe.predict(example))
        info = log_model_compat(pipe, signature, example)

        print(f"[{run_name}] run_id={run.info.run_id}  val_rmse={report['val']['rmse']:.0f}  "
              f"val_r2={report['val']['r2']:.3f}  test_rmse={report['test']['rmse']:.0f}")

        if args.save_local:  # used by the DVC pipeline (outs + metrics)
            (ROOT / "models").mkdir(exist_ok=True)
            joblib.dump(pipe, ROOT / "models" / "model.joblib")
            (ROOT / "reports").mkdir(exist_ok=True)
            (ROOT / "reports" / "metrics.json").write_text(json.dumps(
                {"mlflow_run_id": run.info.run_id, "git_commit": g["git_commit"],
                 "dataset_md5": ds["dataset_md5"], **{f"{s}_{k}": v for s, m in report.items()
                                                      for k, v in m.items()}}, indent=2))

        if args.register:
            name_reg = cfg["mlflow"]["registered_model_name"]
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", name_reg)
            client = mlflow.MlflowClient()
            for k, v in {"source_run_id": run.info.run_id, "candidate": name, "git_commit": g["git_commit"],
                         "dataset_md5": ds["dataset_md5"], "dataset_version": g["dataset_version"],
                         "val_rmse": f"{report['val']['rmse']:.2f}",
                         "test_rmse": f"{report['test']['rmse']:.2f}",
                         "lifecycle_state": "registered"}.items():
                client.set_model_version_tag(name_reg, mv.version, k, v)
            client.update_model_version(name_reg, mv.version,
                                        f"{cand['description']} trained on {g['dataset_version']} "
                                        f"(md5 {ds['dataset_md5'][:8]}), commit {g['git_commit_short']}")
            print(f"registered {name_reg} version {mv.version}")
    return run.info.run_id, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="all")
    ap.add_argument("--use-processed", action="store_true")
    ap.add_argument("--save-local", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--purpose", default="candidate_training")
    args = ap.parse_args()

    cfg = load_config()
    setup_mlflow(cfg)
    splits = load_splits(cfg, args.use_processed)
    names = list(cfg["candidates"]) if args.candidate == "all" else [args.candidate]
    rows = []
    for n in names:
        rid, rep = run_candidate(cfg, n, cfg["candidates"][n], splits, args)
        rows.append({"candidate": n, "run_id": rid, **{f"val_{k}": v for k, v in rep["val"].items()},
                     **{f"test_{k}": v for k, v in rep["test"].items()}})
    if len(rows) > 1:
        t = pd.DataFrame(rows).sort_values("val_rmse")
        print("\nRuns ranked by validation RMSE (selection criterion):")
        print(t.round(3).to_string(index=False))
        t.to_csv(ROOT / "reports" / "candidate_comparison.csv", index=False)


if __name__ == "__main__":
    main()
