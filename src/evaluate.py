"""Evaluate any MLflow model URI on the held-out TEST split and log the result.

    python src/evaluate.py --model-uri models:/HousePriceModel@production
    python src/evaluate.py --model-uri models:/HousePriceModel/3
"""
import argparse
import json
import re

import mlflow
import numpy as np

from common import ROOT, git_info, load_config, load_splits, regression_metrics, setup_mlflow


def evaluate_model(model_uri: str, cfg: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    model = mlflow.pyfunc.load_model(model_uri)
    X, y = load_splits(cfg)["test"]
    pred = np.asarray(model.predict(X)).ravel()
    return regression_metrics(y, pred), np.asarray(y), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", required=True)
    args = ap.parse_args()
    cfg = load_config()
    setup_mlflow(cfg)
    metrics, y, pred = evaluate_model(args.model_uri, cfg)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", args.model_uri).strip("_")
    report = {"model_uri": args.model_uri, "split": "test", "n_rows": int(len(y)), **metrics, **git_info()}
    (ROOT / "reports" / f"evaluation_{safe}.json").write_text(json.dumps(report, indent=2))
    with mlflow.start_run(run_name=f"evaluation_{safe}"):
        mlflow.set_tags({"purpose": "evaluation", "evaluated_model_uri": args.model_uri,
                         "git_commit": report["git_commit"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_dict(report, "evaluation_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
