"""Dataset helpers for Lab 5.

    python src/dataset_tools.py info   [--tag dataset-v1 --note "baseline"]   # rows/cols/size/md5 (+ log row)
    python src/dataset_tools.py make-v2        # controlled change 1: drop rows with missing numeric features
    python src/dataset_tools.py make-v3        # controlled change 2 (optional): drop capped-target rows
    python src/dataset_tools.py verify         # does data/dataset.csv match the md5 in the CURRENT .dvc file?
    python src/dataset_tools.py table          # print reports/dataset_versions.csv
"""
import argparse
import csv
import subprocess

import pandas as pd

from common import ROOT, dataset_summary, dvc_md5, git_info, load_config, md_table

cfg = load_config()
CSV = ROOT / cfg["data"]["path"]
LOG = ROOT / "reports" / "dataset_versions.csv"


def cmd_info(tag, note):
    s = dataset_summary(cfg)
    g = git_info()
    if tag:  # use the commit the tag points to, if the tag already exists
        tagged = subprocess.run(["git", "rev-parse", "--short=8", f"{tag}^{{commit}}"], cwd=ROOT,
                                capture_output=True, text=True)
        if tagged.returncode == 0:
            g["git_commit_short"] = tagged.stdout.strip()
    print(f"rows={s['dataset_rows']}  cols={s['dataset_cols']}  size={s['dataset_size_bytes']} bytes  "
          f"nulls={s['dataset_nulls']}  md5={s['dataset_md5']}")
    if tag:
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["Version", "Git commit", "Rows", "Columns", "Size (bytes)", "MD5", "Change description"])
            w.writerow([tag, g["git_commit_short"], s["dataset_rows"], s["dataset_cols"],
                        s["dataset_size_bytes"], s["dataset_md5"], note])
        print(f"logged {tag} to {LOG.relative_to(ROOT)}")


def cmd_make(version):
    df = pd.read_csv(CSV)
    before = len(df)
    df = df.dropna(subset=cfg["data"]["numeric_features"])
    if version == "v3":
        df = df[df[cfg["data"]["target"]] < df[cfg["data"]["target"]].max()]
    df.to_csv(CSV, index=False)
    print(f"{version}: {before} -> {len(df)} rows ({before - len(df)} removed). {cfg['dataset_changes'][version]}")


def cmd_verify():
    expected, actual = dvc_md5(cfg), dataset_summary(cfg)["dataset_md5"]
    print(f".dvc metadata md5 : {expected}\nfile on disk md5  : {actual}")
    print("VERIFIED: the file matches the dataset version recorded in Git" if expected == actual
          else "MISMATCH: run `dvc checkout` / `dvc pull`")
    return expected == actual


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["info", "make-v2", "make-v3", "verify", "table"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.action == "info":
        cmd_info(a.tag, a.note)
    elif a.action.startswith("make"):
        cmd_make(a.action[-2:])
    elif a.action == "verify":
        raise SystemExit(0 if cmd_verify() else 1)
    else:
        print(md_table(pd.read_csv(LOG)))
