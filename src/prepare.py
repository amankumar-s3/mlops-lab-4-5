"""DVC pipeline stage 1: split the raw CSV into train/val/test CSVs (data/processed/)."""
from common import ROOT, load_config, load_data, split_data

cfg = load_config()
out = ROOT / cfg["data"]["processed_dir"]
out.mkdir(parents=True, exist_ok=True)
for name, part in zip(("train", "val", "test"), split_data(load_data(cfg), cfg)):
    part.to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(part)} rows -> {out / (name + '.csv')}")
