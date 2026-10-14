"""OPTIONAL fallback: create a synthetic CSV with the same schema as the Kaggle
California Housing Prices file, so every script can be tested without Kaggle access.
For the graded lab, use the real Kaggle dataset (scripts/01_get_data.sh).

    python scripts/make_sample_data.py data/dataset.csv
"""
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
rng = np.random.default_rng(0)
n = 6000
lon = rng.uniform(-124.3, -114.3, n)
lat = rng.uniform(32.5, 42.0, n)
age = rng.integers(1, 53, n).astype(float)
rooms = rng.gamma(4, 600, n).round()
bedrooms = (rooms * rng.uniform(0.15, 0.25, n)).round()
pop = rng.gamma(3, 450, n).round()
hh = (pop / rng.uniform(2, 4, n)).round()
inc = np.clip(rng.lognormal(1.2, 0.5, n), 0.5, 15)
ocean = rng.choice(["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"], n,
                   p=[0.44, 0.32, 0.13, 0.10, 0.01])
bonus = pd.Series(ocean).map({"<1H OCEAN": 20000, "INLAND": -30000, "NEAR OCEAN": 40000,
                              "NEAR BAY": 35000, "ISLAND": 90000}).to_numpy()
value = 45000 * inc + 900 * (42 - lat) * 10 - 300 * age + bonus + rng.normal(0, 30000, n) + 60000
value = np.clip(value, 15000, 500001).round()
df = pd.DataFrame({"longitude": lon.round(2), "latitude": lat.round(2), "housing_median_age": age,
                   "total_rooms": rooms, "total_bedrooms": bedrooms, "population": pop,
                   "households": hh, "median_income": inc.round(4), "median_house_value": value,
                   "ocean_proximity": ocean})
df.loc[rng.choice(n, 100, replace=False), "total_bedrooms"] = np.nan  # missing values like the real file
df.to_csv(out, index=False)
print(f"wrote {out}: {df.shape}")
