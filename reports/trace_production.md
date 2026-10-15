# Trace of HousePriceModel (production)

- Registered model version : **v3**  (aliases: ['production'])
- Lifecycle state tag      : production
- Source MLflow run        : `4f2298b134e24e1b80920c6686273728`  (run name: gbr)
- Algorithm / candidate    : gbr - Gradient boosting regressor
- Metrics                  : val_rmse=48841, test_rmse=49588, test_r2=0.812
- Git commit (code)        : `607608c57832cc59622a75cc6e40b24a0e5c27ae`  (uncommitted changes at training time: False)
- Dataset version tag      : untagged
- Dataset md5 (DVC hash)   : `d1c47305887e2252bf1ccbd74ff159a6`  rows=20640 cols=10
- Git commit(s) whose .dvc file recorded that hash : f800451

## Is my current checkout the same code and data?

- Current Git commit : `28b20a4956add4550b31885a6e9e62d9b92d3629`  -> different
- Current .dvc md5   : `ad9f215fb18c5671ff4082c6c2fad273`  -> different dataset (or no .dvc file in this checkout)

## Commands to recover exactly what produced this model

```bash
git switch -c investigate-v3 607608c57832cc59622a75cc6e40b24a0e5c27ae      # training code at that time
git checkout f800451 -- data/dataset.csv.dvc   # restore the dataset metadata (md5 d1c47305...)
dvc pull                                                  # fetch that dataset from the DVC remote
python src/dataset_tools.py verify                        # file on disk == hash in .dvc ?
mlflow ui --backend-store-uri sqlite:///mlflow.db         # inspect run 4f2298b1... (params, plots)
```