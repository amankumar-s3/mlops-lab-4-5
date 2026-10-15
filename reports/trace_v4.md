# Trace of HousePriceModel (v4)

- Registered model version : **v4**  (aliases: none)
- Lifecycle state tag      : registered
- Source MLflow run        : `008315287705416f8daf452d1e66c1f1`  (run name: gbr_on_dataset-v3)
- Algorithm / candidate    : gbr - Gradient boosting regressor
- Metrics                  : val_rmse=43139, test_rmse=44088, test_r2=0.796
- Git commit (code)        : `28b20a4956add4550b31885a6e9e62d9b92d3629`  (uncommitted changes at training time: False)
- Dataset version tag      : dataset-v3
- Dataset md5 (DVC hash)   : `ad9f215fb18c5671ff4082c6c2fad273`  rows=19475 cols=10
- Git commit(s) whose .dvc file recorded that hash : fd9c087

## Is my current checkout the same code and data?

- Current Git commit : `28b20a4956add4550b31885a6e9e62d9b92d3629`  -> SAME as training commit
- Current .dvc md5   : `ad9f215fb18c5671ff4082c6c2fad273`  -> SAME dataset

## Commands to recover exactly what produced this model

```bash
git switch -c investigate-v4 28b20a4956add4550b31885a6e9e62d9b92d3629      # training code at that time
git checkout fd9c087 -- data/dataset.csv.dvc   # restore the dataset metadata (md5 ad9f215f...)
dvc pull                                                  # fetch that dataset from the DVC remote
python src/dataset_tools.py verify                        # file on disk == hash in .dvc ?
mlflow ui --backend-store-uri sqlite:///mlflow.db         # inspect run 00831528... (params, plots)
```