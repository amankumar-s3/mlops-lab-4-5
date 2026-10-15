# Autologging inspection (run d94b3bd6360944aabf5933d72e68a26e)

MLflow version: 3.17.1

## Parameters logged automatically (56)

- `memory` = None
- `model` = RandomForestRegressor(max_depth=8, min_samples_leaf=5, n_estimators=50,
                      n_jobs=-1, random_state=42)
- `model__bootstrap` = True
- `model__ccp_alpha` = 0.0
- `model__criterion` = squared_error
- `model__max_depth` = 8
- `model__max_features` = 1.0
- `model__max_leaf_nodes` = None
- `model__max_samples` = None
- `model__min_impurity_decrease` = 0.0
- `model__min_samples_leaf` = 5
- `model__min_samples_split` = 2
- `model__min_weight_fraction_leaf` = 0.0
- `model__monotonic_cst` = None
- `model__n_estimators` = 50
- `model__n_jobs` = -1
- `model__oob_score` = False
- `model__random_state` = 42
- `model__verbose` = 0
- `model__warm_start` = False
- `pre` = ColumnTransformer(transformers=[('num',
                                 Pipeline(steps=[('imp',
                                                  SimpleImputer(strategy='median')),
                                                 ('sc', StandardScaler())]),
                                 ['longitude', 'latitude', 'housing_median_age',
                                  'total_rooms', 'total_bedrooms', 'population',
                                  'households', 'median_income']),
                                ('cat', OneHotEncoder(handle_unknown='ignore'),
                                 ['ocean_proximity'])])
- `pre__cat` = OneHotEncoder(handle_unknown='ignore')
- `pre__cat__categories` = auto
- `pre__cat__drop` = None
- `pre__cat__dtype` = <class 'numpy.float64'>
- `pre__cat__feature_name_combiner` = concat
- `pre__cat__handle_unknown` = ignore
- `pre__cat__max_categories` = None
- `pre__cat__min_frequency` = None
- `pre__cat__sparse_output` = True
- `pre__n_jobs` = None
- `pre__num` = Pipeline(steps=[('imp', SimpleImputer(strategy='median')),
                ('sc', StandardScaler())])
- `pre__num__imp` = SimpleImputer(strategy='median')
- `pre__num__imp__add_indicator` = False
- `pre__num__imp__copy` = True
- `pre__num__imp__fill_value` = None
- `pre__num__imp__keep_empty_features` = False
- `pre__num__imp__missing_values` = nan
- `pre__num__imp__strategy` = median
- `pre__num__memory` = None

## Metrics logged automatically (6)

- `Pipeline_score_X_val` = 0.7454
- `training_mean_absolute_error` = 37021.1351
- `training_mean_squared_error` = 2839659436.0654
- `training_r2_score` = 0.7872
- `training_root_mean_squared_error` = 53288.4550
- `training_score` = 0.7872

## Run-level artifacts

- `estimator.html`
- `metric_info.json`

## Tags (non-system)

- `purpose` = autolog_demo
- `estimator_name` = Pipeline
- `estimator_class` = sklearn.pipeline.Pipeline

## What autolog did NOT record (verify yourself)

- Git commit id: NOT logged
- Dataset hash/version (DVC): NOT logged
- Validation metrics for the split I defined (val_rmse): NOT logged unless a metric function is called
- Custom plots (residuals, feature importance): NOT logged
- Acceptance decision / business context: NOT logged