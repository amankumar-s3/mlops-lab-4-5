# Signature demo for `runs:/4f2298b134e24e1b80920c6686273728/model`

MLflow 3.17.1

## Input schema

| column | type |
|---|---|
| longitude | DataType.double |
| latitude | DataType.double |
| housing_median_age | DataType.double |
| total_rooms | DataType.double |
| total_bedrooms | DataType.double |
| population | DataType.double |
| households | DataType.double |
| median_income | DataType.double |
| ocean_proximity | DataType.string |

## Output schema

`[Tensor('float64', (-1,))]`

## Experiments

- **1. Valid request**: ACCEPTED -> predictions [126317.0, 191625.0, 211775.0]
- **2. Missing column (median_income dropped)**: REJECTED -> MlflowException: Model is missing inputs ['median_income'].
- **3. Wrong type (text in numeric column)**: REJECTED -> MlflowException: Incompatible input types for column median_income. Can not safely convert str to float64.
- **4. Integer column where the signature says double (int64 -> float64)**: REJECTED -> MlflowException: Incompatible input types for column housing_median_age. Can not safely convert int64 to float64.
- **5. Extra unexpected column**: ACCEPTED -> predictions [126317.0, 191625.0, 211775.0]
- **6. Columns in a different order**: ACCEPTED -> predictions [126317.0, 191625.0, 211775.0]
- **7. Right schema, absurd values (negative income, age 9999)**: ACCEPTED -> predictions [113650.0, 251757.0, 274474.0]
- **8. Right schema, unseen category value**: ACCEPTED -> predictions [198217.0, 200375.0, 202138.0]

## Interpretation

A signature validates **structure** (names, types, optionality). Tests 2-3 show malformed requests being stopped before they reach the model. Tests 7-8 show what it cannot guarantee: sensible value ranges, unseen categories, distribution drift or prediction quality. Add input validation/monitoring for those.