
import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from house_explainer.dataprep.transformations import transformation


def train_regressor(train_data: pd.DataFrame, run_id: str) -> Any:
    (train_data, new_columns) = transformation(train_data)

    feature_cols = train_data.columns
    feature_cols = feature_cols.drop(['Id', 'SalePrice']).to_list()
    X_train_raw, y_train_raw = train_data[feature_cols], train_data['SalePrice']
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.30, random_state=42)

    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'metric': ['rmse', 'mape'],
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        "max_depth": 8,
        "num_leaves": 31,
        "max_bin": 512,
        "force_row_wise": "true",
        "verbose": -1,
        "num_iterations": 1000
    }

    # using log/exp transformation to convert target variable price to/back log-valued
    log_lgb_model = TransformedTargetRegressor(LGBMRegressor(n_jobs=-1, **lgb_params),
                                               func=np.log1p, inverse_func=np.expm1)
    log_lgb_model.fit(X=X_train, y=y_train, eval_set=[(X_val, y_val)],
                      eval_metric=['rmse'],
                      callbacks=[early_stopping(stopping_rounds=100, first_metric_only=True)])

    # Evaluate model
    y_val_pred = log_lgb_model.predict(X_val)
    rmse_test = round(mean_squared_log_error(y_val_pred, y_val) ** 0.5, 5)
    print(f'The rmse of prediction is: {rmse_test}')
    plt.scatter(y_val, y_val_pred)
    plt.xlabel("Expected")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Expected")
    plt.show()

    print(f"Logging the pipeline to MLflow for run_id: {run_id}")

    temp_dir = f'/tmp/devfest-2023/{run_id}'
    os.makedirs(temp_dir)
    X_train.to_parquet(f'{temp_dir}/background_sets.parquet')
    mlflow.log_artifacts(f'{temp_dir}', "background_sets")
    mlflow.log_metric('rmse_test', rmse_test)
    mlflow.sklearn.log_model(sk_model=log_lgb_model,
                             artifact_path='log_lgb_model',
                             registered_model_name='log_lgb_model',
                             await_registration_for=900)

    return log_lgb_model
