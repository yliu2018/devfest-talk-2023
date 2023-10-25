# %%
# run `pip install -e .` before executing this notebook
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import shap

from house_explainer.dataprep.transformations import transformation
from house_explainer.explain.explainer import log_explainer_model
from house_explainer.train.train_factory import get_train_test_func

# %%
train_data = pd.read_csv(
    "../data/house-prices-advanced-regression-techniques/train.csv")

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

EXPERIMENT_NAME = "devfest-2023"
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_tracking_uri("http://localhost")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id
print("experiment_id:", experiment_id)

# %%
run_name = 'house_model_train_test'
curr_model_to_build = 'lightgbm'
with mlflow.start_run(run_name=run_name) as run:
    training_run_id = run.info.run_id
    train_function = get_train_test_func(curr_model_to_build)
    trained_model = train_function.train_regressor(train_data, training_run_id)


# %%
# use logged regressor to load the model back to memory to do house sold_price prediction
trained_regressor_uri = f"runs:/{training_run_id}/log_lgb_model"
logged_regressor = mlflow.sklearn.load_model(trained_regressor_uri)

test_data = pd.read_csv(
    "../data/house-prices-advanced-regression-techniques/test.csv")
df_test, _ = transformation(test_data, False)
df_test = df_test.reindex(columns=logged_regressor.regressor_.feature_name_)
df_test_prediction = logged_regressor.predict(df_test)
print(df_test_prediction)

# %%
# create and log an explainer for the trained pricing model
explainer_run_name = 'explainer_model_creation'
explainer_run_id = log_explainer_model(
    experiment_id, training_run_id, explainer_run_name)
print(explainer_run_id)

# %%
# load the logged explainer to explain one test record
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

logged_explainer = mlflow.pyfunc.load_model(
    f"runs:/{explainer_run_id}/house_explainer_model")
shap_values = pickle.loads(logged_explainer.predict(df_test.head(1)))

print(shap_values)
shap.bar_plot(shap_values[0],
              feature_names=logged_regressor.regressor_.feature_name_)
