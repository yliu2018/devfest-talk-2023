import pickle
import warnings

import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

MODEL_ARTIFACT_PATH = 'house_explainer_model'


def log_explainer_model(experiment_id, trained_model_run_id, pipeline_run_name):
    # we could save any arbitrary binary or key-value pairs, which can then be used for a customized pyfunc model
    background_set_path = f"s3://mlflow/{experiment_id}/{trained_model_run_id}/"\
        "artifacts/background_sets"
    artifacts = {"background_set_path": background_set_path}

    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        print(trained_model_run_id)
        trained_model_uri = f'runs:/{trained_model_run_id}/model'
        print(trained_model_uri)

        explainer_pipeline_uri = f'runs:/{mlrun.info.run_id}/{MODEL_ARTIFACT_PATH}'
        print(explainer_pipeline_uri)
        mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH,
                                python_model=ExplainerPipeline(
                                    trained_model_uri, trained_model_run_id),
                                registered_model_name=MODEL_ARTIFACT_PATH,
                                artifacts=artifacts,
                                await_registration_for=900
                                )

        print("trained model uri is: %s", trained_model_uri)
        print("explainer_pipeline_uri is: %s", explainer_pipeline_uri)

        mlflow.log_param("trained_model_uri", trained_model_uri)
        mlflow.log_param("explainer_pipeline_uri", explainer_pipeline_uri)

        mlflow.set_tag('pipeline_step', __file__)

    client = MlflowClient()
    latest_model_info = client.get_latest_versions(
        MODEL_ARTIFACT_PATH, stages=['None'])
    print(latest_model_info)
    latest_model_uri = f"models:/{MODEL_ARTIFACT_PATH}/{latest_model_info[0].version}"
    print(f"latest explainer model version uri is: {latest_model_uri}")

    return mlrun.info.run_id


class ExplainerPipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.background_set_path = context.artifacts['background_set_path']
        print(f"get artifacts path from path {self.background_set_path}")
        return

    def __init__(self, trained_model_uri, trained_model_run_id):
        self.trained_model_uri = trained_model_uri
        self.trained_model_run_id = trained_model_run_id
        self.trained_model = mlflow.sklearn.load_model(self.trained_model_uri)

    def dynamic_explainer(self, df_input):
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.3f}".format(x)})

        background_set = (pd.read_parquet(
            f"{self.background_set_path}/background_sets.parquet")).sample(100)
        explainer = shap.KernelExplainer(
            self.trained_model.predict, background_set)
        baseline = float(explainer.expected_value)
        print("{0:0.3f}".format(baseline))

        # for simplicity, just do one record's explaining
        shap_values = explainer.shap_values(
            df_input.tail(1), check_additivity=False)

        print(shap_values)
        print(self.trained_model.predict(df_input.tail(1)))
        shap.bar_plot(shap_values[0],
                      feature_names=self.trained_model.regressor_.feature_name_)

        return pickle.dumps(shap_values)

    def predict(self, context, model_input):
        results = self.dynamic_explainer(model_input)
        return results
