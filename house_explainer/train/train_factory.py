import mlflow


def get_train_test_func(model_category: str):
    """A simple factory pattern to decide which trainer function to use

    Args:
        model_category (str): a string that represents the ML model we will train

    Returns:
        _type_: train_test function name
    """
    if model_category == "lightgbm":
        import house_explainer.train.lightgbm_trainer as lightgbm_trainer
        mlflow.sklearn.autolog(silent=True)
        mlflow.lightgbm.autolog(silent=True)
        return lightgbm_trainer
    else:
        raise _exception_factory(
            ValueError, "invalid model_category, not supported")


def _exception_factory(exception, message):
    return exception(message)
