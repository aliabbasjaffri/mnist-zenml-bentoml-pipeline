import bentoml
from model import SimpleConvNet
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from train import train, test_model, cross_validate
from zenml.pipelines import pipeline
from zenml.steps import step, BaseStepConfig
from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME
from mlflow import log_param


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    k_folds: int = 2
    lr: float = 0.001


@step
def cross_validate_dataset(config: TrainerConfig) -> dict:
    return cross_validate(epochs=config.epochs, k_folds=config.k_folds)


@step
def train_model(config: TrainerConfig) -> SimpleConvNet:
    return train(epochs=config.epochs, learning_rate=config.lr)


@step
def test_model_performance(model: SimpleConvNet) -> dict:
    return test_model(model=model, _test_loader=None)


@step
def _save_model(cv_results: dict, test_results: dict, model: SimpleConvNet) -> None:

    metadata = {
        "acc": float(test_results["correct"]) / test_results["total"],
        "cv_stats": cv_results,
    }

    # bentoml save model
    model_name = "pytorch_mist"

    bentoml.pytorch.save(
        model_name,
        model,
        metadata=metadata,
    )


@pipeline
def mnist_pipeline(_cross_validator, _trainer, _test_model, _save_model):
    """Links all the steps together in a pipeline"""
    cv_results = _cross_validator()
    model = _trainer()
    test_results = _test_model(model=model)
    _save_model(cv_results=cv_results, test_results=test_results, model=model)


if __name__ == "__main__":
    # Run the pipeline
    first_param_cv = TrainerConfig(epochs=1, k_folds=2)
    first_param_tr = TrainerConfig(epochs=1, lr=0.0003)
    log_param(key="cross_validation_param", value=first_param_cv)
    log_param(key="training_param", value=first_param_tr)
    p1 = mnist_pipeline(
        _cross_validator=cross_validate_dataset(config=first_param_cv),
        _trainer=train_model(config=first_param_tr),
        _test_model=test_model_performance(),
        _save_model=_save_model(),
    )
    p1.run()

    second_param_cv = TrainerConfig(epochs=2, k_folds=2)
    second_param_tr = TrainerConfig(epochs=2, lr=0.0004)
    log_param(key="cross_validation_param", value=second_param_cv)
    log_param(key="training_param", value=second_param_tr)
    p2 = mnist_pipeline(
        _cross_validator=cross_validate_dataset(config=second_param_cv),
        _trainer=train_model(config=second_param_tr),
        _test_model=test_model_performance(),
        _save_model=_save_model(),
    )
    p2.run()

    # mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
    # print(
    #     "Now run \n "
    #     f"    mlflow ui --backend-store-uri {mlflow_env.tracking_uri}\n"
    #     "To inspect your experiment runs within the mlflow ui.\n"
    #     "You can find your runs tracked within the `mlflow_example_pipeline`"
    #     "experiment. Here you'll also be able to compare the two runs.)"
    # )
