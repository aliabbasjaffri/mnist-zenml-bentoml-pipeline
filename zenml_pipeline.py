import bentoml
from model import SimpleConvNet
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from train import train, test_model, cross_validate
from zenml.pipelines import pipeline
from zenml.steps import step, BaseStepConfig
from zenml.environment import Environment
from zenml.integrations.mlflow.mlflow_environment import MLFLOW_ENVIRONMENT_NAME


class TrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    k_folds: int = 2
    lr: float = 0.001


@enable_mlflow
@step
def cross_validate_dataset(config: TrainerConfig) -> dict:
    return cross_validate(
        epochs=config.epochs, k_folds=config.k_folds, learning_rate=config.lr
    )


@enable_mlflow
@step
def train_model(config: TrainerConfig) -> SimpleConvNet:
    return train(epochs=config.epochs, learning_rate=config.lr)


@enable_mlflow
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


@pipeline(enable_cache=False)
def mnist_pipeline(_cross_validator, _trainer, _test_model, _save_model):
    """Links all the steps together in a pipeline"""
    cv_results = _cross_validator()
    model = _trainer()
    test_results = _test_model(model=model)
    _save_model(cv_results=cv_results, test_results=test_results, model=model)


if __name__ == "__main__":
    # Run the pipeline

    configs = [
        {"epochs": 1, "k_folds": 2, "lr": 0.0003},
        {"epochs": 2, "k_folds": 2, "lr": 0.0004},
    ]

    for config in configs:
        pipeline_def = mnist_pipeline(
            _cross_validator=cross_validate_dataset(
                config=TrainerConfig(
                    epochs=config["epochs"], k_folds=config["k_folds"], lr=config["lr"]
                )
            ),
            _trainer=train_model(
                config=TrainerConfig(epochs=config["epochs"], lr=config["lr"])
            ),
            _test_model=test_model_performance(),
            _save_model=_save_model(),
        )
        pipeline_def.run()

    mlflow_env = Environment()[MLFLOW_ENVIRONMENT_NAME]
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {mlflow_env.tracking_uri}\n"
        "To inspect your experiment runs within the mlflow ui.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
