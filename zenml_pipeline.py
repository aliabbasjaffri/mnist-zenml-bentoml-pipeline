import bentoml
from model import SimpleConvNet
from train import train, test_model, cross_validate
from zenml.pipelines import pipeline
from zenml.steps import step

NUM_EPOCHS = 1
K_FOLDS = 2


@step
def cross_validate_dataset() -> dict:
    return cross_validate(epochs=NUM_EPOCHS, k_folds=K_FOLDS)


@step
def train_model() -> SimpleConvNet:
    return train(epochs=NUM_EPOCHS)


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
    p = mnist_pipeline(
        _cross_validator=cross_validate_dataset(),
        _trainer=train_model(),
        _test_model=test_model_performance(),
        _save_model=_save_model(),
    )
    p.run()
