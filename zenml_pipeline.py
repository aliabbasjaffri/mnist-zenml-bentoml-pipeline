from typing import Dict, Tuple
import bentoml
from model import SimpleConvNet
from train import train, test_model, cross_validate
from zenml.pipelines import pipeline
from zenml.steps import step
from torch.nn import Module

NUM_EPOCHS = 1
K_FOLDS = 2


@step()
def cross_validate_dataset():
    return cross_validate(epochs=NUM_EPOCHS, k_folds=K_FOLDS)


@step
def train_model() -> Module:
    return train(epochs=NUM_EPOCHS)


@step
def test_model_performance(model: SimpleConvNet):
    return test_model(model=model, _test_loader=None)


@step
def _save_model(
        # cv_results: Dict[int, float],
        # test_results: Tuple[int, int],
        model: SimpleConvNet
) -> None:
    # training related
    # correct = float(test_results[0])
    # total = test_results[1]

    # metadata = {
    #     "acc": float(correct) / total,
    #     "cv_stats": cv_results,
    # }

    # bentoml save model

    model_name = "pytorch_mnist"

    bentoml.pytorch.save(
        model_name,
        model,
#        metadata=metadata,
    )


@pipeline
def mnist_pipeline(_cross_validator, _trainer, _test_model, _save_model):
    """Links all the steps together in a pipeline"""
    cv_results = _cross_validator()
    model = _trainer()
    test_results = _test_model(model=model)
    # _save_model(cv_results, test_results, model)
    _save_model(model)


if __name__ == "__main__":
    # Run the pipeline
    p = mnist_pipeline(
        _cross_validator=cross_validate_dataset(),
        _trainer=train_model(),
        _test_model=test_model_performance(),
        _save_model=_save_model(),
    )
    p.run()
