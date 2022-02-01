import random
from bentoml.pytorch import save
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Tuple

from datasource import get_mnist_dataset, _get_loader, get_loader
from model import SimpleConvNet

NUM_EPOCHS = 1
K_FOLDS = 5

# reproducible setup for testing
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def cross_validate(epochs=NUM_EPOCHS, k_folds=K_FOLDS) -> Dict[int, float]:
    results = {}
    dataset = get_mnist_dataset(is_train_dataset=True)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    print("--------------------------------")

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print(f"FOLD {fold}")
        print("--------------------------------")

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = _get_loader(dataset=dataset, dataset_sampler=train_subsampler)
        test_loader = _get_loader(dataset=dataset, dataset_sampler=test_subsampler)

        # Train this fold
        model = SimpleConvNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_function = CrossEntropyLoss()
        for epoch in range(epochs):
            train_epoch(model, optimizer, loss_function, train_loader, epoch)

        # Evaluation for this fold
        correct, total = test_model(model, test_loader)
        print("Accuracy for fold %d: %d %%" % (fold, 100.0 * correct / total))
        print("--------------------------------")
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {K_FOLDS} FOLDS")
    print("--------------------------------")
    _sum = 0.0
    for key, value in results.items():
        print(f"Fold {key}: {value} %")
        _sum += value

    print(f"Average: {_sum / len(results.items())} %")

    return results


def train_epoch(model, optimizer, loss_function, train_loader, epoch, device="cpu"):
    # Mark training flag
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % 499 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def train(epochs=NUM_EPOCHS, _device="cpu"):
    train_loader = get_loader(is_train_set=True)

    model = SimpleConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = CrossEntropyLoss()
    for epoch in range(epochs):
        train_epoch(model, optimizer, loss_function, train_loader, epoch, _device)
    return model


def test_model(
    model: SimpleConvNet, _test_loader: DataLoader = None, _device="cpu"
) -> Tuple[int, int]:
    _correct, _total = 0, 0

    if _test_loader is None:
        _test_loader = get_loader(is_train_set=False)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(_test_loader):
            inputs, targets = inputs.to(_device), targets.to(_device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _total += targets.size(0)
            _correct += (predicted == targets).sum().item()

    return _correct, _total


if __name__ == "__main__":
    cuda = False
    model_name = "pytorch_mnist"

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if K_FOLDS > 1:
        cv_results = cross_validate(NUM_EPOCHS, K_FOLDS)
    else:
        cv_results = {}

    test_loader = get_loader(is_train_set=False)
    trained_model = train(NUM_EPOCHS, device.type)
    correct, total = test_model(trained_model, test_loader, device.type)

    # training related
    metadata = {
        "acc": float(correct) / total,
        "cv_stats": cv_results,
    }

    # bentoml save model
    save(
        model_name,
        trained_model,
        metadata=metadata,
    )
