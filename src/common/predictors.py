from pathlib import Path

# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch
# noinspection PyPackageRequirements, PyUnresolvedReferences
import torch.nn as nn


def get_new_fully_connected_model(number_of_hidden_layers: int, number_of_features: int, neurons_per_layer: int) -> nn.Module:
    assert number_of_hidden_layers >= 1, "number of hidden layers must be greater than or equal to 1"

    # initialize layer list with input -> first hidden
    layers = [nn.Linear(number_of_features, neurons_per_layer), nn.ReLU()]

    # hidden -> hidden
    for _ in range(number_of_hidden_layers - 1):
        layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        layers.append(nn.ReLU())

    # last hidden -> output
    layers.append(nn.Linear(neurons_per_layer, 1))

    return nn.Sequential(*layers)


def load_fully_connected_model(weights_path: Path, number_of_hidden_layers: int, number_of_features: int, neurons_per_layer: int) -> nn.Module:
    model = get_new_fully_connected_model(number_of_hidden_layers, number_of_features, neurons_per_layer)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model