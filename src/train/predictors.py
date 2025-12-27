

def get_fully_connected(number_of_hidden_layers: int, input_parameter_count: int, neurons_per_layer: int):
    assert number_of_hidden_layers >= 1, "number of hidden layers must be greater than or equal to 1"

    # noinspection PyPackageRequirements
    import torch.nn as nn

    # initialize layer list with input -> first hidden
    layers = [nn.Linear(input_parameter_count, neurons_per_layer), nn.ReLU()]

    # hidden -> hidden
    for _ in range(number_of_hidden_layers - 1):
        layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
        layers.append(nn.ReLU())

    # last hidden -> output
    layers.append(nn.Linear(neurons_per_layer, 1))

    return nn.Sequential(*layers)