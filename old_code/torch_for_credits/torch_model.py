import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        self.layers = torch.nn.ModuleList()
        prev_size = input_size

        for size in hidden_layers:
            self.layers.append(torch.nn.Linear(prev_size, size))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout_rate))
            prev_size = size

        self.layers.append(torch.nn.Linear(prev_size, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x