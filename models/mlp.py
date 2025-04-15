import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # No activation after output layer
                layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)