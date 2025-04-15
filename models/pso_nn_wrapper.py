import torch

class PSONNWrapper:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.param_shapes = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
    
    def flatten_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()])
    
    def unflatten_params(self, flat_params):
        pointer = 0
        for i, param in enumerate(self.model.parameters()):
            param.data = flat_params[pointer:pointer+self.param_sizes[i]].view(self.param_shapes[i])
            pointer += self.param_sizes[i]
    
    def evaluate(self, flat_params, data_loader):
        self.unflatten_params(flat_params)
        total_loss = 0.0
        with torch.no_grad():
            for X, y in data_loader:
                outputs = self.model(X)
                total_loss += self.criterion(outputs, y).item() * len(y)
        return total_loss / len(data_loader.dataset)