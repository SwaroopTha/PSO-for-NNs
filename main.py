import torch.nn as nn
from models.mlp import MLP
from data import load_iris_data
from trainers.train_pso import train_pso

if __name__ == "__main__":
    input_dim = 4       # Iris features
    hidden_dims = [16, 16]
    output_dim = 3      # Iris classes

    model = MLP(input_dim, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    train_loader = load_iris_data(batch_size=32)

    best_params = train_pso(model, train_loader, criterion, num_iterations=100)

    from models.pso_nn_wrapper import PSONNWrapper
    wrapper = PSONNWrapper(model, criterion)
    final_loss = wrapper.evaluate(best_params, train_loader)
    print(f"\nFinal Evaluation Loss: {final_loss:.4f}")