import torch.nn as nn
import torch
from models.mlp import MLP
from data.data import load_ucidata
from trainers.train_pso import train_pso
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_loader, test_loader, input_dim, output_dim = load_ucidata('iris')
    hidden_dims = [16]

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    variants = ['BPSO', 'PPSO', 'SGPSO', "PSOGSA", "GSA"]
    model_factory = lambda: MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)

    optimizers = {}
    histories = {}
    for variant in variants:
        model = model_factory()
        optimizer, best_params = train_pso(
            model, 
            train_loader, 
            criterion, 
            variant=variant, 
            plot_convergence=False
        )
        optimizers[variant] = optimizer
        histories[variant] = optimizer.history

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    for variant, history in histories.items():
        plt.plot(history, label=variant, alpha=0.8, linewidth=2)

    plt.title('PSO Variants Convergence Comparison', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save and show
    plt.savefig('pso_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()