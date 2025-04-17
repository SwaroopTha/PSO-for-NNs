import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.mlp import MLP
import sys
sys.path.append('./')
from data.data import load_ucidata
from trainers.train_pso import train_pso
from trainers.train_sgd_adam import train_gd  # <-- Import the GD trainer

def vector_to_state_dict(model, vector):
    state_dict = model.state_dict()
    param_shapes = [param.shape for param in state_dict.values()]
    param_sizes = [torch.tensor(shape).prod().item() for shape in param_shapes]

    new_state_dict = {}
    idx = 0
    for key, shape, size in zip(state_dict.keys(), param_shapes, param_sizes):
        param_vector = vector[idx:idx + size]
        new_state_dict[key] = param_vector.view(shape)
        idx += size

    return new_state_dict

if __name__ == "__main__":
    datasets = ['iris', 'wine', 'breast_cancer']
    pso_variants = ['BPSO', 'PPSO', 'SGPSO', 'PSOGSA', 'GSA']
    # pso_variants = []
    gd_variants = ['SGD', 'Adam']
    all_variants = pso_variants + gd_variants

    hidden_dims = [16]
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy_table = {}

    for dataset_name in datasets:
        print(f"\n=== Running on dataset: {dataset_name.upper()} ===")
        train_loader, test_loader, input_dim, output_dim = load_ucidata(dataset_name)
        model_factory = lambda: MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        histories = {}
        accuracy_table[dataset_name] = {}

        for variant in all_variants:
            model = model_factory().to(device)

            if variant in pso_variants:
                optimizer, best_params = train_pso(
                    model,
                    train_loader,
                    criterion,
                    variant=variant,
                    plot_convergence=False
                )
                # Load best weights
                state_dict = vector_to_state_dict(model, best_params)
                model.load_state_dict(state_dict)
                history = optimizer.history

            else:  # Gradient descent training
                model, history, acc_hist = train_gd(
                    model,
                    train_loader,
                    criterion,
                    optimizer_type=variant,
                    lr=0.01 if variant == "SGD" else 0.001,
                    num_epochs=500,
                    plot_convergence=False
                )

            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == y_batch).sum().item()
                    total += y_batch.size(0)
            final_accuracy = correct / total
            accuracy_table[dataset_name][variant] = final_accuracy

            # Store history for plotting
            cpu_history = [h.item() if torch.is_tensor(h) else h for h in history]
            histories[variant] = cpu_history

        # Plot convergence
        plt.figure(figsize=(10, 6))
        for variant, history in histories.items():
            plt.plot(history, label=variant, alpha=0.8, linewidth=2)
        plt.title(f'{dataset_name.capitalize()} - Optimizer Convergence', fontsize=14)
        plt.xlabel('Epoch / Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'loss_plot_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Accuracy Table
    header = ["Dataset"] + all_variants
    rows = []
    for dataset, accs in accuracy_table.items():
        row = [dataset] + [f"{accs[v]*100:.2f}%" for v in all_variants]
        rows.append(row)

    print("\n=== Final Accuracy Table ===")
    print("{:<15}{}".format("Dataset", "".join([f"{v:>10}" for v in all_variants])))
    for row in rows:
        print("{:<15}{}".format(row[0], "".join([f"{val:>10}" for val in row[1:]])))

    with open("final_accuracy_table.txt", "w") as f:
        f.write("{:<15}{}\n".format("Dataset", "".join([f"{v:>10}" for v in all_variants])))
        for row in rows:
            f.write("{:<15}{}\n".format(row[0], "".join([f"{val:>10}" for val in row[1:]])))
