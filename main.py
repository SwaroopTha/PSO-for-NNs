import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.mlp import MLP
import sys
sys.path.append('./')
from data.data import load_ucidata
from trainers.train_pso import train_pso
from trainers.train_sgd_adam import train_gd  # <-- Import the GD trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

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
    # datasets = ['iris', 'wine', 'breast_cancer', 'diabetes']
    datasets = ['digits']
    pso_variants = ['BPSO', 'PPSO', 'SGPSO', 'PSOGSA', 'GSA']
    # pso_variants = []
    gd_variants = ['SGD', 'Adam']
    all_variants = pso_variants + gd_variants

    hidden_dims = [16]
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracy_table = {}

    os.makedirs("confusion_matrices", exist_ok=True)

    for dataset_name in datasets:
        print(f"\n=== Running on dataset: {dataset_name.upper()} ===")
        train_loader, test_loader, input_dim, output_dim = load_ucidata(dataset_name)
        model_factory = lambda: MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        histories = {}
        accuracy_table[dataset_name] = {}

        for variant in all_variants:
            model = model_factory().to(device)

            if variant in pso_variants:
                optimizer, best_params, accuracy_history = train_pso(
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
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    predictions = outputs.argmax(dim=1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    correct += (predictions == y_batch).sum().item()
                    total += y_batch.size(0)

            final_accuracy = correct / total
            accuracy_table[dataset_name][variant] = final_accuracy

            # Confusion matrix as percentage (row-normalized)
            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # normalize by true label count

            fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
            disp.plot(
                cmap='Blues',
                values_format=".2%",
                ax=ax,
                colorbar=False
            )
            ax.set_title(f"{dataset_name.capitalize()} - {variant} Confusion Matrix (%)", fontsize=14)
            ax.tick_params(axis='x', labelrotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            plt.tight_layout()
            plt.savefig(f"confusion_matrices/cm_{dataset_name}_{variant}.png", dpi=300, bbox_inches='tight')
            plt.close()



            # Store history for plotting
            cpu_history = [h.item() if torch.is_tensor(h) else h for h in history]
            histories[variant] = cpu_history
            if 'accuracy_histories' not in locals():
                accuracy_histories = {}
            accuracy_histories[variant] = accuracy_history

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

    plt.figure(figsize=(10, 6))
    for variant, acc_history in accuracy_histories.items():
        plt.plot(acc_history, label=variant, alpha=0.8, linewidth=2)
    plt.title(f'{dataset_name.capitalize()} - PSO Variants Accuracy', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'pso_accuracy_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
