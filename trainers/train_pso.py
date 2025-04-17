# trainers/train_pso.py
from optim.pso import ParticleSwarmOptimizer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def train_pso(
    model, 
    train_loader, 
    criterion,
    variant="BPSO",
    num_particles=50,
    num_iterations=500,
    plot_convergence=True
):

    # Initialize the appropriate optimizer variant
    optimizer = ParticleSwarmOptimizer(
        model=model,
        variant=variant,
        num_particles=num_particles,
        max_iter=num_iterations
    )

    # print(optimizer.inertia)
    
    # Training loop
    for epoch in range(num_iterations):
        loss = optimizer.step(train_loader, criterion)
        if loss is None:
            break

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(next(model.parameters()).device), y_batch.to(next(model.parameters()).device)
                outputs = model(x_batch)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total

        if not hasattr(optimizer, 'accuracy_history'):
            optimizer.accuracy_history = []
        optimizer.accuracy_history.append(accuracy)

        if epoch % 50 == 0:  # Reduced print frequency
            print(f"[{variant}] Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    # Plotting
    if plot_convergence:
        plt.plot(optimizer.history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{variant} Training Convergence')
        plt.show()
    
    return optimizer, optimizer.global_best_position  # Best found parameters
