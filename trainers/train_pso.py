# trainers/train_pso.py
from optim.pso import ParticleSwarmOptimizer
import matplotlib.pyplot as plt
import torch
import time
from collections import defaultdict
import torch.nn as nn


def train_pso(
    model,
    train_loader,
    criterion,
    variant="BPSO",
    num_particles=50,
    num_iterations=500,
    plot_convergence=True,
    config=None
):

    device = next(model.parameters()).device
    metrics = defaultdict(list)
    start_time = time.time()
    iteration_times = []

    # Initialize the appropriate optimizer variant
    optimizer = ParticleSwarmOptimizer(
        model=model,
        variant=variant,
        num_particles=num_particles,
        max_iter=num_iterations,
        config=config,
        device=device,
    )

    # print(optimizer.inertia)

    # Training loop
    for epoch in range(num_iterations):
        iter_start_time = time.time()
        loss = optimizer.step(train_loader, criterion)
        iter_end_time = time.time()
        iteration_time = iter_end_time - iter_start_time
        iteration_times.append(iteration_time)

        if loss is None:
            break

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total

        if not hasattr(optimizer, 'accuracy_history'):
            optimizer.accuracy_history = []
        optimizer.accuracy_history.append(accuracy)

        if epoch % 50 == 0:  # Reduced print frequency
            print(f"[{variant}] Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}, Iteration Time = {iteration_time:.4f} sec")

    end_time = time.time()
    total_training_time = end_time - start_time
    average_iteration_time = sum(iteration_times) / len(iteration_times) if iteration_times else 0

    print(f"[{variant}] Total Training Time: {total_training_time:.2f} sec")
    print(f"[{variant}] Average Time per Iteration: {average_iteration_time:.4f} sec")

    # Plotting
    if plot_convergence:
        plt.plot(optimizer.history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{variant} Training Convergence')
        plt.show()

    return optimizer, optimizer.global_best_position, optimizer.accuracy_history  # Best found parameters