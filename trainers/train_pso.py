# trainers/train_pso.py
from optim.pso import ParticleSwarmOptimizer
import matplotlib.pyplot as plt
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
        if epoch % 50 == 0:  # Reduced print frequency
            print(f"[{variant}] Epoch {epoch}: Loss = {loss:.4f}")

    # Plotting
    if plot_convergence:
        plt.plot(optimizer.history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{variant} Training Convergence')
        plt.show()
    
    return optimizer, optimizer.global_best_position  # Best found parameters
