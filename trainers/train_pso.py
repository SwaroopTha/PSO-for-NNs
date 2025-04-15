# trainers/train_pso.py
from optim.pso import ParticleSwarmOptimizer
from models.pso_nn_wrapper import PSONNWrapper
import matplotlib.pyplot as plt

def train_pso(model, train_loader, criterion, num_iterations=100):
    wrapper = PSONNWrapper(model, criterion)
    pso = ParticleSwarmOptimizer(
        model, num_particles=20, inertia=0.6, 
        cognitive_coeff=2.0, social_coeff=2.0
    )
    
    for epoch in range(num_iterations):
        pso.step(train_loader, criterion)
        print(f"Epoch {epoch}: Loss = {pso.global_best_score:.4f}")

    plt.plot(pso.history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Particle Swarm Optimization Loss')
    
    return pso.global_best_position  # Best found parameters