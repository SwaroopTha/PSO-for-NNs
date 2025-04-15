import torch
import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self,
                 model,
                 num_particles=20,
                 inertia=0.6,
                 cognitive_coeff=2.0,
                 social_coeff=2.0,
                 max_param_value=3.0,
                 min_param_value=-3.0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.num_particles = num_particles
        self.device = device
        self.history = []


        # Initialize particles
        self.param_shape = [p.shape for p in model.parameters()]
        self.param_sizes = [p.numel() for p in model.parameters()]
        self.total_params = sum(self.param_sizes)

        # initialize the positions
        self.positions = torch.randn(num_particles, self.total_params, device=device) * 0.1 # because the weights are small
        self.velocities = torch.randn(self.positions.shape, device=device) * 0.1 # small random velocities

        # Best positions
        self.best_positions = self.positions.clone()
        with torch.no_grad():
            self.best_scores = torch.tensor(
                [float('inf')] * num_particles, device=device
            ) # Initialize best scores to infinity

        self.global_best_position = self.positions[0].clone()
        self.global_best_score = float('inf')

        # Hyperparameters
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.min_param = min_param_value
        self.max_param = max_param_value



    def _update_model_parameters(self, particle_idx):
        """
        Update model with flattened parameters from single particle
        """
        params = self.positions[particle_idx]
        pointer = 0 # Pointer to track the current position in the flattened parameters
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = params[pointer:pointer + param_size].view(param.shape)
            pointer += param_size


    def _evaluate_particles(self, data_loader, criterion):
        """
        Evaluate all particles on current batch of data
        """
        scores = torch.zeros(self.num_particles, device=self.device)
        for i in range(self.num_particles):
            self._update_model_parameters(i)
            with torch.no_grad():
                for X, y in data_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = criterion(outputs, y)
                    loss_value = loss.detach().item()
                    scores[i] += loss_value * X.size(0)  

        return scores / len(data_loader.dataset)  # Average loss over the dataset
    

    def step(self, data_loader, criterion):
        scores = self._evaluate_particles(data_loader, criterion)

        # update personal bests
        improved = scores < self.best_scores
        self.best_positions[improved] = self.positions[improved]
        self.best_scores[improved] = scores[improved]


        # update global best
        min_score, min_idx = torch.min(scores, dim=0)
        if min_score < self.global_best_score:
            self.global_best_score = min_score
            self.global_best_position = self.positions[min_idx].clone()
        
        # update velocities and positions
        r_p = torch.rand_like(self.positions)
        r_g = torch.rand_like(self.positions)

        self.velocities = (
            self.inertia * self.velocities +
            self.cognitive_coeff * r_p * (self.best_positions - self.positions) +
            self.social_coeff * r_g * (self.global_best_position - self.positions)
        )

        self.positions += self.velocities
        self.positions.clamp_(self.min_param, self.max_param)

        self.history.append(self.global_best_score)
        return self.global_best_score  # Optional: return current best
        
