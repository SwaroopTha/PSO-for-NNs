import torch
import numpy as np
# from sobol_seq import i4_sobol_generate
from scipy.stats import qmc
import math

class ParticleSwarmOptimizer:
    def __init__(self,
                 model,
                 variant="BPSO", # [BPSO, SGPSO, GSA, PSOGSA, PPSO]
                 num_particles=50,
                 max_iter=500,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model
        self.variant = variant
        self.num_particles = num_particles
        self.device = device
        self.max_iter = max_iter
        self.current_iter = 0
        self.history = []

        self._setup_variant_parameters()
        self._initialize_particles()

        self.inertia = self.params["inertia_range"][0] 


    def _setup_variant_parameters(self):
        self.params = {
            "cognitive_coeff": 1.5,
            "social_coeff": 1.5,
            "inertia_range": (0.9, 0.3), # (start, end)
            "use_sobol": False,
            "G0": 1.0 # Gravitational constant for GSA/PSOGSA
        }

        if self.variant == "PPSO":
            self.params.update({
                "cognitive_coeff": 1.6,
                "social_coeff": 1.7,
                "inertia_range": (0.4, 0.9),
                "use_sobol": True,
            })
        elif self.variant == "SGPSO":
            self.params.update({
                "c3": 0.5,
                "geometric_center": 100,
            })
        elif self.variant == "PSOGSA":
            self.params.update({
                "cognitive_coeff": 1.0,
                "social_coeff": 1.0,
                "inertia_range": (0.9, 0.5),
            })
        elif self.variant == "GSA":
            self.params.update({
                "cognitive_coeff": 0.0, # Disable PSO component
                "social_coeff": 0.0,
                "inertia_range": (0.0, 0.0), # Disable inertia
                "G0": 1.0, # Gravitational constant
            })

    def _initialize_particles(self):
        # Initialize particles
        self.param_shape = [p.shape for p in self.model.parameters()]
        self.param_sizes = [p.numel() for p in self.model.parameters()]
        self.total_params = sum(self.param_sizes)

        # initialize the positions
        if self.variant == "PPSO":
            # Sobol sequence initialization
            adj_particles = 2 ** math.ceil(math.log2(self.num_particles))
            sampler = qmc.Sobol(d=self.total_params, scramble=True)
            sample = sampler.random(n=adj_particles)
            self.positions = torch.tensor(2 * sample[:self.num_particles] - 1, 
                                          dtype=torch.float32, 
                                          device=self.device)
        else:
            self.positions = torch.randn(self.num_particles, self.total_params, device=self.device) * 0.1 # because the weights are small
        
        self.velocities = torch.zeros_like(self.positions, device=self.device)
        self.best_positions = self.positions.clone()
        self.best_scores = torch.full((self.num_particles,), float('inf'), device=self.device)
        self.global_best_position = self.positions[0].clone()
        self.global_best_score = float('inf')

        if self.variant in ["GSA", "PSOGSA"]:
            self.masses = torch.zeros(self.num_particles, device=self.device)
            self.accelerations = torch.zeros_like(self.positions)

        

    def _update_inertia(self):
        if self.variant == "PPSO":
            # linear decrease of inertia
            self.inertia = self.params["inertia_range"][0] + \
                np.tanh(self.current_iter  * ((self.params["inertia_range"][1] - self.params["inertia_range"][0]) / self.max_iter))
        elif self.variant == ["BPSO", "SGPSO", "PSOGSA"]:
            progress = self.current_iter / self.max_iter
            self.inertia = self.params["inertia_range"][0] - \
                (self.params["inertia_range"][0] - self.params["inertia_range"][1]) * progress
            
    def _compute_gsa_foces(self):
        distances = torch.cdist(self.positions, self.positions) + 1e-10  # Avoid division by zero
        force = self.params["G0"] * (self.masses.view(-1, 1) * self.masses) / distances
        position_diffs = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        net_force = torch.sum(force.unsqueeze(-1) * position_diffs, dim=1)
        return net_force

    def _update_velocities(self):
        r_p = torch.rand_like(self.positions)
        r_g = torch.rand_like(self.positions)

        # Base PSO
        cognitive = self.params["cognitive_coeff"] * r_p * (self.best_positions - self.positions)
        social = self.params["social_coeff"] * r_g * (self.global_best_position - self.positions)

        if self.variant == "SGPSO":
            center = torch.mean(self.positions, dim=0)
            geometric = self.params["c3"] * (center - self.positions)
            return self.inertia * self.velocities + cognitive + social + geometric
        elif self.variant == "PSOGSA":
            gsa_force = self._compute_gsa_foces()
            return self.inertia * self.velocities + cognitive + social + gsa_force
        elif self.variant == "GSA":
            return self._compute_gsa_foces()
        else:
            return self.inertia * self.velocities + cognitive + social


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


    def step(self, data_loader, criterion):
        """Execute one optimization step"""
        self._update_inertia()
        
        # Evaluate particles
        scores = torch.zeros(self.num_particles, device=self.device)
        for i in range(self.num_particles):
            self._update_model_parameters(i)
            with torch.no_grad():
                for X, y in data_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    scores[i] += criterion(outputs, y).item() * len(y)
        scores /= len(data_loader.dataset)

        # Update best positions
        improved = scores < self.best_scores
        self.best_positions[improved] = self.positions[improved]
        self.best_scores[improved] = scores[improved]
        
        # Update global best
        min_score, min_idx = torch.min(scores, dim=0)
        if min_score < self.global_best_score:
            self.global_best_score = min_score
            self.global_best_position = self.positions[min_idx].clone()

        # GSA-specific mass update
        if self.variant in ["GSA", "PSOGSA"]:
            worst_score = torch.max(scores)
            self.masses = (worst_score - scores) / (worst_score - torch.min(scores) + 1e-10)

        # Update positions
        self.velocities = self._update_velocities()
        self.positions += self.velocities
        self.positions.clamp_(-3.0, 3.0)  # Paper uses [-1,1] but we match their code
        
        self.history.append(self.global_best_score)
        self.current_iter += 1
        # if self._check_early_stop():
        #     print(f"Early stopping at iteration {self.current_iter}")
        #     return None


        return self.global_best_score

        
    def _check_early_stop(self, patience=20, tol=1e-5):
        """Check if loss hasn't improved for 'patience' iterations"""
        if len(self.history) < patience + 1:
            return False
            
        # Check if relative improvement < tol
        best_recent = min(self.history[-patience:])
        best_previous = min(self.history[:-patience])
        relative_improvement = (best_previous - best_recent) / (abs(best_previous) + 1e-10)
        
        return relative_improvement < tol