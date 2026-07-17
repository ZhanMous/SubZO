"""Full-rank Evolution Strategies baseline.

Standard ES with unstructured Gaussian perturbations.
Used as a comparison baseline for PC-MEGGROLL's speed/memory claims.
"""

import torch
import torch.nn as nn
from typing import Optional


class FullRankES:
    """Full-rank ES optimizer (baseline for comparison).

    Args:
        model: The model to optimize
        sigma: Perturbation magnitude
        population_size: Number of perturbation directions per step
        lr: Learning rate
        weight_decay: Weight decay
        antithetic: Use antithetic sampling
        fitness_shaping: Utility shaping ('rank' or 'centered')
    """

    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.02,
        population_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        antithetic: bool = True,
        fitness_shaping: str = "rank",
    ):
        self.model = model
        self.sigma = sigma
        self.population_size = population_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.antithetic = antithetic
        self.fitness_shaping_type = fitness_shaping

        # Flatten parameters
        self.param_vector, self.param_shapes, self.param_sizes = self._flatten_params()
        self.num_params = self.param_vector.numel()

    def _flatten_params(self) -> tuple[torch.Tensor, list, list]:
        params = []
        shapes = []
        sizes = []
        for p in self.model.parameters():
            shapes.append(p.shape)
            sizes.append(p.numel())
            params.append(p.data.view(-1))
        return torch.cat(params), shapes, sizes

    def _restore_params(self):
        offset = 0
        for p, size in zip(self.model.parameters(), self.param_sizes):
            p.data.copy_(self.param_vector[offset:offset + size].view(p.shape))
            offset += size

    def _compute_fitness_shaping(self, fitnesses: torch.Tensor) -> torch.Tensor:
        n = len(fitnesses)
        if self.fitness_shaping_type == "rank":
            ranks = fitnesses.argsort().argsort().float() + 1
            utilities = torch.clamp(torch.log(torch.tensor(n / 2.0 + 1)) - torch.log(ranks), min=0)
            utilities = utilities / utilities.sum()
            utilities = utilities - 1.0 / n
            return utilities
        elif self.fitness_shaping_type == "centered":
            return (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        else:
            raise ValueError(f"Unknown fitness shaping: {self.fitness_shaping_type}")

    def step(self, fitness_fn) -> float:
        """Perform one optimization step with full-rank perturbations.

        Args:
            fitness_fn: Callable(params_vector) -> scalar fitness

        Returns:
            Mean fitness of the population
        """
        device = self.param_vector.device
        half_pop = self.population_size // 2 if self.antithetic else self.population_size

        fitnesses = []
        perturbations = []

        for _ in range(half_pop):
            # Full-rank random perturbation
            eps = torch.randn(self.num_params, device=device)

            # Evaluate positive
            self.param_vector.add_(eps * self.sigma)
            self._restore_params()
            fit_pos = fitness_fn(self.param_vector)

            if self.antithetic:
                # Evaluate negative
                self.param_vector.sub_(eps * self.sigma * 2)
                self._restore_params()
                fit_neg = fitness_fn(self.param_vector)
                self.param_vector.add_(eps * self.sigma)

                fitnesses.extend([fit_pos, fit_neg])
                perturbations.extend([eps, -eps])
            else:
                fitnesses.append(fit_pos)
                perturbations.append(eps)
                self.param_vector.sub_(eps * self.sigma)
                self._restore_params()

        # Gradient estimate
        fitnesses = torch.tensor(fitnesses, device=device)
        perturbations = torch.stack(perturbations, dim=0)
        utilities = self._compute_fitness_shaping(fitnesses)
        grad_estimate = (utilities.unsqueeze(1) * perturbations).sum(dim=0) / self.sigma

        # Update
        self.param_vector.add_(self.lr * grad_estimate)
        if self.weight_decay > 0:
            self.param_vector.mul_(1 - self.lr * self.weight_decay)

        self._restore_params()
        return fitnesses.mean().item()

    def get_perturbation_memory(self) -> int:
        """Return memory in bytes for one full-rank perturbation."""
        return self.num_params * 4  # float32

    def get_speed_estimate(self) -> dict:
        return {
            "num_params": self.num_params,
            "perturbation_elements": self.num_params,
        }
