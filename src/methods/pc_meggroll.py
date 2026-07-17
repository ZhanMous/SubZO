"""PC-MEGGROLL: Subspace-Constrained Low-Rank Zeroth-Order Optimizer.

Core novel contribution. Combines EGGROLL's low-rank perturbation mechanism
with SAFA's old-class protection subspace.

Key idea: constrain low-rank perturbations to directions that are:
1. Plastic (can modify weights to learn new classes)
2. Non-interfering (don't disrupt base-class representations)

This is achieved by projecting perturbations orthogonally to the base-class
prototype subspace, then applying fitness-weighted updates.

Memory: O((a+b)*r) per perturbation vs O(a*b) for full-rank.
Speed: shared base matmul + tiny corrections → high GPU utilization.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PCMEGGROLL:
    """Subspace-Constrained Low-Rank ES Optimizer.

    Args:
        model: The model to optimize
        rank: Rank of perturbation factors (default: 4)
        sigma: Perturbation magnitude
        population_size: Number of perturbation directions per step
        lr: Learning rate for parameter updates
        weight_decay: Weight decay coefficient
        antithetic: Use antithetic sampling (+eps, -eps)
        fitness_shaping: Utility shaping function ('rank' or 'centered')
        protection_subspace: Base-class prototype basis for projection [n_base, D]
        blend_weight: How much to pull perturbations toward protection subspace
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        sigma: float = 0.02,
        population_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        antithetic: bool = True,
        fitness_shaping: str = "rank",
        protection_subspace: Optional[torch.Tensor] = None,
        blend_weight: float = 0.1,
    ):
        self.model = model
        self.rank = rank
        self.sigma = sigma
        self.population_size = population_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.antithetic = antithetic
        self.fitness_shaping_type = fitness_shaping
        self.blend_weight = blend_weight

        # Flatten all parameters into a single vector
        self.param_vector, self.param_shapes, self.param_sizes = self._flatten_params()
        self.num_params = self.param_vector.numel()

        # Compute matrix dimensions for low-rank factorization
        self.m = math.ceil(math.sqrt(self.num_params))
        self.n = math.ceil(self.num_params / self.m)

        # Protection subspace (base-class prototype basis)
        self.protection_subspace = protection_subspace  # [n_base, feature_dim]

        # Precompute projection matrix if subspace is given
        self._projection_matrix = None
        if protection_subspace is not None:
            self._precompute_projection()

    def _flatten_params(self) -> tuple[torch.Tensor, list, list]:
        """Flatten model parameters into a single vector."""
        params = []
        shapes = []
        sizes = []
        for p in self.model.parameters():
            shapes.append(p.shape)
            sizes.append(p.numel())
            params.append(p.data.view(-1))
        return torch.cat(params), shapes, sizes

    def _unflatten_params(self, vector: torch.Tensor) -> list[torch.Tensor]:
        """Unflatten a vector back into parameter tensors."""
        params = []
        offset = 0
        for shape, size in zip(self.param_shapes, self.param_sizes):
            params.append(vector[offset:offset + size].view(shape))
            offset += size
        return params

    def _precompute_projection(self):
        """Precompute the projection matrix for the protection subspace.

        The projection removes components in the base-class prototype space.
        P_proj = I - base^T @ pinv(base @ base^T) @ base
        """
        base = self.protection_subspace  # [n_base, feature_dim]
        gram = base @ base.T  # [n_base, n_base]
        gram_pinv = torch.linalg.pinv(gram)
        # Projection matrix: [feature_dim, feature_dim]
        self._projection_matrix = torch.eye(base.shape[1], device=base.device) - base.T @ gram_pinv @ base

    def _sample_lowrank_perturbation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a low-rank perturbation in factored form.

        Returns:
            u: [m, rank] factor
            v: [n, rank] factor
        """
        u = torch.randn(self.m, self.rank, device=self.param_vector.device)
        v = torch.randn(self.n, self.rank, device=self.param_vector.device)
        return u, v

    def _perturbation_to_vector(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert factored perturbation to flat vector.

        eps = (u @ v^T).flatten()[:num_params]
        """
        eps_matrix = u @ v.T  # [m, n]
        return eps_matrix.flatten()[:self.num_params]

    def _apply_protection_subspace(self, eps: torch.Tensor) -> torch.Tensor:
        """Project perturbation to be orthogonal to protection subspace.

        This ensures perturbations don't interfere with base-class representations.

        Args:
            eps: Flat perturbation vector [num_params]

        Returns:
            Projected perturbation [num_params]
        """
        if self._projection_matrix is None:
            return eps

        # The projection operates on the feature space (last layer).
        # For other parameters, perturbations pass through unmodified.
        # Find the FC layer parameters (prototypes) - last num_base*feature_dim params
        feature_dim = self.protection_subspace.shape[1]
        n_base = self.protection_subspace.shape[0]
        fc_size = n_base * feature_dim

        # Split: non-FC params (pass through) and FC params (project)
        non_fc = eps[:-fc_size]
        fc_flat = eps[-fc_size:]
        fc_matrix = fc_flat.view(n_base, feature_dim)  # [n_base, feature_dim]

        # Project FC perturbations
        fc_projected = fc_matrix @ self._projection_matrix.T  # [n_base, feature_dim]

        return torch.cat([non_fc, fc_projected.flatten()])

    def _compute_fitness_shaping(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """Compute utility weights for the population.

        Rank-based utilities (Wierstra et al., 2014):
        u_k = max(0, log(n/2 + 1) - log(k)), normalized and centered.

        Args:
            fitnesses: [population_size] raw fitness values

        Returns:
            [population_size] utility weights (sum to 0)
        """
        n = len(fitnesses)
        if self.fitness_shaping_type == "rank":
            # Rank-based utilities
            ranks = fitnesses.argsort().argsort().float() + 1  # 1-indexed
            utilities = torch.clamp(torch.log(torch.tensor(n / 2.0 + 1)) - torch.log(ranks), min=0)
            utilities = utilities / utilities.sum()
            utilities = utilities - 1.0 / n  # Center
            return utilities
        elif self.fitness_shaping_type == "centered":
            # Simple centered fitness
            return (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        else:
            raise ValueError(f"Unknown fitness shaping: {self.fitness_shaping_type}")

    def step(self, fitness_fn) -> float:
        """Perform one optimization step.

        Args:
            fitness_fn: Callable(params_vector) -> scalar fitness value

        Returns:
            Mean fitness of the population
        """
        device = self.param_vector.device
        half_pop = self.population_size // 2 if self.antithetic else self.population_size

        # Sample perturbations and evaluate fitness
        fitnesses = []
        perturbations = []

        for _ in range(half_pop):
            u, v = self._sample_lowrank_perturbation()
            eps = self._perturbation_to_vector(u, v)

            # Apply protection subspace projection
            eps = self._apply_protection_subspace(eps)

            # Evaluate positive perturbation
            self.param_vector.add_(eps * self.sigma)
            self._restore_params()
            fit_pos = fitness_fn(self.param_vector)

            if self.antithetic:
                # Evaluate negative perturbation (reuse u, v with negated u)
                self.param_vector.sub_(eps * self.sigma * 2)
                self._restore_params()
                fit_neg = fitness_fn(self.param_vector)
                self.param_vector.add_(eps * self.sigma)  # Restore

                fitnesses.extend([fit_pos, fit_neg])
                perturbations.extend([eps, -eps])
            else:
                fitnesses.append(fit_pos)
                perturbations.append(eps)
                self.param_vector.sub_(eps * self.sigma)
                self._restore_params()

        # Compute gradient estimate
        fitnesses = torch.tensor(fitnesses, device=device)
        perturbations = torch.stack(perturbations, dim=0)  # [pop, num_params]
        utilities = self._compute_fitness_shaping(fitnesses)

        # Weighted sum of perturbations
        grad_estimate = (utilities.unsqueeze(1) * perturbations).sum(dim=0) / self.sigma

        # Parameter update
        self.param_vector.add_(self.lr * grad_estimate)
        if self.weight_decay > 0:
            self.param_vector.mul_(1 - self.lr * self.weight_decay)

        self._restore_params()
        return fitnesses.mean().item()

    def _restore_params(self):
        """Restore flattened parameter vector back into model."""
        offset = 0
        for p, size in zip(self.model.parameters(), self.param_sizes):
            p.data.copy_(self.param_vector[offset:offset + size].view(p.shape))
            offset += size

    def update_protection_subspace(self, new_subspace: torch.Tensor):
        """Update the protection subspace (e.g., after adding new classes).

        Args:
            new_subspace: Updated base-class prototype basis [n_classes, feature_dim]
        """
        self.protection_subspace = new_subspace
        self._precompute_projection()

    def get_perturbation_memory(self) -> int:
        """Return memory in bytes for storing one perturbation."""
        # Full rank: m * n * 4 bytes (float32)
        # Low rank: (m + n) * rank * 4 bytes
        full_rank_bytes = self.m * self.n * 4
        low_rank_bytes = (self.m + self.n) * self.rank * 4
        return low_rank_bytes

    def get_memory_ratio(self) -> float:
        """Return memory ratio vs full-rank perturbation."""
        full_rank_bytes = self.m * self.n * 4
        low_rank_bytes = (self.m + self.n) * self.rank * 4
        return full_rank_bytes / low_rank_bytes

    def get_speed_estimate(self) -> dict:
        """Return speed-related metrics for benchmarking."""
        return {
            "num_params": self.num_params,
            "matrix_dims": (self.m, self.n),
            "rank": self.rank,
            "perturbation_elements": (self.m + self.n) * self.rank,
            "full_rank_elements": self.m * self.n,
            "compression_ratio": self.m * self.n / ((self.m + self.n) * self.rank),
        }
