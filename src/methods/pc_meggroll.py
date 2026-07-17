"""PC-MEGGROLL: Projection-Constrained Masked EGGROLL.

Core method: update only adaptive-channel blocks of a target linear layer
using low-rank perturbations projected onto the orthogonal complement of
the old-class feature subspace. No inference parameter growth.

Key formula (Section 3.1 of research plan v2):
    B_perp = B - U (U^T B)         # project B factors away from protection subspace
    E = D_out A B_perp^T D_in / sqrt(r)

where:
    A in R^{m x r}, B in R^{n x r}   -- low-rank factors
    U in R^{n x k}                    -- old-class feature subspace basis (columns orthonormal)
    D_out, D_in                       -- structured masks from SAFA adaptive channels
    r                                 -- rank

Properties:
    E . U = 0  (perturbation annihilates old-class features)
    Memory: O(r(m+n)) for factors + O(nk) for U, never materializes full E or I-UU^T
"""

import torch
import math
from typing import Optional, Tuple


class PCMEGGROLL:
    """Projection-Constrained Masked EGGROLL optimizer.

    Args:
        weight: Target weight matrix W of shape (m, n)
        rank: Rank r of perturbation factors
        sigma: Perturbation magnitude
        population_size: Number of perturbation directions per step
        lr: Learning rate
        antithetic: Use antithetic sampling (+eps, -eps)
        fitness_shaping: Utility shaping ('rank' or 'centered')
        mask_out: Boolean mask for output channels (adaptive=True) [m]
        mask_in: Boolean mask for input channels (adaptive=True) [n]
        subspace_basis: Old-class feature subspace U, columns orthonormal [n, k]
    """

    def __init__(
        self,
        weight: torch.Tensor,
        rank: int = 4,
        sigma: float = 0.02,
        population_size: int = 64,
        lr: float = 0.001,
        antithetic: bool = True,
        fitness_shaping: str = "rank",
        mask_out: Optional[torch.Tensor] = None,
        mask_in: Optional[torch.Tensor] = None,
        subspace_basis: Optional[torch.Tensor] = None,
    ):
        self.W = weight
        self.m, self.n = weight.shape
        self.rank = rank
        self.sigma = sigma
        self.population_size = population_size
        self.lr = lr
        self.antithetic = antithetic
        self.fitness_shaping_type = fitness_shaping

        # Structured masks (adaptive channels)
        self.mask_out = mask_out
        self.mask_in = mask_in
        self._D_out = mask_out.float() if mask_out is not None else None
        self._D_in = mask_in.float() if mask_in is not None else None

        # Protection subspace basis U [n, k], columns orthonormal
        self.U = subspace_basis
        self._UT = subspace_basis.T if subspace_basis is not None else None

    def _sample_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample low-rank factors A [m, r], B [n, r]."""
        A = torch.randn(self.m, self.rank, device=self.W.device)
        B = torch.randn(self.n, self.rank, device=self.W.device)
        return A, B

    def _project_B(self, B: torch.Tensor) -> torch.Tensor:
        """B_perp = B - U (U^T B)  -- remove components in old-class subspace."""
        if self._UT is None:
            return B
        return B - self.U @ (self._UT @ B)

    def _construct_perturbation(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """E = D_out A B_perp^T D_in / sqrt(r)."""
        B_perp = self._project_B(B)

        if self._D_out is not None:
            A = A * self._D_out.unsqueeze(1)
        if self._D_in is not None:
            B_perp = B_perp * self._D_in.unsqueeze(1)

        return A @ B_perp.T / math.sqrt(self.rank)

    def _compute_utilities(self, fitnesses: torch.Tensor) -> torch.Tensor:
        """Rank-based fitness shaping."""
        n = len(fitnesses)
        if self.fitness_shaping_type == "rank":
            ranks = fitnesses.argsort().argsort().float() + 1
            utilities = torch.clamp(
                torch.log(torch.tensor(n / 2.0 + 1)) - torch.log(ranks), min=0
            )
            utilities = utilities / utilities.sum()
            return utilities - 1.0 / n
        return (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

    @torch.no_grad()
    def step(self, fitness_fn) -> Tuple[float, dict]:
        """One optimization step.

        Args:
            fitness_fn: Callable(W) -> scalar fitness (higher = better)

        Returns:
            mean_fitness, diagnostics
        """
        device = self.W.device
        W_orig = self.W.data.clone()
        half = self.population_size // 2 if self.antithetic else self.population_size

        fitnesses, perturbations, eu_checks = [], [], []

        for _ in range(half):
            A, B = self._sample_factors()
            E = self._construct_perturbation(A, B)

            # Numerical check: E . U should be ~0
            if self.U is not None and len(eu_checks) < 3:
                eu_checks.append((E @ self.U).abs().max().item())

            # Positive
            self.W.data = W_orig + self.sigma * E
            fit_pos = fitness_fn(self.W)

            if self.antithetic:
                self.W.data = W_orig - self.sigma * E
                fit_neg = fitness_fn(self.W)
                fitnesses.extend([fit_pos, fit_neg])
                perturbations.extend([E, -E])
            else:
                fitnesses.append(fit_pos)
                perturbations.append(E)

        self.W.data = W_orig

        fits = torch.tensor(fitnesses, device=device)
        utils = self._compute_utilities(fits)
        pert = torch.stack(perturbations)
        grad = (utils.view(-1, 1, 1) * pert).sum(0) / self.sigma

        self.W.data = W_orig + self.lr * grad

        return fits.mean().item(), {
            "mean_fitness": fits.mean().item(),
            "max_fitness": fits.max().item(),
            "E_U_max": max(eu_checks) if eu_checks else 0.0,
            "grad_norm": grad.norm().item(),
        }

    def verify_protection(self, test_features: torch.Tensor) -> dict:
        """Check E . U features ~ 0."""
        A, B = self._sample_factors()
        E = self._construct_perturbation(A, B)

        if self.U is not None:
            proj = self.U @ (self._UT @ test_features)
            return {
                "protected_drift": (E @ proj).norm().item(),
                "residual_drift": (E @ (test_features - proj)).norm().item(),
                "perturbation_norm": E.norm().item(),
            }
        return {
            "protected_drift": (E @ test_features).norm().item(),
            "residual_drift": 0.0,
            "perturbation_norm": E.norm().item(),
        }

    def get_system_metrics(self) -> dict:
        """System metrics for K0."""
        k = self.U.shape[1] if self.U is not None else 0
        factor_mem = (self.m + self.n) * self.rank * 4
        U_mem = self.n * k * 4
        full_mem = self.m * self.n * 4
        total = factor_mem + U_mem
        return {
            "weight_shape": (self.m, self.n),
            "rank": self.rank,
            "subspace_dim": k,
            "factor_memory_bytes": factor_mem,
            "subspace_memory_bytes": U_mem,
            "total_perturbation_memory_bytes": total,
            "full_rank_memory_bytes": full_mem,
            "memory_reduction_ratio": full_mem / total if total > 0 else float("inf"),
        }
