"""Tests for low-rank perturbation generation and memory characteristics."""

import pytest
import torch
import math
import sys
sys.path.insert(0, "src")

from methods.pc_meggroll import PCMEGGROLL
from methods.fullrank_es import FullRankES
from models.vgg9_snn import VGG9SNN


class TestLowRankPerturbation:
    """Test low-rank perturbation mechanics."""

    def _make_dummy_model(self):
        """Create a small model for testing."""
        return torch.nn.Linear(100, 10)

    def test_perturbation_rank(self):
        """Generated perturbation has rank <= specified rank."""
        model = self._make_dummy_model()
        opt = PCMEGGROLL(model, rank=2, population_size=4)
        u, v = opt._sample_lowrank_perturbation()
        eps_matrix = u @ v.T
        # Rank should be <= 2
        rank = torch.linalg.matrix_rank(eps_matrix).item()
        assert rank <= 2

    def test_perturbation_shape(self):
        """Factored perturbation reconstructs to correct shape."""
        model = self._make_dummy_model()
        opt = PCMEGGROLL(model, rank=4)
        u, v = opt._sample_lowrank_perturbation()
        eps = opt._perturbation_to_vector(u, v)
        assert eps.shape[0] == opt.num_params

    def test_antithetic_perturbation(self):
        """Antithetic perturbations are negatives of each other."""
        model = self._make_dummy_model()
        opt = PCMEGGROLL(model, rank=2, antithetic=True)
        u, v = opt._sample_lowrank_perturbation()
        eps_pos = opt._perturbation_to_vector(u, v)
        eps_neg = opt._perturbation_to_vector(-u, v)
        assert torch.allclose(eps_pos, -eps_neg)


class TestMemoryAdvantage:
    """Test that low-rank perturbations use less memory."""

    def test_memory_ratio_linear(self):
        """Linear model shows memory reduction."""
        model = torch.nn.Linear(1000, 1000)
        opt = PCMEGGROLL(model, rank=4)
        ratio = opt.get_memory_ratio()
        assert ratio > 1.0, f"Expected memory ratio > 1, got {ratio}"

    def test_memory_ratio_vgg9(self):
        """VGG9SNN shows significant memory reduction."""
        model = VGG9SNN(time_steps=4, num_classes=100)
        opt = PCMEGGROLL(model, rank=4)
        ratio = opt.get_memory_ratio()
        assert ratio > 1.5, f"Expected memory ratio > 1.5, got {ratio}"

    def test_lowrank_less_memory_than_fullrank(self):
        """Low-rank uses less memory than full-rank for same model."""
        model = VGG9SNN(time_steps=4, num_classes=100)
        lowrank = PCMEGGROLL(model, rank=4)
        fullrank = FullRankES(model)
        assert lowrank.get_perturbation_memory() < fullrank.get_perturbation_memory()

    def test_speed_estimate_compression(self):
        """Speed estimate shows compression ratio > 1."""
        model = VGG9SNN(time_steps=4, num_classes=100)
        opt = PCMEGGROLL(model, rank=4)
        info = opt.get_speed_estimate()
        assert info["compression_ratio"] > 1.0

    def test_rank_scales_compression(self):
        """Higher rank → less compression (trade-off)."""
        model = VGG9SNN(time_steps=4, num_classes=100)
        ratios = []
        for rank in [1, 2, 4, 8, 16]:
            opt = PCMEGGROLL(model, rank=rank)
            ratios.append(opt.get_memory_ratio())
        # Ratios should decrease as rank increases
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], f"Rank {i} should have >= ratio than rank {i+1}"
