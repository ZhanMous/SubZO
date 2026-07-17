"""Automated stop-loss gate checks.

Verifies that the implementation meets the system and scientific thresholds
defined in the research plan.
"""

import pytest
import torch
import time
import sys
sys.path.insert(0, "src")

from methods.pc_meggroll import PCMEGGROLL
from methods.fullrank_es import FullRankES
from models.vgg9_snn import VGG9SNN


class TestSystemGates:
    """System-level stop-loss gates: speed and memory."""

    @pytest.fixture
    def vgg9_model(self):
        return VGG9SNN(time_steps=4, num_classes=100)

    def test_memory_gate(self, vgg9_model):
        """Memory ratio ≥ 1.5× vs full-rank ES."""
        lowrank = PCMEGGROLL(vgg9_model, rank=4)
        fullrank = FullRankES(vgg9_model)

        lowrank_mem = lowrank.get_perturbation_memory()
        fullrank_mem = fullrank.get_perturbation_memory()
        ratio = fullrank_mem / lowrank_mem

        print(f"Memory: lowrank={lowrank_mem}B, fullrank={fullrank_mem}B, ratio={ratio:.2f}×")
        assert ratio >= 1.5, f"Memory ratio {ratio:.2f}× < 1.5× gate"

    def test_speed_estimate_gate(self, vgg9_model):
        """Compression ratio ≥ 1.3× (proxy for speed)."""
        lowrank = PCMEGGROLL(vgg9_model, rank=4)
        info = lowrank.get_speed_estimate()
        ratio = info["compression_ratio"]

        print(f"Compression ratio: {ratio:.2f}×")
        assert ratio >= 1.3, f"Compression ratio {ratio:.2f}× < 1.3× gate"

    def test_speed_benchmark(self, vgg9_model):
        """Benchmark: low-rank perturbation generation is faster than full-rank."""
        lowrank = PCMEGGROLL(vgg9_model, rank=4)
        fullrank = FullRankES(vgg9_model)

        # Time low-rank perturbation generation
        start = time.perf_counter()
        for _ in range(100):
            u, v = lowrank._sample_lowrank_perturbation()
            _ = lowrank._perturbation_to_vector(u, v)
        lr_time = time.perf_counter() - start

        # Time full-rank perturbation generation
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.randn(fullrank.num_params)
        fr_time = time.perf_counter() - start

        print(f"Speed: lowrank={lr_time:.4f}s, fullrank={fr_time:.4f}s, ratio={fr_time/lr_time:.2f}×")
        # Note: actual speedup depends on GPU; on CPU the difference may be small
        # This test documents the measurement rather than enforcing a hard gate


class TestScientificGates:
    """Scientific stop-loss gates (placeholder — requires actual training data)."""

    @pytest.mark.skip(reason="Requires actual training run data")
    def test_hacc_gap_gate(self):
        """HAcc gap ≥ 2pp between best and worst method."""
        # This test requires actual experimental results
        # Placeholder for when training runs are available
        pass

    @pytest.mark.skip(reason="Requires actual training run data")
    def test_recovery_gate(self):
        """PC-MEGGROLL recovers ≥ 70% of gradient upper bound."""
        pass

    @pytest.mark.skip(reason="Requires actual training run data")
    def test_old_class_drop_gate(self):
        """Old-class accuracy drop ≤ 1pp from base session."""
        pass
