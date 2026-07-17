"""Tests for LIF neuron with stable/adaptive channel partitioning."""

import pytest
import torch
import sys
sys.path.insert(0, "src")

from models.lif_neuron import LIFSpike, _SpikeZO


class TestSpikeZO:
    """Test zeroth-order surrogate gradient function."""

    def test_forward_threshold(self):
        """Spikes fire when input > 0."""
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
        spike = _SpikeZO.apply(x, 0.5)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.equal(spike, expected)

    def test_backward_gradient(self):
        """Gradient flows through surrogate."""
        x = torch.tensor([0.0], requires_grad=True)
        spike = _SpikeZO.apply(x, 0.5)
        spike.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_backward_zero_gradient_far_from_threshold(self):
        """Gradient is zero far from threshold."""
        x = torch.tensor([-10.0], requires_grad=True)
        spike = _SpikeZO.apply(x, 0.5)
        spike.sum().backward()
        # Gradient should be ~0 for inputs far from threshold
        assert x.grad.abs().sum() < 0.01


class TestLIFSpike:
    """Test LIF neuron module."""

    def test_init(self):
        """Initialization creates correct shapes."""
        lif = LIFSpike(channels=64, adaptive_ratio=0.5)
        assert lif.thresh.shape == (1, 64, 1, 1)
        assert lif.adaptive_mask.shape == (64,)
        assert lif.base_rate.shape == (64,)

    def test_adaptive_mask_ratio(self):
        """Adaptive mask has approximately the requested ratio."""
        lif = LIFSpike(channels=100, adaptive_ratio=0.3)
        num_adaptive = lif.adaptive_mask.sum().item()
        assert 25 <= num_adaptive <= 35  # ~30% with some randomness

    def test_forward_shape(self):
        """Forward produces correct output shape."""
        lif = LIFSpike(channels=16, delta=0.5)
        x = torch.randn(2, 4, 16, 8, 8)  # [B, T, C, H, W]
        out = lif(x)
        assert out.shape == (2, 4, 16, 8, 8)

    def test_forward_output_is_binary(self):
        """Output spikes are binary (0 or 1)."""
        lif = LIFSpike(channels=16, delta=0.5)
        x = torch.randn(2, 4, 16, 8, 8) * 5  # Large input to ensure spikes
        out = lif(x)
        assert ((out == 0) | (out == 1)).all()

    def test_adapt_threshold_stable_channels(self):
        """Stable channels get aggressive threshold correction."""
        lif = LIFSpike(channels=10, adaptive_ratio=0.5, beta=0.1, theta=0.01)
        lif.base_rate.fill_(0.5)

        # Current rate higher than base → threshold should increase
        current_rate = torch.ones(10) * 0.8
        initial_thresh = lif.thresh.data.clone()

        lif.adapt_threshold(current_rate, session=1)

        # Stable channels: thresh += 0.1 * (0.8 - 0.5) = 0.03
        stable_mask = ~lif.adaptive_mask
        expected_change = 0.1 * (0.8 - 0.5) * torch.exp(torch.tensor(-1.0 / 100.0))
        actual_change = (lif.thresh.data[0, stable_mask, 0, 0] - initial_thresh[0, stable_mask, 0, 0]).mean()
        assert abs(actual_change.item() - expected_change.item()) < 0.001

    def test_adapt_threshold_adaptive_channels(self):
        """Adaptive channels get gentle threshold correction."""
        lif = LIFSpike(channels=10, adaptive_ratio=0.5, beta=0.1, theta=0.01)
        lif.base_rate.fill_(0.5)

        current_rate = torch.ones(10) * 0.8
        initial_thresh = lif.thresh.data.clone()

        lif.adapt_threshold(current_rate, session=1)

        # Adaptive channels: thresh += 0.01 * (0.8 - 0.5) = 0.003
        adaptive_mask = lif.adaptive_mask
        expected_change = 0.01 * (0.8 - 0.5) * torch.exp(torch.tensor(-1.0 / 100.0))
        actual_change = (lif.thresh.data[0, adaptive_mask, 0, 0] - initial_thresh[0, adaptive_mask, 0, 0]).mean()
        assert abs(actual_change.item() - expected_change.item()) < 0.001

    def test_adapt_threshold_session0_noop(self):
        """Session 0 adaptation is a no-op."""
        lif = LIFSpike(channels=10)
        initial_thresh = lif.thresh.data.clone()
        lif.adapt_threshold(torch.ones(10), session=0)
        assert torch.equal(lif.thresh.data, initial_thresh)

    def test_adapt_threshold_decay(self):
        """Later sessions have smaller threshold changes."""
        lif = LIFSpike(channels=10, adaptive_ratio=0.5, beta=0.1, tau_decay=100)
        lif.base_rate.fill_(0.5)
        current_rate = torch.ones(10) * 0.8

        # Session 1
        thresh_before_s1 = lif.thresh.data.clone()
        lif.adapt_threshold(current_rate, session=1)
        change_s1 = (lif.thresh.data - thresh_before_s1).abs().mean().item()

        # Session 10 (should be smaller due to decay)
        thresh_before_s10 = lif.thresh.data.clone()
        lif.adapt_threshold(current_rate, session=10)
        change_s10 = (lif.thresh.data - thresh_before_s10).abs().mean().item()

        assert change_s10 < change_s1

    def test_get_channel_info(self):
        """Channel info dict has expected keys."""
        lif = LIFSpike(channels=64, adaptive_ratio=0.3)
        info = lif.get_channel_info()
        assert "total_channels" in info
        assert "num_adaptive" in info
        assert "num_stable" in info
        assert info["total_channels"] == 64
