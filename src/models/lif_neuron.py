"""LIF (Leaky Integrate-and-Fire) neuron with stable/adaptive channel partitioning.

Ported from SAFA-SNN (https://github.com/ZhangHuiJing2020/SAFA-SNN).
Implements threshold-regulated channel splitting for Few-Shot Class-Incremental Learning.

Key mechanism:
- Stable channels: aggressively corrected back to base firing rates (beta)
- Adaptive channels: allowed to deviate more freely (theta, smaller correction)
- Decay factor exp(-session/tau_decay) ensures diminishing adaptation over sessions
"""

import torch
import torch.nn as nn
import math


class _SpikeZO(torch.autograd.Function):
    """Zeroth-Order surrogate gradient for spike generation.

    Forward: binary threshold (input > 0).
    Backward: gradient estimated via antithetic perturbations.
    """

    @staticmethod
    def forward(ctx, input, delta):
        ctx.save_for_backward(input)
        ctx.delta = delta
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        delta = ctx.delta
        pos = (input + delta > 0).float()
        neg = (input - delta > 0).float()
        grad_estimate = (pos - neg) / (2 * delta)
        return grad_output * grad_estimate, None


class LIFSpike(nn.Module):
    """Leaky Integrate-and-Fire neuron with stable/adaptive channel partitioning.

    Args:
        channels: Number of channels (neurons)
        init_thresh: Initial firing threshold
        tau: Membrane potential leak factor
        adaptive_ratio: Fraction of channels designated as adaptive (rest are stable)
        beta: Threshold correction rate for stable channels
        theta: Threshold correction rate for adaptive channels
        tau_decay: Exponential decay rate for threshold adaptation over sessions
        delta: Perturbation magnitude for ZOO gradient estimation
    """

    def __init__(
        self,
        channels: int,
        init_thresh: float = 1.0,
        tau: float = 0.5,
        adaptive_ratio: float = 0.5,
        beta: float = 0.1,
        theta: float = 0.01,
        tau_decay: float = 100.0,
        delta: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        self.tau = tau
        self.beta = beta
        self.theta = theta
        self.tau_decay = tau_decay
        self.delta = delta

        # Learnable threshold per channel
        self.thresh = nn.Parameter(torch.ones(1, channels, 1, 1) * init_thresh)

        # Adaptive channel mask
        num_adaptive = max(1, int(channels * adaptive_ratio))
        perm = torch.randperm(channels)
        self.register_buffer("adaptive_mask", self._make_mask(channels, perm[:num_adaptive]))

        # Base-session firing rates (populated after base training)
        self.register_buffer("base_rate", torch.zeros(channels))
        self.register_buffer("session", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _make_mask(num_channels: int, indices: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(num_channels, dtype=torch.bool)
        mask[indices] = True
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with membrane dynamics.

        Args:
            x: Input tensor of shape [B, T, C, H, W]

        Returns:
            Spike train of shape [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        mem = torch.zeros(B, C, H, W, device=x.device)
        spikes = []
        for t in range(T):
            mem = mem * self.tau + x[:, t]
            spike = _SpikeZO.apply(mem - self.thresh, self.delta)
            mem = mem * (1 - spike)
            spikes.append(spike)
        return torch.stack(spikes, dim=1)

    @torch.no_grad()
    def update_base_rate(self, spike_rates: torch.Tensor):
        """Store average firing rates from the base session.

        Args:
            spike_rates: Average firing rate per channel, shape [C]
        """
        self.base_rate.copy_(spike_rates)

    @torch.no_grad()
    def adapt_threshold(self, current_rate: torch.Tensor, session: int):
        """Adapt thresholds based on firing rate changes from base session.

        Called during incremental sessions (>0). Stable channels are aggressively
        corrected back to base rates; adaptive channels get more freedom.

        Args:
            current_rate: Current average firing rate per channel [C]
            session: Current session number
        """
        if session == 0:
            return

        self.session.fill_(session)
        decay_factor = math.exp(-session / self.tau_decay)
        rate_diff = current_rate - self.base_rate

        # Stable channels: aggressive correction (beta)
        stable_mask = ~self.adaptive_mask
        self.thresh.data[0, stable_mask, 0, 0] += decay_factor * self.beta * rate_diff[stable_mask]

        # Adaptive channels: gentle correction (theta, smaller)
        self.thresh.data[0, self.adaptive_mask, 0, 0] += (
            decay_factor * self.theta * rate_diff[self.adaptive_mask]
        )

    def get_channel_info(self) -> dict:
        """Return channel partition info for logging."""
        return {
            "total_channels": self.channels,
            "num_adaptive": self.adaptive_mask.sum().item(),
            "num_stable": (~self.adaptive_mask).sum().item(),
            "current_thresh_mean": self.thresh.data.mean().item(),
            "current_thresh_std": self.thresh.data.std().item(),
        }
