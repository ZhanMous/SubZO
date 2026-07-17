"""VGG9SNN: 9-layer VGG-style Spiking Neural Network.

Ported from SAFA-SNN (https://github.com/ZhangHuiJing2020/SAFA-SNN).

Architecture:
  Block 1: Layer(3→64) → Layer(64→64) → AvgPool2d(2)
  Block 2: Layer(64→128) → Layer(128→128) → AvgPool2d(2)
  Block 3: Layer(128→256) → Layer(256→256) → Layer(256→256) → AvgPool2d(2)
  Classifier: Linear(256*4*4 → 1024) for CIFAR-100 (32×32 input → 4×4 after 3 pools)

Each Layer: Conv2d + BatchNorm2d + LIFSpike

Input: [B, C, H, W] images replicated across T timesteps
Output: [B, T, num_classes] spike-based logits (averaged across T for classification)
"""

from typing import Optional

import torch
import torch.nn as nn
from models.lif_neuron import LIFSpike


class SpikingLayer(nn.Module):
    """Single spiking layer: Conv2d + BatchNorm2d + LIFSpike."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neuron_kwargs: dict,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFSpike(out_channels, **neuron_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with timestep dimension.

        Args:
            x: [B, T, C, H, W]

        Returns:
            [B, T, C', H', W']
        """
        B, T, C, H, W = x.shape
        # Process each timestep through conv+bn, then LIF across all timesteps
        out = []
        for t in range(T):
            out.append(self.bn(self.conv(x[:, t])))
        x = torch.stack(out, dim=1)  # [B, T, C', H', W']
        return self.lif(x)


class VGG9SNN(nn.Module):
    """VGG9-style Spiking Neural Network for CIFAR-100 FSCIL.

    Args:
        time_steps: Number of simulation timesteps T
        num_classes: Number of output classes
        neuron_kwargs: Dict of kwargs passed to each LIFSpike neuron
    """

    def __init__(
        self,
        time_steps: int = 4,
        num_classes: int = 100,
        neuron_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.time_steps = time_steps
        if neuron_kwargs is None:
            neuron_kwargs = {}

        # Block 1: 3 → 64 → 64
        self.block1 = nn.ModuleList([
            SpikingLayer(3, 64, neuron_kwargs),
            SpikingLayer(64, 64, neuron_kwargs),
        ])
        self.pool1 = nn.AvgPool2d(2)

        # Block 2: 64 → 128 → 128
        self.block2 = nn.ModuleList([
            SpikingLayer(64, 128, neuron_kwargs),
            SpikingLayer(128, 128, neuron_kwargs),
        ])
        self.pool2 = nn.AvgPool2d(2)

        # Block 3: 128 → 256 → 256 → 256
        self.block3 = nn.ModuleList([
            SpikingLayer(128, 256, neuron_kwargs),
            SpikingLayer(256, 256, neuron_kwargs),
            SpikingLayer(256, 256, neuron_kwargs),
        ])
        self.pool3 = nn.AvgPool2d(2)

        # Classifier head
        self.fc = nn.Linear(256 * 4 * 4, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W] (CIFAR-100: [B, 3, 32, 32])

        Returns:
            Logits [B, T, num_classes]
        """
        B = x.shape[0]
        # Replicate across timesteps: [B, C, H, W] → [B, T, C, H, W]
        x = x.unsqueeze(1).expand(-1, self.time_steps, -1, -1, -1)

        # Block 1
        for layer in self.block1:
            x = layer(x)
        x = self._apply_pool(x, self.pool1)

        # Block 2
        for layer in self.block2:
            x = layer(x)
        x = self._apply_pool(x, self.pool2)

        # Block 3
        for layer in self.block3:
            x = layer(x)
        x = self._apply_pool(x, self.pool3)

        # Flatten spatial dims: [B, T, C, H, W] → [B, T, C*H*W]
        B, T, C, H, W = x.shape
        x = x.reshape(B, T, C * H * W)

        # Classifier: apply FC per timestep
        logits = torch.stack([self.fc(x[:, t]) for t in range(T)], dim=1)
        return logits  # [B, T, num_classes]

    def _apply_pool(self, x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
        """Apply pooling across timesteps."""
        B, T = x.shape[:2]
        out = []
        for t in range(T):
            out.append(pool(x[:, t]))
        return torch.stack(out, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features (for prototype computation).

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Features [B, T, 256*4*4]
        """
        B = x.shape[0]
        x = x.unsqueeze(1).expand(-1, self.time_steps, -1, -1, -1)

        for layer in self.block1:
            x = layer(x)
        x = self._apply_pool(x, self.pool1)

        for layer in self.block2:
            x = layer(x)
        x = self._apply_pool(x, self.pool2)

        for layer in self.block3:
            x = layer(x)
        x = self._apply_pool(x, self.pool3)

        B, T, C, H, W = x.shape
        return x.reshape(B, T, C * H * W)

    def get_all_lif_neurons(self) -> list[LIFSpike]:
        """Collect all LIFSpike neurons for threshold adaptation."""
        neurons = []
        for layer in self.block1:
            neurons.append(layer.lif)
        for layer in self.block2:
            neurons.append(layer.lif)
        for layer in self.block3:
            neurons.append(layer.lif)
        return neurons
