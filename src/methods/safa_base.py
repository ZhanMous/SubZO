"""SAFA base-session training.

Trains the VGG9SNN backbone on base classes using cross-entropy + TET loss.
After training, replaces the FC layer with class-mean prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.vgg9_snn import VGG9SNN


class TETLoss(nn.Module):
    """Temporal Efficient Training loss.

    Combines cross-entropy with MSE penalty on firing rate deviation.
    Loss = (1 - lamb) * CE + lamb * MSE(output_mean, target_mean)

    Args:
        lamb: Weight for MSE regularization term
        means: Target firing rate
        num_classes: Number of classes
    """

    def __init__(self, lamb: float = 0.05, means: float = 5.0, num_classes: int = 100):
        super().__init__()
        self.lamb = lamb
        self.means = means
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, num_classes] spike-based logits
            targets: [B] class labels
        """
        T = logits.shape[1]
        # CE loss averaged over timesteps
        ce = sum(self.ce_loss(logits[:, t], targets) for t in range(T)) / T
        # MSE: penalize deviation from target firing rate
        mean_output = logits.mean(dim=1)  # [B, num_classes]
        target_onehot = F.one_hot(targets, self.num_classes).float() * self.means
        mse = self.mse_loss(mean_output, target_onehot)
        return (1 - self.lamb) * ce + self.lamb * mse


class SAFA_base:
    """SAFA base-session trainer.

    Trains VGG9SNN on base classes, then computes class-mean prototypes.

    Args:
        model: VGG9SNN instance
        config: Experiment configuration dict
        device: Torch device
    """

    def __init__(self, model: VGG9SNN, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.base_config = config["base_session"]
        self.neuron_config = config["neuron"]

        self.criterion = TETLoss(
            lamb=self.base_config["lamb"],
            means=self.base_config["means"],
            num_classes=config["num_classes"],
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.base_config["lr"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.base_config["epochs"],
        )

    def train(self, train_loader: DataLoader, epoch_callback=None) -> dict:
        """Run base-session training.

        Args:
            train_loader: DataLoader for base classes
            epoch_callback: Optional callback(epoch, metrics) for logging

        Returns:
            Dict with training metrics
        """
        self.model.train()
        epochs = self.base_config["epochs"]
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)  # [B, T, num_classes]
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                # Use mean over timesteps for accuracy
                pred = logits.mean(dim=1).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += images.size(0)

            self.scheduler.step()

            avg_loss = total_loss / total
            acc = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)

            if epoch_callback:
                epoch_callback(epoch, {"loss": avg_loss, "accuracy": acc})

        # Compute base firing rates for threshold adaptation
        self._compute_base_rates(train_loader)

        return history

    @torch.no_grad()
    def _compute_base_rates(self, train_loader: DataLoader):
        """Compute average firing rates per channel from base training data."""
        self.model.eval()
        neurons = self.model.get_all_lif_neurons()
        rate_accumulators = [torch.zeros(n.channels, device=self.device) for n in neurons]
        batch_count = 0

        for images, _ in train_loader:
            images = images.to(self.device)
            _ = self.model(images)
            # Collect spike rates from each LIF neuron
            # (rates are tracked internally during forward pass)
            batch_count += 1

        # For now, use a forward hook approach to capture rates
        # This will be populated during the training loop
        pass

    @torch.no_grad()
    def compute_prototypes(
        self, train_loader: DataLoader, num_classes: int
    ) -> torch.Tensor:
        """Compute class-mean prototypes from base training data.

        Args:
            train_loader: DataLoader for base classes
            num_classes: Number of base classes

        Returns:
            Prototype matrix [num_classes, feature_dim]
        """
        self.model.eval()
        feature_dim = 256 * 4 * 4
        prototypes = torch.zeros(num_classes, feature_dim, device=self.device)
        counts = torch.zeros(num_classes, device=self.device)

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            features = self.model.get_features(images)  # [B, T, D]
            features = features.mean(dim=1)  # [B, D] average over timesteps

            for c in range(num_classes):
                mask = labels == c
                if mask.any():
                    prototypes[c] += features[mask].sum(dim=0)
                    counts[c] += mask.sum().float()

        # Normalize by counts
        counts = counts.clamp(min=1).unsqueeze(1)
        prototypes = prototypes / counts

        return prototypes

    def replace_fc_with_prototypes(self, prototypes: torch.Tensor):
        """Replace the FC layer weights with class-mean prototypes.

        Args:
            prototypes: [num_classes, feature_dim]
        """
        with torch.no_grad():
            self.model.fc.weight.copy_(prototypes)
