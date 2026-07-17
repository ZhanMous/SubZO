"""Prototype classifier with orthogonal subspace projection.

Implements SAFA-SNN's old-class protection mechanism:
- After base session: FC weights replaced with class-mean prototypes
- During incremental: new-class prototypes projected orthogonally to base-class space
- Classification via cosine similarity with temperature scaling
"""

import torch
import torch.nn.functional as F


class PrototypeClassifier:
    """Prototype-based classifier with subspace projection for FSCIL.

    Args:
        temperature: Softmax temperature for cosine similarity
        shift_weight: Blending weight for subspace projection
    """

    def __init__(self, temperature: float = 16.0, shift_weight: float = 0.1):
        self.temperature = temperature
        self.shift_weight = shift_weight
        self.base_prototypes: torch.Tensor | None = None
        self.all_prototypes: torch.Tensor | None = None

    @torch.no_grad()
    def set_base_prototypes(self, prototypes: torch.Tensor):
        """Store base-class prototypes (computed after base session training).

        Args:
            prototypes: [num_base_classes, feature_dim]
        """
        self.base_prototypes = prototypes.clone()
        self.all_prototypes = prototypes.clone()

    @torch.no_grad()
    def add_incremental_prototypes(
        self,
        new_features: torch.Tensor,
        new_labels: torch.Tensor,
        num_new_classes: int,
    ) -> torch.Tensor:
        """Compute and add new-class prototypes with subspace projection.

        Args:
            new_features: Feature vectors for new-class samples [N, D]
            new_labels: Class labels for new-class samples [N]
            num_new_classes: Number of new classes in this session

        Returns:
            Updated prototype matrix [total_classes, D]
        """
        device = new_features.device
        feature_dim = new_features.shape[1]

        # Compute raw prototypes for new classes
        new_prototypes = torch.zeros(num_new_classes, feature_dim, device=device)
        counts = torch.zeros(num_new_classes, device=device)

        # Map new_labels to local indices (0..num_new_classes-1)
        unique_labels = new_labels.unique(sorted=True)
        for i, label in enumerate(unique_labels):
            mask = new_labels == label
            new_prototypes[i] = new_features[mask].mean(dim=0)
            counts[i] = mask.sum().float()

        # Apply orthogonal subspace projection to protect base-class space
        if self.base_prototypes is not None:
            new_prototypes = self._project_away_from_base(new_prototypes)

        # Append to existing prototypes
        self.all_prototypes = torch.cat([self.all_prototypes, new_prototypes], dim=0)
        return self.all_prototypes

    def _project_away_from_base(self, new_prototypes: torch.Tensor) -> torch.Tensor:
        """Project new-class prototypes orthogonally to base-class subspace.

        Formula:
            proj = (new @ base^T) @ pinv(base @ base^T) @ base
            result = (1 - shift_weight) * new + shift_weight * proj

        The `proj` term is the component of new_prototypes that lies in the
        base-class subspace. By blending it in with shift_weight, we slightly
        pull new prototypes toward the base space (controlled interference),
        while the (1-shift_weight) term preserves new-class identity.

        Args:
            new_prototypes: [num_new_classes, D]

        Returns:
            Projected prototypes [num_new_classes, D]
        """
        base = self.base_prototypes  # [n_base, D]

        # Compute projection coefficients
        # new @ base^T: [n_new, n_base]
        cross = new_prototypes @ base.T
        # base @ base^T: [n_base, n_base]
        gram = base @ base.T
        # pinv for numerical stability
        gram_pinv = torch.linalg.pinv(gram)
        # proj = cross @ gram_pinv @ base: [n_new, D]
        proj = (cross @ gram_pinv) @ base

        # Blend: mostly keep new identity, slightly pull toward base space
        result = (1 - self.shift_weight) * new_prototypes + self.shift_weight * proj
        return result

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features using cosine similarity to prototypes.

        Args:
            features: [B, D] feature vectors

        Returns:
            [B, num_classes] logits (cosine similarity scaled by temperature)
        """
        # Normalize features and prototypes
        features_norm = F.normalize(features, dim=1)
        protos_norm = F.normalize(self.all_prototypes, dim=1)

        # Cosine similarity scaled by temperature
        logits = features_norm @ protos_norm.T * self.temperature
        return logits

    def get_seen_classes(self) -> int:
        """Return number of classes seen so far."""
        return self.all_prototypes.shape[0]

    def get_prototypes(self) -> torch.Tensor:
        """Return current prototype matrix."""
        return self.all_prototypes.clone()
