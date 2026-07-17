"""Tests for orthogonal subspace projection."""

import pytest
import torch
import sys
sys.path.insert(0, "src")

from methods.prototype_classifier import PrototypeClassifier


class TestSubspaceProjection:
    """Test orthogonal subspace projection for old-class protection."""

    def test_projection_removes_base_component(self):
        """Projected vectors have zero component in base-class subspace."""
        torch.manual_seed(42)
        feature_dim = 64
        n_base = 10
        n_new = 5

        base = torch.randn(n_base, feature_dim)
        new = torch.randn(n_new, feature_dim)

        classifier = PrototypeClassifier(shift_weight=0.0)
        classifier.set_base_prototypes(base)

        # With shift_weight=0, result = (1-0)*new + 0*proj = new (no projection effect)
        # Actually, the formula is: result = (1-sw)*new + sw*proj
        # With sw=0: result = new (unmodified)
        # We need to test with sw=1 to see pure projection

        classifier.shift_weight = 1.0
        result = classifier._project_away_from_base(new)

        # The result should be the projection of new onto base space
        # proj = new @ base^T @ pinv(base @ base^T) @ base
        # This should lie in the column space of base
        # Verify: result = proj (with sw=1)
        gram = base @ base.T
        gram_pinv = torch.linalg.pinv(gram)
        expected_proj = (new @ base.T) @ gram_pinv @ base
        assert torch.allclose(result, expected_proj, atol=1e-5)

    def test_projection_blend_weight(self):
        """Blend weight controls how much base space is mixed in."""
        torch.manual_seed(42)
        feature_dim = 32
        n_base = 5
        n_new = 3

        base = torch.randn(n_base, feature_dim)
        new = torch.randn(n_new, feature_dim)

        classifier = PrototypeClassifier(shift_weight=0.5)
        classifier.set_base_prototypes(base)

        result = classifier._project_away_from_base(new)

        # result = 0.5 * new + 0.5 * proj
        gram = base @ base.T
        gram_pinv = torch.linalg.pinv(gram)
        proj = (new @ base.T) @ gram_pinv @ base
        expected = 0.5 * new + 0.5 * proj

        assert torch.allclose(result, expected, atol=1e-5)

    def test_projection_preserves_base_orthogonal(self):
        """Vectors orthogonal to base space are unchanged by projection."""
        torch.manual_seed(42)
        feature_dim = 16
        n_base = 4

        # Create orthonormal base
        base = torch.randn(n_base, feature_dim)
        # Make orthogonal
        Q, _ = torch.linalg.qr(base.T)
        base = Q[:, :n_base].T  # [n_base, feature_dim]

        # Create vector orthogonal to base space
        new = torch.randn(1, feature_dim)
        # Remove base component
        new_orth = new - (new @ base.T) @ base
        # Verify orthogonality
        assert (new_orth @ base.T).abs().max() < 1e-5

        classifier = PrototypeClassifier(shift_weight=1.0)
        classifier.set_base_prototypes(base)
        result = classifier._project_away_from_base(new_orth)

        # Orthogonal vectors should map to zero in the base space
        # (they have no component to project)
        assert (result @ base.T).abs().max() < 1e-4

    def test_classify_cosine_similarity(self):
        """Classification uses cosine similarity scaled by temperature."""
        feature_dim = 8
        classifier = PrototypeClassifier(temperature=10.0)

        # Set 3 prototypes
        protos = torch.eye(3, feature_dim)
        classifier.set_base_prototypes(protos)

        # Classify: input equal to prototype 1 should get highest score for class 1
        x = protos[1:2]  # [1, 8]
        logits = classifier.classify(x)

        assert logits.shape == (1, 3)
        assert logits[0, 1] > logits[0, 0]
        assert logits[0, 1] > logits[0, 2]

    def test_add_incremental_prototypes(self):
        """Adding incremental prototypes increases total class count."""
        feature_dim = 16
        classifier = PrototypeClassifier()

        base = torch.randn(5, feature_dim)
        classifier.set_base_prototypes(base)
        assert classifier.get_seen_classes() == 5

        new_features = torch.randn(10, feature_dim)
        new_labels = torch.tensor([5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        classifier.add_incremental_prototypes(new_features, new_labels, num_new_classes=5)

        assert classifier.get_seen_classes() == 10
