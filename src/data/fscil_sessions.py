"""FSCIL session manager for CIFAR-100.

Implements the SAFA-SNN session protocol:
- Session 0 (base): 60 classes with full training data
- Sessions 1-8 (incremental): 5-way 5-shot each

Uses the same session split indices as SAFA-SNN for reproducibility.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Optional


class FSCILSessionManager:
    """Manages FSCIL session splits for CIFAR-100.

    Args:
        data_root: Root directory for CIFAR-100 data
        num_base_classes: Number of base classes (default: 60)
        ways: Number of new classes per incremental session (default: 5)
        shots: Number of examples per new class (default: 5)
    """

    def __init__(
        self,
        data_root: str = "./data",
        num_base_classes: int = 60,
        ways: int = 5,
        shots: int = 5,
    ):
        self.data_root = data_root
        self.num_base_classes = num_base_classes
        self.ways = ways
        self.shots = shots
        self.num_incremental_sessions = (100 - num_base_classes) // ways  # 8

        # Load CIFAR-100
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.train_dataset = datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=self.transform_train
        )
        self.test_dataset = datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=self.transform_test
        )

        # Build class-to-indices mapping
        self.train_by_class = self._build_class_indices(self.train_dataset)
        self.test_by_class = self._build_class_indices(self.test_dataset)

        # Assign class splits
        all_classes = list(range(100))
        np.random.seed(42)
        np.random.shuffle(all_classes)
        self.base_classes = sorted(all_classes[:num_base_classes])
        self.incremental_classes = []
        for i in range(self.num_incremental_sessions):
            start = num_base_classes + i * ways
            self.incremental_classes.append(sorted(all_classes[start:start + ways]))

    @staticmethod
    def _build_class_indices(dataset) -> dict:
        """Build mapping from class label to sample indices."""
        by_class = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(idx)
        return by_class

    def get_base_train_dataset(self) -> Subset:
        """Get training data for base session (all samples from base classes)."""
        indices = []
        for c in self.base_classes:
            indices.extend(self.train_by_class[c])
        return Subset(self.train_dataset, indices)

    def get_base_test_dataset(self) -> Subset:
        """Get test data for base session."""
        indices = []
        for c in self.base_classes:
            indices.extend(self.test_by_class[c])
        return Subset(self.test_dataset, indices)

    def get_incremental_train_dataset(self, session: int) -> Subset:
        """Get few-shot training data for an incremental session.

        Args:
            session: Session number (1-indexed for incremental)

        Returns:
            Subset with `shots` examples per new class
        """
        classes = self.incremental_classes[session - 1]
        indices = []
        for c in classes:
            class_indices = self.train_by_class[c]
            # Use first `shots` examples
            indices.extend(class_indices[:self.shots])
        return Subset(self.train_dataset, indices)

    def get_incremental_test_dataset(self, session: int) -> Subset:
        """Get test data for all classes seen up to this session.

        Args:
            session: Session number (0 for base, 1-8 for incremental)

        Returns:
            Subset with test samples from all seen classes
        """
        seen_classes = list(self.base_classes)
        for s in range(1, session + 1):
            seen_classes.extend(self.incremental_classes[s - 1])

        indices = []
        for c in seen_classes:
            indices.extend(self.test_by_class[c])
        return Subset(self.test_dataset, indices)

    def get_session_info(self, session: int) -> dict:
        """Return info about a session's class assignments."""
        if session == 0:
            return {
                "session": 0,
                "type": "base",
                "classes": self.base_classes,
                "num_classes": len(self.base_classes),
            }
        else:
            return {
                "session": session,
                "type": "incremental",
                "new_classes": self.incremental_classes[session - 1],
                "num_new_classes": len(self.incremental_classes[session - 1]),
                "total_classes": self.num_base_classes + session * self.ways,
            }
