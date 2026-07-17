"""Few-shot episode sampler for FSCIL.

Generates N-way K-shot episodes for incremental session training and evaluation.
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator


class FewShotSampler(Sampler):
    """Sampler that generates N-way K-shot episodes.

    Args:
        dataset: Dataset with (image, label) pairs
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        num_episodes: Number of episodes to generate
        seed: Random seed
    """

    def __init__(
        self,
        dataset,
        n_way: int = 5,
        k_shot: int = 5,
        num_episodes: int = 100,
        seed: int = 42,
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_episodes = num_episodes

        # Build class-to-indices
        self.by_class = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in self.by_class:
                self.by_class[label] = []
            self.by_class[label].append(idx)

        self.classes = sorted(self.by_class.keys())
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.num_episodes):
            # Sample N classes
            selected_classes = self.rng.choice(self.classes, self.n_way, replace=False)
            episode_indices = []
            for c in selected_classes:
                # Sample K examples from this class
                class_indices = self.by_class[c]
                selected = self.rng.choice(class_indices, self.k_shot, replace=False)
                episode_indices.extend(selected.tolist())
            yield episode_indices

    def __len__(self) -> int:
        return self.num_episodes
