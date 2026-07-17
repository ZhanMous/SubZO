# SubZO

**Subspace-Constrained Low-Rank Zeroth-Order Optimization for Few-Shot Continual Learning in Spiking Neural Networks**

## Overview

Combines [SAFA-SNN](https://arxiv.org/abs/2510.03648)'s stable/adaptive channels and old-class protection subspace with [EGGROLL](https://openreview.net/forum?id=bfVJ4GsHrO)'s low-rank perturbation mechanism. The key idea: constrain low-rank perturbations to the "plastic but non-interfering" direction, update existing weight matrices in-place, and add zero inference parameters.

## Quickstart

```bash
conda activate <your-env>
pip install -r requirements.txt

# Base session training
PYTHONPATH=src python scripts/train_base.py --config configs/cifar100_fscil.json

# Incremental sessions with PC-MEGGROLL
PYTHONPATH=src python scripts/train_incremental.py --config configs/cifar100_fscil.json --method pc_meggroll

# Full pipeline
bash scripts/run_full_pipeline.sh
```

## Stop-Loss Criteria

| Gate | Threshold | Meaning |
|------|-----------|---------|
| System: speed | ≥1.3× vs full-rank ES | Low-rank perturbation must be faster |
| System: memory | ≥1.5× reduction | Perturbation memory must shrink |
| Science: HAcc gap | ≥2pp | Room for improvement must exist |
| Science: recovery | ≥70% of gradient upper bound | PC-MEGGROLL must recover most of the signal |
| Science: old-class drop | ≤1pp | Old classes must not degrade |

If LocalZO/OPZO is uniformly better, pivot to projected LocalZO.

## Target Configuration

**CIFAR100 / VGG9 / T=4** only. Other SAFA data protocols have known paper-code discrepancies.

## Project Structure

```
pc-meggroll/
├── configs/         # Experiment configurations
├── src/
│   ├── models/      # VGG9SNN + LIF neuron
│   ├── methods/     # SAFA, PC-MEGGROLL, full-rank ES, prototype classifier
│   ├── data/        # FSCIL session splits, few-shot sampler
│   └── utils/       # Metrics, seeding, config
├── scripts/         # Training and benchmarking entry points
├── tests/           # Unit tests + stop-loss gate checks
└── docs/            # Research plan, methodology
```
