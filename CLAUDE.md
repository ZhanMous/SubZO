# PC-MEGGROLL

Subspace-constrained low-rank ZO optimization for FSCIL in SNNs.

## Conventions

- All Python source lives under `src/`
- Run via `PYTHONPATH=src python scripts/<script>.py`
- Config files are JSON in `configs/`
- Each run outputs to `runs/<experiment_name>/`
- Python env: use conda environment with PyTorch
- Tests: `PYTHONPATH=src pytest tests/`

## Architecture

- **LIF Neuron**: Leaky Integrate-and-Fire with stable/adaptive channel partitioning
- **VGG9SNN**: 9-layer VGG backbone (64→128→256) with LIF activations, T=4 timesteps
- **Prototype Classifier**: Cosine similarity with orthogonal subspace projection
- **PC-MEGGROLL**: Low-rank perturbation ES constrained to protection subspace

## Key Files

- `src/models/vgg9_snn.py` — VGG9SNN backbone
- `src/models/lif_neuron.py` — LIF neuron with stable/adaptive threshold adaptation
- `src/methods/pc_meggroll.py` — Core PC-MEGGROLL optimizer
- `src/methods/prototype_classifier.py` — Prototype computation + subspace projection
- `src/methods/safa_base.py` — Base-session SAFA training

## Target

CIFAR100/VGG9/T=4 only. Other protocols have known discrepancies.

## Stop-Loss

System: ≥1.3× speed or ≥1.5× memory reduction vs full-rank ES.
Science: ≥2pp HAcc gap, ≥70% recovery, ≤1pp old-class drop.
If LocalZO/OPZO is better, pivot to projected LocalZO.
