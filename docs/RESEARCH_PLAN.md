# Research Plan: PC-MEGGROLL

## Objective

Develop **Subspace-Constrained Low-Rank Zeroth-Order Optimization** for Few-Shot Continual Learning in Spiking Neural Networks.

## Method Overview

Combine two existing approaches:
1. **SAFA-SNN** (ICLR 2026): Stable/adaptive channel partitioning + orthogonal subspace projection for FSCIL
2. **EGGROLL** (2025): Low-rank perturbation ES for memory-efficient zeroth-order optimization

### Key Innovation

Constrain EGGROLL's low-rank perturbations to the "plastic but non-interfering" subspace:
- Perturbations are generated as `eps = A @ B^T` (rank-r)
- Projected orthogonally to base-class prototype space
- Fitness-weighted update merged directly into weight matrices
- No additional inference parameters

### Mathematical Formulation

For weight matrix W of shape (a, b):
1. Sample factors: A ~ N(0,I) ∈ R^{a×r}, B ~ N(0,I) ∈ R^{b×r}
2. Perturbation: ε = AB^T ∈ R^{a×b}, rank(ε) ≤ r
3. Subspace projection: ε_proj = ε(I - P^T(PP^T)^{-1}P) where P = base prototypes
4. Fitness evaluation: F(W ± σ·ε_proj)
5. Update: W ← W + (lr/σ) Σ u_i · ε_proj_i

## Target Configuration

**CIFAR-100 / VGG9 / T=4** (code-audited only)

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-100 |
| Backbone | VGG9SNN (9 conv layers, 64→128→256) |
| Timesteps | T=4 |
| Base classes | 60 |
| Incremental | 8 sessions × 5-way 5-shot |
| Base training | 300 epochs, Adam, lr=0.001, cosine annealing |
| Incremental | 100 epochs per session |

## Stop-Loss Criteria

### System Gates (must pass to continue)
- **Speed**: ≥1.3× faster than full-rank ES perturbation generation
- **Memory**: ≥1.5× perturbation memory reduction vs full-rank ES

### Scientific Gates (must pass to continue)
- **HAcc gap**: ≥2pp optimization space must exist
- **Recovery**: PC-MEGGROLL must recover ≥70% of gradient upper bound
- **Old-class drop**: ≤1pp degradation on base classes

### Fallback
If LocalZO/OPZO is uniformly better, pivot to projected LocalZO.

## Milestones

### Phase 1: Foundation (Current)
- [x] Repository scaffold
- [x] VGG9SNN + LIF neuron port
- [x] Prototype classifier + subspace projection
- [x] PC-MEGGROLL optimizer
- [x] Full-rank ES baseline
- [x] Unit tests + stop-loss gate tests
- [ ] Base-session training verification
- [ ] Benchmark: memory/speed gates pass

### Phase 2: Validation
- [ ] Base-session HAcc matches SAFA-SNN reported values
- [ ] Incremental sessions with PC-MEGGROLL
- [ ] Compare against full-rank ES baseline
- [ ] Verify old-class drop ≤ 1pp

### Phase 3: Analysis
- [ ] Rank ablation (1, 2, 4, 8, 16)
- [ ] Subspace projection ablation
- [ ] Population size sensitivity
- [ ] Comparison with LocalZO/OPZO

### Phase 4: Extension (if Phase 2-3 succeed)
- [ ] Transfer to EvoSNN framework
- [ ] Additional datasets (if paper-code discrepancies resolved)
- [ ] Larger models

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SAFA paper-code discrepancy | Only use CIFAR100/VGG9/T=4 (code-audited) |
| EGGROLL doesn't converge in SNN | LocalZO/OPZO fallback |
| Subspace projection too aggressive | Blend weight hyperparameter sweep |
| Low rank too restrictive | Rank ablation study |
