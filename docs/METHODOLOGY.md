# Methodology: PC-MEGGROLL

## Mathematical Framework

### 1. Problem Setting: Few-Shot Class-Incremental Learning (FSCIL)

Given:
- Base session: D_base = {(x_i, y_i)} with y_i ∈ C_base (60 classes for CIFAR-100)
- Incremental session t: D_t = {(x_i, y_i)} with y_i ∈ C_t (5 new classes, 5 examples each)
- Goal: learn new classes without forgetting base classes

### 2. SAFA-SNN: Sparsity-Aware Feature Adaptation

#### 2.1 Stable/Adaptive Channel Partitioning

LIF neuron with threshold regulation:
- Membrane: m(t) = τ·m(t-1) + x(t)
- Spike: s(t) = H(m(t) - θ) where H is Heaviside step function
- Reset: m(t) = m(t)·(1 - s(t))

Channel partitioning:
- Adaptive channels: randomly selected (adaptive_ratio of total)
- Stable channels: remaining channels

Threshold adaptation (session > 0):
```
δ = exp(-session / τ_decay)
θ_stable += δ · β · (rate_current - rate_base)     # β = 0.1 (aggressive)
θ_adaptive += δ · θ_coeff · (rate_current - rate_base)  # θ_coeff = 0.01 (gentle)
```

#### 2.2 Prototype Classifier

After base training, replace FC weights with class-mean prototypes:
```
p_c = (1/|D_c|) Σ_{(x,y)∈D_c, y=c} f(x)
```

Classification via cosine similarity:
```
P(y=c|x) = softmax(sim(f(x), p_c) / T)
```

#### 2.3 Orthogonal Subspace Projection

For new-class prototypes, project away from base-class space:
```
proj = p_new · P_base^T · (P_base · P_base^T)^{-1} · P_base
p_new_proj = (1 - α) · p_new + α · proj
```

where α = shift_weight controls blending.

### 3. EGGROLL: Low-Rank Perturbation ES

#### 3.1 Standard ES Gradient Estimator

```
∇_θ E[F(θ)] ≈ (1/Nσ) Σ_{i=1}^N F(θ + σε_i) · ε_i
```

where ε_i ~ N(0, I_D).

Problem: O(D) memory per perturbation, O(D) compute per forward pass.

#### 3.2 Low-Rank Factorization

Instead of full D-dimensional noise, sample:
```
u ~ N(0, I_m), v ~ N(0, I_n), where m·n ≈ D
ε = u · v^T ∈ R^{m×n}, rank(ε) ≤ r
```

Memory: O((m+n)·r) instead of O(m·n).

#### 3.3 Antithetic Sampling

For each (u, v), evaluate both +ε and -ε:
```
ε+ = u · v^T
ε- = (-u) · v^T = -(u · v^T)
```

### 4. PC-MEGGROLL: Subspace-Constrained Low-Rank ZO

#### 4.1 Core Algorithm

```
Input: Model θ, base prototypes P_base, rank r, population N, σ, lr
For each incremental session:
  For each optimization step:
    1. Sample N/2 pairs (u_i, v_i) ~ N(0, I)
    2. For each pair:
       a. ε_i = u_i · v_i^T (low-rank perturbation)
       b. Flatten to ε_i ∈ R^D
       c. Project: ε_i ← ε_i · (I - P^T(PP^T)^{-1}P)
       d. Evaluate: f+ = F(θ + σ·ε_i), f- = F(θ - σ·ε_i)
    3. Compute utilities: u_i = rank_utility([f+, f-])
    4. Gradient: g = (1/Nσ) Σ u_i · ε_i
    5. Update: θ ← θ + lr · g
```

#### 4.2 Protection Subspace Projection

The projection matrix removes components in the base-class prototype space:
```
M_proj = I - P^T(PP^T)^{-1}P
```

Properties:
- M_proj is idempotent: M_proj² = M_proj
- M_proj is symmetric: M_proj = M_proj^T
- P · M_proj = 0 (base prototypes are in the null space)

This ensures perturbations do not interfere with base-class representations.

#### 4.3 Fitness Shaping

Rank-based utilities (Wierstra et al., 2014):
```
u_k = max(0, log(N/2 + 1) - log(k))
u_k ← u_k / Σ u_k (normalize)
u_k ← u_k - 1/N (center)
```

### 5. Memory and Speed Analysis

#### 5.1 Memory per Perturbation

| Method | Memory | CIFAR100/VGG9 (D≈2.4M) |
|--------|--------|------------------------|
| Full-rank ES | O(D) | ~9.2 MB |
| PC-MEGGROLL (r=4) | O((m+n)·r) | ~19.6 KB |
| **Reduction** | | **~470×** |

#### 5.2 Perturbation Generation

| Method | Operations | Notes |
|--------|-----------|-------|
| Full-rank | D samples from N(0,1) | |
| Low-rank | (m+n)·r samples + rank-r outer product | Shared base matmul |

#### 5.3 GPU Utilization

EGGROLL decomposes the forward pass into:
1. One shared large matmul (high utilization)
2. Tiny per-member corrections (negligible)

This achieves up to 91% of pure batch inference throughput.

### 6. Convergence Properties

Theorem (from EGGROLL paper): As D → ∞, the low-rank ES gradient estimator converges to the full-rank estimator with error O(1/r).

For PC-MEGGROLL, the subspace projection does not affect convergence rates because:
1. Projection is a linear operation
2. It removes only directions that are uninformative for new-class learning
3. The remaining subspace retains full gradient information for plastic directions
