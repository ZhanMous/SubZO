"""K0 System Kill Test: benchmark low-rank + projection on SAFA VGG9 target layer.

Tests whether structured masking + projection preserves EGGROLL's system advantages
on the actual 4096->1024 linear projection in SAFA Spiking-VGG9.

GO criterion: >=1.3x time OR >=1.5x memory reduction vs full-rank projected ES.
Projection overhead <=20% of total step time.

Usage:
    PYTHONPATH=src python scripts/k0_system_kill.py [--device cuda] [--output runs/k0]
"""

import argparse
import json
import time
import torch
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from methods.pc_meggroll import PCMEGGROLL


def make_target_layer(m=4096, n=1024, device="cpu"):
    """Create a target weight matrix matching SAFA VGG9's linear projection."""
    W = torch.randn(m, n, device=device) * 0.01
    return W


def make_adaptive_mask(dim, ratio=0.5, device="cpu"):
    """Create a random adaptive channel mask."""
    mask = torch.zeros(dim, dtype=torch.bool, device=device)
    num_adaptive = int(dim * ratio)
    perm = torch.randperm(dim, device=device)
    mask[perm[:num_adaptive]] = True
    return mask


def make_subspace_basis(n, k, device="cpu"):
    """Create orthonormal basis for old-class feature subspace."""
    X = torch.randn(n, k, device=device)
    Q, _ = torch.linalg.qr(X)
    return Q[:, :k]


def benchmark_step(optimizer, fitness_fn, num_steps=20, label=""):
    """Benchmark optimization steps."""
    # Warmup
    for _ in range(3):
        optimizer.step(fitness_fn)

    times = []
    for _ in range(num_steps):
        start = time.perf_counter()
        _, diag = optimizer.step(fitness_fn)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return {
        "label": label,
        "avg_step_time_s": avg_time,
        "avg_step_time_ms": avg_time * 1000,
        "last_diagnostics": diag,
    }


def main():
    parser = argparse.ArgumentParser(description="K0 System Kill Test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/k0")
    parser.add_argument("--rank", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--m", type=int, default=4096, help="Output dim (SAFA: 4096)")
    parser.add_argument("--n", type=int, default=1024, help="Input dim (SAFA: 1024)")
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32, 64], help="Subspace dims")
    parser.add_argument("--mask-ratio", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"K0 System Kill Test")
    print(f"  Target layer: {args.m} x {args.n}")
    print(f"  Device: {device}")
    print(f"  Population: {args.population}")
    print()

    results = []

    # --- Full-rank baseline ---
    W_full = make_target_layer(args.m, args.n, device)
    U_full = make_subspace_basis(args.n, 64, device)
    mask_out_full = make_adaptive_mask(args.m, 0.5, device)
    mask_in_full = make_adaptive_mask(args.n, 0.5, device)

    fullrank_opt = PCMEGGROLL(
        W_full, rank=min(args.m, args.n), sigma=0.02,
        population_size=args.population, mask_out=mask_out_full,
        mask_in=mask_in_full, subspace_basis=U_full,
    )
    fitness_fn = lambda W: -(W ** 2).sum()

    print("Full-rank projected ES baseline...")
    fullrank_result = benchmark_step(fullrank_opt, fitness_fn, args.steps, "fullrank_projected")
    fullrank_result["system_metrics"] = fullrank_opt.get_system_metrics()
    results.append(fullrank_result)
    print(f"  Time: {fullrank_result['avg_step_time_ms']:.1f} ms")
    print(f"  Memory: {fullrank_result['system_metrics']['full_rank_memory_bytes']:,} bytes")
    print()

    # --- Masked EGGROLL (no projection) ---
    for ratio in args.mask_ratio:
        for rank in args.rank:
            W = make_target_layer(args.m, args.n, device)
            mask_out = make_adaptive_mask(args.m, ratio, device)
            mask_in = make_adaptive_mask(args.n, ratio, device)

            opt = PCMEGGROLL(
                W, rank=rank, sigma=0.02,
                population_size=args.population,
                mask_out=mask_out, mask_in=mask_in,
            )

            label = f"masked_eggroll_r{rank}_mask{int(ratio*100)}"
            print(f"  {label}...")
            result = benchmark_step(opt, fitness_fn, args.steps, label)
            result["system_metrics"] = opt.get_system_metrics()
            results.append(result)
            print(f"    Time: {result['avg_step_time_ms']:.1f} ms, "
                  f"Mem reduction: {result['system_metrics']['memory_reduction_ratio']:.1f}x")

    # --- PC-MEGGROLL (mask + projection) ---
    for ratio in args.mask_ratio:
        for rank in args.rank:
            for k in args.k:
                if k >= args.n:
                    continue
                W = make_target_layer(args.m, args.n, device)
                mask_out = make_adaptive_mask(args.m, ratio, device)
                mask_in = make_adaptive_mask(args.n, ratio, device)
                U = make_subspace_basis(args.n, k, device)

                opt = PCMEGGROLL(
                    W, rank=rank, sigma=0.02,
                    population_size=args.population,
                    mask_out=mask_out, mask_in=mask_in,
                    subspace_basis=U,
                )

                label = f"pc_meggroll_r{rank}_mask{int(ratio*100)}_k{k}"
                print(f"  {label}...")
                result = benchmark_step(opt, fitness_fn, args.steps, label)
                result["system_metrics"] = opt.get_system_metrics()

                # Verify E . U ~ 0
                verify = opt.verify_protection(torch.randn(args.n, 10, device=device))
                result["protection_check"] = verify

                results.append(result)
                print(f"    Time: {result['avg_step_time_ms']:.1f} ms, "
                      f"Mem: {result['system_metrics']['memory_reduction_ratio']:.1f}x, "
                      f"E.U max: {verify['protected_drift']:.2e}")

    # --- GO/STOP evaluation ---
    print()
    print("=" * 60)
    print("K0 Gate Evaluation")
    print("=" * 60)

    fullrank_time = fullrank_result["avg_step_time_ms"]
    fullrank_mem = fullrank_result["system_metrics"]["full_rank_memory_bytes"]

    go_results = []
    for r in results:
        if r["label"] == "fullrank_projected":
            continue
        speed_ratio = fullrank_time / r["avg_step_time_ms"]
        mem_ratio = fullrank_mem / r["system_metrics"]["total_perturbation_memory_bytes"]
        r["speed_ratio"] = speed_ratio
        r["memory_ratio"] = mem_ratio
        r["speed_pass"] = speed_ratio >= 1.3
        r["memory_pass"] = mem_ratio >= 1.5
        r["gate_pass"] = r["speed_pass"] or r["memory_pass"]

        status = "GO" if r["gate_pass"] else "STOP"
        print(f"  {r['label']}: speed={speed_ratio:.2f}x {'PASS' if r['speed_pass'] else 'FAIL'}, "
              f"mem={mem_ratio:.1f}x {'PASS' if r['memory_pass'] else 'FAIL'} -> {status}")

        if r["gate_pass"]:
            go_results.append(r)

    if go_results:
        best = max(go_results, key=lambda x: x.get("speed_ratio", 0) + x.get("memory_ratio", 0))
        print(f"\n  Best: {best['label']}")
        print(f"  -> PROCEED to K1")
    else:
        print(f"\n  All variants FAILED system gate")
        print(f"  -> STOP PC-MEGGROLL, pivot to deep SNN-EGGROLL benchmark")

    # Save
    with open(output_dir / "k0_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'k0_results.json'}")


if __name__ == "__main__":
    main()
