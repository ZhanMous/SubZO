"""Speed and memory benchmarking: PC-MEGGROLL vs full-rank ES.

Measures:
1. Perturbation memory (bytes per perturbation)
2. Perturbation generation time
3. Compression ratio

Usage:
    PYTHONPATH=src python scripts/benchmark_speed.py
"""

import argparse
import torch
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.vgg9_snn import VGG9SNN
from methods.pc_meggroll import PCMEGGROLL
from methods.fullrank_es import FullRankES


def benchmark_perturbation_generation(opt, num_trials=1000):
    """Benchmark perturbation generation speed."""
    device = next(opt.model.parameters()).device

    if isinstance(opt, PCMEGGROLL):
        start = time.perf_counter()
        for _ in range(num_trials):
            u, v = opt._sample_lowrank_perturbation()
            _ = opt._perturbation_to_vector(u, v)
        elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        for _ in range(num_trials):
            _ = torch.randn(opt.num_params, device=device)
        elapsed = time.perf_counter() - start

    return elapsed / num_trials


def main():
    parser = argparse.ArgumentParser(description="Benchmark PC-MEGGROLL vs full-rank ES")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="runs/benchmark/results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    results = []

    # Full-rank baseline
    model = VGG9SNN(time_steps=4, num_classes=100).to(device)
    fullrank = FullRankES(model)

    fullrank_mem = fullrank.get_perturbation_memory()
    fullrank_time = benchmark_perturbation_generation(fullrank, args.trials)
    fullrank_info = fullrank.get_speed_estimate()

    print(f"Full-rank ES:")
    print(f"  Parameters: {fullrank_info['num_params']:,}")
    print(f"  Perturbation memory: {fullrank_mem:,} bytes")
    print(f"  Generation time: {fullrank_time*1e6:.1f} µs")

    baseline = {
        "method": "fullrank_es",
        "num_params": fullrank_info["num_params"],
        "memory_bytes": fullrank_mem,
        "gen_time_us": fullrank_time * 1e6,
    }

    # Low-rank variants
    for rank in args.ranks:
        model = VGG9SNN(time_steps=4, num_classes=100).to(device)
        lowrank = PCMEGGROLL(model, rank=rank)

        lr_mem = lowrank.get_perturbation_memory()
        lr_time = benchmark_perturbation_generation(lowrank, args.trials)
        lr_info = lowrank.get_speed_estimate()
        mem_ratio = fullrank_mem / lr_mem
        time_ratio = fullrank_time / lr_time if lr_time > 0 else float('inf')

        print(f"\nPC-MEGGROLL (rank={rank}):")
        print(f"  Perturbation memory: {lr_mem:,} bytes ({mem_ratio:.2f}× reduction)")
        print(f"  Generation time: {lr_time*1e6:.1f} µs ({time_ratio:.2f}× {'faster' if time_ratio > 1 else 'slower'})")
        print(f"  Compression ratio: {lr_info['compression_ratio']:.2f}×")

        results.append({
            "method": "pc_meggroll",
            "rank": rank,
            "num_params": lr_info["num_params"],
            "memory_bytes": lr_mem,
            "memory_ratio": mem_ratio,
            "gen_time_us": lr_time * 1e6,
            "time_ratio": time_ratio,
            "compression_ratio": lr_info["compression_ratio"],
        })

    # Check stop-loss gates
    print(f"\n{'='*60}")
    print("Stop-Loss Gate Check")
    print(f"{'='*60}")

    for r in results:
        rank = r["rank"]
        mem_pass = r["memory_ratio"] >= 1.5
        speed_pass = r["time_ratio"] >= 1.3
        print(f"  Rank {rank}: memory {'PASS' if mem_pass else 'FAIL'} ({r['memory_ratio']:.2f}×), "
              f"speed {'PASS' if speed_pass else 'FAIL'} ({r['time_ratio']:.2f}×)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"baseline": baseline, "variants": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
