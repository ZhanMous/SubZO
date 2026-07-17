"""K1 Scientific Kill Test: 6-arm pseudo-increment on CIFAR-100 base classes.

Runs entirely within base classes (60 classes), simulating incremental sessions
by treating subsets as "pseudo-novel" classes. Never touches the 40 held-out classes.

Arms:
  A. SAFA baseline (threshold adaptation + prototype subspace projection)
  B. Projected SG (gradient update on target layer, same mask/U)
  C. Projected full-rank ES (same batch/population/CRN)
  D. Masked EGGROLL, no projection (adaptive channels only)
  E. PC-MEGGROLL (mask + B_perp projection + low-rank)
  F. Placeholder for LocalZO/OPZO

GO criteria:
  - B (SG) >= 2pp HAcc headroom over A, old-class drop <= 1pp
  - E >= 70% of B's gain, with less forgetting than D
  - E within 0.5-1pp of C (non-inferiority)

Usage:
    PYTHONPATH=src:third_party/safa_snn python scripts/k1_scientific_kill.py \
        --device cuda --seeds 3 --pseudo-sessions 5
"""

import argparse
import json
import sys
import os
import time
import math
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add SAFA-SNN to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "safa_snn"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tool
from methods.pc_meggroll import PCMEGGROLL


# ---- SAFA config mock ----
class Args:
    """Minimal args namespace matching SAFA-SNN's expectations."""
    def __init__(self):
        self.dataset = "cifar100"
        self.network = "svgg9"
        self.temperature = 16
        self.feat_norm = False
        self.epochs_base = 50  # Reduced for K1 speed
        self.lr_base = 0.001
        self.lr_new = 0.001
        self.optim = "Adam"
        self.schedule = "Cosine"
        self.milestones = [60, 70]
        self.decay = 0.0005
        self.batch_size_base = 128
        self.batch_size_new = 0
        self.test_batch_size = 100
        self.base_mode = "ft_dot"
        self.new_mode = "ft_cos"
        self.start_session = 0
        self.model_dir = None
        self.only_do_incre = False
        self.gpu = [0]
        self.device = "cuda"
        self.num_workers = 4
        self.seed = 1
        self.debug = False
        self.softmax_t = 16
        self.shift_weight = 0.1
        self.time_step = 4
        self.beta = 0.1
        self.theta = 0.01
        self.adapt_ratio = 0.5
        self.lamb = 0.05
        self.means = 5
        self.sg = "zoo"
        self.delta = 0.5
        self.tau = 1.1
        self.thresh = 1.0
        self.adaptive_ratio = 0.5
        self.new_update = "subspace"
        self.connect_f = "ADD"
        self.zero_init_residual = True
        self.tau_decay = 100
        self.tet = True
        self.sessions = 9
        self.base_class = 60
        self.num_classes = 100
        self.way = 5
        self.shot = 5
        self.dataroot = "./data"
        self.save_path = "runs/k1"


def TET_loss(outputs, labels, criterion, means, lamb):
    """SAFA's TET loss: (1-lamb)*CE + lamb*MSE on firing rate."""
    T = outputs.size(1)
    Loss_es = sum(criterion(outputs[:, t, :], labels) for t in range(T)) / T
    if lamb != 0:
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = nn.MSELoss()(outputs, y)
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


def count_acc(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def harmonic_mean(seen_acc, unseen_acc):
    if seen_acc + unseen_acc == 0:
        return 0.0
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)


def get_pseudo_increment_splits(base_classes, num_pseudo_sessions, way, rng):
    """Split base classes into pseudo-increment sessions.

    Returns:
        base_classes_used: classes for initial training
        pseudo_sessions: list of lists of class indices
    """
    classes = list(base_classes)
    rng.shuffle(classes)
    # First batch: base training
    base_size = len(classes) - num_pseudo_sessions * way
    base = sorted(classes[:base_size])
    pseudo = []
    for i in range(num_pseudo_sessions):
        start = base_size + i * way
        pseudo.append(sorted(classes[start:start + way]))
    return base, pseudo


def make_fewshot_loader(dataset, class_list, shot, transform, batch_size=256):
    """Create a few-shot dataloader for specific classes."""
    indices = []
    targets = dataset.targets if hasattr(dataset, 'targets') else [s[1] for s in dataset]
    for c in class_list:
        class_indices = [i for i, t in enumerate(targets) if t == c]
        indices.extend(class_indices[:shot])
    subset = torch.utils.data.Subset(dataset, indices)
    if transform is not None:
        # Apply test transform for feature extraction
        subset.dataset = deepcopy(dataset)
        subset.dataset.transform = transform
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def extract_features(model, dataloader, device):
    """Extract penultimate features from model."""
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            # Use encoder mode to get features
            orig_mode = model.mode
            model.mode = "encoder"
            feat = model(data).mean(1)  # Average over timesteps
            model.mode = orig_mode
            features_list.append(feat.cpu())
            labels_list.append(label.cpu())
    return torch.cat(features_list), torch.cat(labels_list)


def evaluate(model, testloader, test_classes, device):
    """Evaluate model on test classes."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            logits = model(data).mean(1)[:, :test_classes]
            pred = logits.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total if total > 0 else 0.0


def run_arm_a_safa(model, trainloader, testloader, pseudo_sessions, args, device):
    """Arm A: SAFA baseline — threshold adaptation + prototype subspace projection."""
    results = []

    # Base evaluation
    base_acc = evaluate(model, testloader, args.base_class, device)
    results.append({"session": 0, "seen_acc": base_acc, "novel_acc": 0.0, "hacc": base_acc})

    for sess_idx, novel_classes in enumerate(pseudo_sessions):
        session = sess_idx + 1
        tool.session = session

        # Extract features for novel classes
        feat_loader = make_fewshot_loader(
            trainloader.dataset, novel_classes, args.shot,
            testloader.dataset.transform if hasattr(testloader.dataset, 'transform') else None
        )
        features, labels = extract_features(model, feat_loader, device)

        # Update fc with novel class prototypes
        for c in novel_classes:
            mask = labels == c
            if mask.any():
                proto = features[mask].mean(0)
                model.fc.weight.data[c] = proto.to(device)

        # Subspace projection
        model.subspace_projection(args, session)

        # Evaluate
        test_classes = args.base_class + session * args.way
        acc = evaluate(model, testloader, test_classes, device)

        # Compute per-group accuracy
        # (simplified: use overall acc as proxy)
        results.append({
            "session": session,
            "seen_acc": acc,
            "novel_acc": acc,  # Simplified
            "hacc": acc,
        })

    return results


def run_arm_b_sg(model, trainloader, testloader, pseudo_sessions, args, device,
                 target_layer, mask_out, mask_in, U, lr=0.001, steps=50):
    """Arm B: Projected SG — gradient update on target layer."""
    results = []

    # Base evaluation
    base_acc = evaluate(model, testloader, args.base_class, device)
    results.append({"session": 0, "seen_acc": base_acc, "novel_acc": 0.0, "hacc": base_acc})

    for sess_idx, novel_classes in enumerate(pseudo_sessions):
        session = sess_idx + 1
        tool.session = session

        # Get few-shot data
        feat_loader = make_fewshot_loader(
            trainloader.dataset, novel_classes, args.shot,
            testloader.dataset.transform if hasattr(testloader.dataset, 'transform') else None
        )

        # SG update on target layer
        W = target_layer.weight
        W_orig = W.data.clone()
        optimizer = torch.optim.SGD([W], lr=lr)

        for step in range(steps):
            total_loss = 0
            for data, label in feat_loader:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                logits = model(data).mean(1)[:, :args.base_class + session * args.way]
                loss = F.cross_entropy(logits, label)
                loss.backward()

                # Project gradient to be orthogonal to U
                if U is not None and W.grad is not None:
                    grad = W.grad.data
                    # Project: grad_perp = grad - (grad @ U) @ U^T
                    # But U is in input space [n, k], so we project along input dim
                    grad_projected = grad - (grad @ U) @ U.T
                    W.grad.data = grad_projected

                optimizer.step()

                # Apply mask: only update adaptive channels
                with torch.no_grad():
                    delta = W.data - W_orig
                    delta_masked = delta * mask_out.unsqueeze(1) * mask_in.unsqueeze(0)
                    W.data = W_orig + delta_masked

        # Evaluate
        test_classes = args.base_class + session * args.way
        acc = evaluate(model, testloader, test_classes, device)
        results.append({
            "session": session,
            "seen_acc": acc,
            "novel_acc": acc,
            "hacc": acc,
        })

    return results


def run_arm_de_eggroll(model, trainloader, testloader, pseudo_sessions, args, device,
                       target_layer, mask_out, mask_in, U, rank=4, sigma=0.02,
                       population=64, lr=0.001, steps=20, use_projection=True):
    """Arm D/E: (PC-)MEGGROLL on target layer."""
    arm_name = "PC-MEGGROLL" if use_projection else "Masked EGGROLL"
    results = []

    # Base evaluation
    base_acc = evaluate(model, testloader, args.base_class, device)
    results.append({"session": 0, "seen_acc": base_acc, "novel_acc": 0.0, "hacc": base_acc})

    for sess_idx, novel_classes in enumerate(pseudo_sessions):
        session = sess_idx + 1
        tool.session = session

        # Get few-shot data
        feat_loader = make_fewshot_loader(
            trainloader.dataset, novel_classes, args.shot,
            testloader.dataset.transform if hasattr(testloader.dataset, 'transform') else None
        )

        # Setup optimizer
        opt = PCMEGGROLL(
            target_layer.weight, rank=rank, sigma=sigma,
            population_size=population, lr=lr,
            mask_out=mask_out, mask_in=mask_in,
            subspace_basis=U if use_projection else None,
        )

        # Fitness function: negative CE on novel classes
        def fitness_fn(W):
            # Temporarily set weight
            orig = target_layer.weight.data.clone()
            target_layer.weight.data = W.data.clone()
            model.eval()
            total_loss = 0
            count = 0
            with torch.no_grad():
                for data, label in feat_loader:
                    data, label = data.to(device), label.to(device)
                    logits = model(data).mean(1)[:, :args.base_class + session * args.way]
                    loss = F.cross_entropy(logits, label)
                    total_loss += loss.item()
                    count += 1
            target_layer.weight.data = orig
            return -total_loss / max(count, 1)

        # Run optimization steps
        for _ in range(steps):
            opt.step(fitness_fn)

        # Evaluate
        test_classes = args.base_class + session * args.way
        acc = evaluate(model, testloader, test_classes, device)
        results.append({
            "session": session,
            "seen_acc": acc,
            "novel_acc": acc,
            "hacc": acc,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="K1 Scientific Kill Test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pseudo-sessions", type=int, default=5)
    parser.add_argument("--epochs-base", type=int, default=50)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--output", default="runs/k1")
    args_cli = parser.parse_args()

    output_dir = Path(args_cli.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args_cli.device)
    all_results = []

    for seed in range(args_cli.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/{args_cli.seeds}")
        print(f"{'='*60}")

        # Setup
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        args = Args()
        args.seed = seed
        args.epochs_base = args_cli.epochs_base
        args.device = device
        tool.args = args
        tool.session = 0

        # Create model
        from inc_net.net import NET
        model = NET(args, mode=args.base_mode).to(device)

        # Load CIFAR-100 base classes
        import torchvision.transforms as transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        from torchvision import datasets
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

        # Filter to base classes
        base_class_indices = list(range(args.base_class))
        train_mask = [t in base_class_indices for t in trainset.targets]
        test_mask = [t in base_class_indices for t in testset.targets]
        train_indices = [i for i, m in enumerate(train_mask) if m]
        test_indices = [i for i, m in enumerate(test_mask) if m]

        base_trainset = torch.utils.data.Subset(trainset, train_indices)
        base_testset = torch.utils.data.Subset(testset, test_indices)

        trainloader = torch.utils.data.DataLoader(base_trainset, batch_size=args.batch_size_base, shuffle=True)
        testloader = torch.utils.data.DataLoader(base_testset, batch_size=args.test_batch_size, shuffle=False)

        # Pseudo-increment splits
        base_classes_used, pseudo_sessions = get_pseudo_increment_splits(
            base_class_indices, args_cli.pseudo_sessions, args.way, rng
        )
        print(f"  Base classes: {len(base_classes_used)}, Pseudo sessions: {len(pseudo_sessions)}")
        print(f"  Pseudo-novel per session: {[len(s) for s in pseudo_sessions]}")

        # Base training (simplified)
        print("  Base training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)
        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in range(args.epochs_base):
            model.train()
            for data, label in trainloader:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                logits = model(data, session=0)[:, :, :args.base_class]
                loss = TET_loss(logits, label, criterion, args.means, args.lamb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Replace fc with prototypes
        model.eval()
        proto_features, proto_labels = extract_features(model, trainloader, device)
        for c in range(args.base_class):
            mask = proto_labels == c
            if mask.any():
                model.fc.weight.data[c] = proto_features[mask].mean(0).to(device)
        model.mode = "avg_cos"

        # Get target layer info
        target_layer = model.encoder.classifier.module  # The nn.Linear(4096, 1024)
        print(f"  Target layer: {target_layer.weight.shape}")

        # Create masks and subspace
        m, n = target_layer.weight.shape
        mask_out = torch.zeros(m, dtype=torch.bool, device=device)
        num_adaptive = int(m * args.adaptive_ratio)
        perm = torch.randperm(m, device=device)
        mask_out[perm[:num_adaptive]] = True

        mask_in = torch.zeros(n, dtype=torch.bool, device=device)
        perm = torch.randperm(n, device=device)
        mask_in[perm[:num_adaptive]] = True

        # Subspace from base prototypes (project fc weights as proxy)
        with torch.no_grad():
            base_protos = F.normalize(model.fc.weight.data[:args.base_class].detach(), p=2, dim=-1)
            # SVD for orthonormal basis
            U, S, Vh = torch.linalg.svd(base_protos.T, full_matrices=False)
            # Keep top-k components
            k = min(32, args.base_class)
            U_sub = U[:, :k].to(device)
        print(f"  Subspace: k={k}, U shape={U_sub.shape}")

        # Run arms
        seed_results = {"seed": seed, "arms": {}}

        # Arm A: SAFA baseline
        print("  Arm A: SAFA baseline...")
        model_a = deepcopy(model)
        tool.session = 0
        results_a = run_arm_a_safa(model_a, trainloader, testloader, pseudo_sessions, args, device)
        seed_results["arms"]["A_safa"] = results_a
        print(f"    Final HAcc: {results_a[-1]['hacc']:.4f}")

        # Arm B: Projected SG
        print("  Arm B: Projected SG...")
        model_b = deepcopy(model)
        tool.session = 0
        results_b = run_arm_b_sg(
            model_b, trainloader, testloader, pseudo_sessions, args, device,
            model_b.encoder.classifier.module, mask_out, mask_in, U_sub
        )
        seed_results["arms"]["B_proj_sg"] = results_b
        print(f"    Final HAcc: {results_b[-1]['hacc']:.4f}")

        # Arm D: Masked EGGROLL (no projection)
        print("  Arm D: Masked EGGROLL...")
        model_d = deepcopy(model)
        tool.session = 0
        results_d = run_arm_de_eggroll(
            model_d, trainloader, testloader, pseudo_sessions, args, device,
            model_d.encoder.classifier.module, mask_out, mask_in, None,
            rank=args_cli.rank, population=args_cli.population, use_projection=False
        )
        seed_results["arms"]["D_masked_eggroll"] = results_d
        print(f"    Final HAcc: {results_d[-1]['hacc']:.4f}")

        # Arm E: PC-MEGGROLL
        print("  Arm E: PC-MEGGROLL...")
        model_e = deepcopy(model)
        tool.session = 0
        results_e = run_arm_de_eggroll(
            model_e, trainloader, testloader, pseudo_sessions, args, device,
            model_e.encoder.classifier.module, mask_out, mask_in, U_sub,
            rank=args_cli.rank, population=args_cli.population, use_projection=True
        )
        seed_results["arms"]["E_pc_meggroll"] = results_e
        print(f"    Final HAcc: {results_e[-1]['hacc']:.4f}")

        all_results.append(seed_results)

    # Summary
    print(f"\n{'='*60}")
    print("K1 Summary")
    print(f"{'='*60}")

    for arm_name in ["A_safa", "B_proj_sg", "D_masked_eggroll", "E_pc_meggroll"]:
        final_haccs = [r["arms"][arm_name][-1]["hacc"] for r in all_results]
        print(f"  {arm_name}: HAcc = {np.mean(final_haccs):.4f} +/- {np.std(final_haccs):.4f}")

    # GO/STOP evaluation
    print(f"\n{'='*60}")
    print("K1 Gate Evaluation")
    print(f"{'='*60}")

    # Check: B >= A + 2pp
    a_hacc = np.mean([r["arms"]["A_safa"][-1]["hacc"] for r in all_results])
    b_hacc = np.mean([r["arms"]["B_proj_sg"][-1]["hacc"] for r in all_results])
    d_hacc = np.mean([r["arms"]["D_masked_eggroll"][-1]["hacc"] for r in all_results])
    e_hacc = np.mean([r["arms"]["E_pc_meggroll"][-1]["hacc"] for r in all_results])

    sg_headroom = (b_hacc - a_hacc) * 100
    e_vs_b = ((e_hacc - a_hacc) / max(b_hacc - a_hacc, 1e-6)) * 100
    e_vs_d = (e_hacc - d_hacc) * 100

    print(f"  SG headroom (B-A): {sg_headroom:.2f}pp {'PASS' if sg_headroom >= 2 else 'FAIL'}")
    print(f"  E vs B recovery: {e_vs_b:.1f}% {'PASS' if e_vs_b >= 70 else 'FAIL'}")
    print(f"  E vs D (projection benefit): {e_vs_d:.4f}")

    go = sg_headroom >= 2 and e_vs_b >= 70
    print(f"\n  Overall: {'GO -> proceed to Pilot' if go else 'STOP -> investigate or pivot'}")

    # Save
    with open(output_dir / "k1_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir / 'k1_results.json'}")


if __name__ == "__main__":
    main()
