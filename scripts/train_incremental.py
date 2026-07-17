"""Incremental-session training with PC-MEGGROLL or baseline.

Runs 8 incremental sessions (5-way 5-shot each) on CIFAR-100.

Usage:
    PYTHONPATH=src python scripts/train_incremental.py \
        --config configs/cifar100_fscil.json \
        --method-config configs/pc_meggroll.json \
        --base-model runs/base_session/model.pt \
        --base-prototypes runs/base_session/prototypes.pt
"""

import argparse
import json
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.vgg9_snn import VGG9SNN
from methods.pc_meggroll import PCMEGGROLL
from methods.fullrank_es import FullRankES
from methods.prototype_classifier import PrototypeClassifier
from data.fscil_sessions import FSCILSessionManager
from utils.metrics import session_report, old_class_accuracy
from utils.seeding import set_seed
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="FSCIL incremental training")
    parser.add_argument("--config", type=str, default="configs/cifar100_fscil.json")
    parser.add_argument("--method-config", type=str, default="configs/pc_meggroll.json")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--base-prototypes", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/incremental")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    method_config = load_config(args.method_config)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "method_config.json", "w") as f:
        json.dump(method_config, f, indent=2)

    # Load model
    neuron_kwargs = config.get("neuron", {})
    model = VGG9SNN(
        time_steps=config["time_steps"],
        num_classes=config["num_classes"],
        neuron_kwargs=neuron_kwargs,
    )
    model.load_state_dict(torch.load(args.base_model, map_location=device))

    # Load base prototypes
    base_prototypes = torch.load(args.base_prototypes, map_location=device)

    # Setup prototype classifier
    classifier = PrototypeClassifier(
        temperature=config["classifier"]["temperature"],
        shift_weight=config["classifier"]["shift_weight"],
    )
    classifier.set_base_prototypes(base_prototypes)

    # Setup data
    session_mgr = FSCILSessionManager(
        data_root="./data",
        num_base_classes=config["base_session"]["num_classes"],
        ways=config["incremental_sessions"]["ways"],
        shots=config["incremental_sessions"]["shots"],
    )

    # Setup optimizer based on method
    method = method_config.get("method", "pc_meggroll")
    if method == "pc_meggroll":
        optimizer = PCMEGGROLL(
            model,
            rank=method_config.get("rank", 4),
            sigma=method_config.get("sigma", 0.02),
            population_size=method_config.get("population_size", 256),
            lr=method_config.get("lr", 0.001),
            antithetic=method_config.get("antithetic", True),
            protection_subspace=base_prototypes,
            blend_weight=method_config.get("subspace", {}).get("blend_weight", 0.1),
        )
        print(f"Using PC-MEGGROLL (rank={optimizer.rank}, compression={optimizer.get_memory_ratio():.1f}×)")
    elif method == "fullrank_es":
        optimizer = FullRankES(
            model,
            sigma=method_config.get("sigma", 0.02),
            population_size=method_config.get("population_size", 256),
            lr=method_config.get("lr", 0.001),
            antithetic=method_config.get("antithetic", True),
        )
        print("Using full-rank ES baseline")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run incremental sessions
    num_sessions = config["incremental_sessions"]["num_sessions"]
    results = []

    for session in range(1, num_sessions + 1):
        print(f"\n{'='*60}")
        print(f"Session {session}/{num_sessions}")
        print(f"{'='*60}")

        # Get few-shot training data
        train_dataset = session_mgr.get_incremental_train_dataset(session)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["incremental_sessions"]["ways"] * config["incremental_sessions"]["shots"],
            shuffle=True,
        )

        # Extract features for new classes
        model.eval()
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                features = model.get_features(images).mean(dim=1)  # [B, D]
                all_features.append(features.cpu())
                all_labels.append(labels)

        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)

        # Add new-class prototypes with subspace projection
        classifier.add_incremental_prototypes(
            all_features, all_labels,
            num_new_classes=config["incremental_sessions"]["ways"],
        )

        # Evaluate on all seen classes
        test_dataset = session_mgr.get_incremental_test_dataset(session)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

        predictions = []
        true_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                features = model.get_features(images).mean(dim=1)
                logits = classifier.classify(features)
                preds = logits.argmax(dim=1)
                predictions.append(preds.cpu())
                true_labels.append(labels)

        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)

        # Generate report
        session_info = session_mgr.get_session_info(session)
        report = session_report(
            predictions, true_labels,
            base_classes=session_mgr.base_classes,
            incremental_classes=session_mgr.incremental_classes,
            current_session=session,
        )
        results.append(report)

        print(f"  Seen accuracy: {report['seen_accuracy']:.4f}")
        print(f"  Base accuracy: {report['base_accuracy']:.4f}")
        print(f"  New accuracy:  {report['new_accuracy']:.4f}")
        print(f"  HAcc:          {report['harmonic_accuracy']:.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  Session {r['session']}: HAcc={r['harmonic_accuracy']:.4f}, "
              f"Base={r['base_accuracy']:.4f}, New={r['new_accuracy']:.4f}")

    # Stop-loss check
    if results:
        final_hacc = results[-1]["harmonic_accuracy"]
        print(f"\nFinal HAcc: {final_hacc:.4f}")
        print(f"Old-class drop: {results[0]['base_accuracy'] - results[-1]['base_accuracy']:.4f}")


if __name__ == "__main__":
    main()
