"""Base-session training for SAFA-SNN.

Trains VGG9SNN on 60 base classes of CIFAR-100, then computes class-mean prototypes.

Usage:
    PYTHONPATH=src python scripts/train_base.py --config configs/cifar100_fscil.json
"""

import argparse
import json
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.vgg9_snn import VGG9SNN
from methods.safa_base import SAFA_base
from data.fscil_sessions import FSCILSessionManager
from utils.seeding import set_seed
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="SAFA base-session training")
    parser.add_argument("--config", type=str, default="configs/cifar100_fscil.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/base_session")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = torch.device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Initialize model
    neuron_kwargs = config.get("neuron", {})
    model = VGG9SNN(
        time_steps=config["time_steps"],
        num_classes=config["num_classes"],
        neuron_kwargs=neuron_kwargs,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    session_mgr = FSCILSessionManager(
        data_root="./data",
        num_base_classes=config["base_session"]["num_classes"],
        ways=config["incremental_sessions"]["ways"],
        shots=config["incremental_sessions"]["shots"],
    )
    train_dataset = session_mgr.get_base_train_dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["base_session"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Train
    trainer = SAFA_base(model, config, device)
    print(f"Starting base-session training for {config['base_session']['epochs']} epochs...")

    def log_epoch(epoch, metrics):
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['base_session']['epochs']}: "
                  f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")

    history = trainer.train(train_loader, epoch_callback=log_epoch)

    # Compute and store prototypes
    print("Computing class-mean prototypes...")
    prototypes = trainer.compute_prototypes(train_loader, config["base_session"]["num_classes"])
    trainer.replace_fc_with_prototypes(prototypes)

    # Save model and prototypes
    torch.save(model.state_dict(), output_dir / "model.pt")
    torch.save(prototypes, output_dir / "prototypes.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Base session complete. Final accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
