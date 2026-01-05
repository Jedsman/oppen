#!/usr/bin/env python3
"""
Production Distributed Data Parallel (DDP) Training for MNIST

Uses PyTorchJob environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
automatically injected by Kubeflow Training Operator.

Usage:
    # Single node (no distribution)
    python train_mnist_ddp.py --epochs 3 --lr 0.001 --batch-size 64

    # Distributed via Kubeflow PyTorchJob (automatic env vars)
    kubectl apply -f pytorch_job_manifest.yaml
"""

import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=1)


def setup_distributed():
    """
    Initialize distributed training if RANK environment variable exists.

    PyTorchJob automatically sets:
    - RANK: Global rank (0 to world_size-1)
    - WORLD_SIZE: Total number of workers
    - LOCAL_RANK: Rank within node (for GPU assignment)
    - MASTER_ADDR: Master node address
    - MASTER_PORT: Master node port

    Returns:
        Tuple of (rank, world_size, local_rank) or (0, 1, 0) if not distributed
    """
    if "RANK" not in os.environ:
        # Single process mode
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize process group with NCCL backend for GPUs
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

    return rank, world_size, local_rank


def cleanup_distributed(rank, world_size):
    """Cleanup distributed training resources"""
    if world_size > 1:
        dist.destroy_process_group()


def train(args):
    """
    Main training function.

    Each worker (rank) processes its own shard of data via DistributedSampler.
    Only rank 0 logs to MLflow to avoid duplicate metrics.
    """
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    is_distributed = (world_size > 1)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    # MLflow setup (rank 0 only)
    if is_main_process:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("mnist-training")

    # Print startup info (all ranks)
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"DISTRIBUTED TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Rank: {rank}/{world_size}")
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed}")
        print(f"Backend: {'nccl' if torch.cuda.is_available() else 'gloo'}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size per worker: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"{'='*80}\n")

    # Create dataset with DistributedSampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        './data',
        train=False,
        transform=transform
    )

    # Use DistributedSampler for data sharding
    # Each rank gets a different subset of the data
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = SimpleCNN().to(device)

    # Wrap with DistributedDataParallel if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    # MLflow run (rank 0 only)
    start_time = time.time()

    if is_main_process:
        mlflow.start_run()
        mlflow.log_param("distributed", is_distributed)
        mlflow.log_param("world_size", world_size)
        mlflow.log_param("rank", rank)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size_per_worker", args.batch_size)
        mlflow.log_param("total_batch_size", args.batch_size * world_size)
        mlflow.log_param("device", str(device))
        mlflow.log_param("backend", "nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            mlflow.log_param("gpu", torch.cuda.get_device_name(local_rank))

    # Training loop
    for epoch in range(args.epochs):
        # IMPORTANT: Set epoch for DistributedSampler
        # Ensures different shuffling each epoch
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = output.argmax(1)
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)

            if batch_idx % 100 == 0 and is_main_process:
                print(f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}, "
                      f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                predicted = output.argmax(1)
                test_correct += (predicted == target).sum().item()
                test_total += target.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total

        # Only rank 0 logs metrics
        if is_main_process:
            print(f"\n[Rank 0] Epoch {epoch+1}/{args.epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
            print(f"  Data per rank: {train_total} samples\n")

            # Log to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

    # Final metrics (rank 0 only)
    duration = time.time() - start_time

    if is_main_process:
        # Log final metrics
        mlflow.log_metric("duration_seconds", duration)
        mlflow.log_metric("final_test_accuracy", test_accuracy)
        mlflow.log_metric("final_test_loss", avg_test_loss)

        # Calculate distributed cost (multiply by world_size)
        cpu_cost = (duration / 3600) * 4 * float(os.getenv("CPU_HOURLY_RATE", "0.05"))
        gpu_cost = ((duration / 3600) * float(os.getenv("GPU_HOURLY_RATE", "0.25"))) \
                   if torch.cuda.is_available() else 0

        # Total cost = per-worker cost × world_size
        per_worker_cost = cpu_cost + gpu_cost
        total_cost = per_worker_cost * world_size

        mlflow.log_metric("per_worker_cost_usd", per_worker_cost)
        mlflow.log_metric("total_cost_usd", total_cost)
        mlflow.log_metric("cpu_cost_usd", cpu_cost * world_size)
        mlflow.log_metric("gpu_cost_usd", gpu_cost * world_size)

        # Log model
        try:
            # Use module if DDP wrapped
            model_to_log = model.module if is_distributed else model
            mlflow.pytorch.log_model(model_to_log, "model")
        except Exception as e:
            print(f"Note: Could not log model to MLflow: {e}")

        # End run
        mlflow.end_run()

        print(f"\n{'='*80}")
        print(f"DISTRIBUTED TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
        print(f"World Size: {world_size} workers")
        print(f"Backend: {'nccl' if torch.cuda.is_available() else 'gloo'}")
        print(f"Cost (per worker): ${per_worker_cost:.4f}")
        print(f"Cost (total): ${total_cost:.4f}")
        print(f"\nKey DDP Concepts Demonstrated:")
        print(f"1. Each rank (worker) got 1/{world_size} of the training data")
        print(f"2. Gradients synchronized via DDP.backward()")
        print(f"3. All ranks converged to same model")
        print(f"4. Only rank 0 logged metrics to MLflow")
        print(f"{'='*80}\n")

    # Cleanup
    cleanup_distributed(rank, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed MNIST Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per worker")

    args = parser.parse_args()
    train(args)
