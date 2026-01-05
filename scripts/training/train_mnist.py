#!/usr/bin/env python3
import argparse, time, os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow, mlflow.pytorch

class SimpleCNN(nn.Module):
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mnist-training")

    with mlflow.start_run():
        # Phase 7: Tag trial number if running under AutoML
        trial_number = os.getenv("OPTUNA_TRIAL_NUMBER")
        if trial_number:
            mlflow.set_tag("trial_number", trial_number)
            mlflow.set_tag("automl", "true")

        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("device", str(device))
        if torch.cuda.is_available():
            mlflow.log_param("gpu", torch.cuda.get_device_name(0))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)

        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()

        start_time = time.time()

        for epoch in range(args.epochs):
            model.train()
            train_loss, correct, total = 0, 0, 0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)

            train_acc = 100 * correct / total
            avg_loss = train_loss / len(train_loader)

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_correct += output.argmax(1).eq(target).sum().item()
                    val_total += target.size(0)

            val_acc = 100 * val_correct / val_total

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")

        duration = time.time() - start_time
        mlflow.log_metric("duration_seconds", duration)

        cpu_cost = (duration / 3600) * 4 * float(os.getenv("CPU_HOURLY_RATE", "0.05"))
        gpu_cost = ((duration / 3600) * float(os.getenv("GPU_HOURLY_RATE", "0.25"))) if torch.cuda.is_available() else 0
        total_cost = cpu_cost + gpu_cost

        mlflow.log_metric("cpu_cost_usd", cpu_cost)
        mlflow.log_metric("gpu_cost_usd", gpu_cost)
        mlflow.log_metric("total_cost_usd", total_cost)

        try:
            mlflow.pytorch.log_model(model, "model")
        except Exception as e:
            print(f"Warning: Could not log model artifact: {e}")

        print(f"\nComplete! Duration: {duration:.1f}s, Cost: ${total_cost:.4f}, Val Acc: {val_acc:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    train(parser.parse_args())
