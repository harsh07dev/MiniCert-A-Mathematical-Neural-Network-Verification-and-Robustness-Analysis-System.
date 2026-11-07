"""Train MiniCertNet on MNIST and validate the saved checkpoint."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_loader import load_model
from models.network import MiniCertNet
from models.train import evaluate, train_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "saved_model.pth")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "training_report.md")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    history: dict[str, list[float]],
    test_metrics: dict[str, float],
    prediction_matches: int,
    test_samples: int,
    sample_prediction: int,
    sample_target: int,
    elapsed_seconds: float,
) -> None:
    headers = ["Epoch", "Training loss", "Training accuracy"]
    if history["val_loss"]:
        headers.extend(["Validation loss", "Validation accuracy"])
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    for epoch in range(args.epochs):
        values = [
            str(epoch + 1),
            f"{history['train_loss'][epoch]:.6f}",
            f"{history['train_acc'][epoch] * 100:.2f}%",
        ]
        if history["val_loss"]:
            values.extend(
                [
                    f"{history['val_loss'][epoch]:.6f}",
                    f"{history['val_acc'][epoch] * 100:.2f}%",
                ]
            )
        table.append("| " + " | ".join(values) + " |")

    report = f"""# MiniCertNet MNIST Training Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Configuration

- Dataset: MNIST (`{args.data_dir}`)
- Training samples: 60,000
- Test samples: {test_samples:,}
- Device: `{device}`
- Epochs: {args.epochs}
- Batch size: {args.batch_size}
- Optimizer: Adam
- Learning rate: {args.learning_rate}
- Loss function: CrossEntropyLoss
- Elapsed time: {elapsed_seconds:.2f} seconds

## Training history

{chr(10).join(table)}

## Test evaluation

- Test loss: {test_metrics['loss']:.6f}
- Test accuracy: {test_metrics['accuracy'] * 100:.2f}%

## Checkpoint and prediction validation

- Checkpoint: `{args.checkpoint}`
- Strict state-dictionary reload into a fresh MiniCertNet: **Passed**
- Reloaded-model predictions matching the trained model: **{prediction_matches:,}/{test_samples:,}**
- Prediction test: **Passed**
- First test sample prediction: {sample_prediction}
- First test sample target: {sample_target}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = MiniCertNet()
    start = time.perf_counter()
    history = train_model(
        model,
        train_loader,
        val_loader=test_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        save_path=str(args.checkpoint),
    )
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    reloaded_model = load_model(MiniCertNet, str(args.checkpoint), device=device)
    prediction_matches = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            trained_predictions = model(inputs).argmax(dim=1)
            reloaded_predictions = reloaded_model(inputs).argmax(dim=1)
            if not torch.equal(trained_predictions, reloaded_predictions):
                raise AssertionError("Reloaded model predictions differ from trained model predictions")
            prediction_matches += inputs.size(0)

    sample_image, sample_target = test_dataset[0]
    with torch.no_grad():
        trained_logits = model(sample_image.unsqueeze(0).to(device))
        reloaded_logits = reloaded_model(sample_image.unsqueeze(0).to(device))
    torch.testing.assert_close(reloaded_logits, trained_logits, rtol=0, atol=0)
    sample_prediction = int(reloaded_logits.argmax(dim=1).item())
    elapsed_seconds = time.perf_counter() - start

    write_report(
        args.report,
        args,
        device,
        history,
        test_metrics,
        prediction_matches,
        len(test_dataset),
        sample_prediction,
        int(sample_target),
        elapsed_seconds,
    )
    print(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Checkpoint reload and {prediction_matches:,} prediction checks passed")
    print(f"Training report saved to '{args.report}'")


if __name__ == "__main__":
    main()
