"""Train a lightweight herb image classifier for PharmaGuide.

Images are read from ``data/herbs/<herb_name>/*.jpg`` (other common image
extensions are accepted by torchvision).  The script deliberately keeps the
training path CPU-friendly and exits cleanly when a photo dataset is not yet
available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "herbs"
MODELS_DIR = ROOT / "models"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8


def _missing_dataset() -> int:
    print("add photos to data/herbs/<herb_name>/ then rerun")
    return 0


def _build_model(torch, torchvision, num_classes: int):
    """Build MobileNetV3-small, using pretrained weights when available."""
    weights = None
    try:
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    except (AttributeError, Exception):
        weights = None

    try:
        model = torchvision.models.mobilenet_v3_small(weights=weights)
    except Exception:
        # A cached weight may be unavailable (or downloading may be disabled).
        model = torchvision.models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def _metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Tuple[float, Dict[str, dict], List[List[int]]]:
    matrix = [[0 for _ in class_names] for _ in class_names]
    for actual, predicted in zip(y_true, y_pred):
        if 0 <= actual < len(class_names) and 0 <= predicted < len(class_names):
            matrix[actual][predicted] += 1

    total = len(y_true)
    accuracy = (sum(a == p for a, p in zip(y_true, y_pred)) / total) if total else 0.0
    report: Dict[str, dict] = {}
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_total = sum(row[index] for row in matrix)
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        report[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1-score": round(f1, 6),
            "support": support,
        }
    return accuracy, report, matrix


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, models, transforms
    except Exception as exc:
        print(f"torch/torchvision are required for training: {exc}")
        return 1

    if not DATA_DIR.exists():
        return _missing_dataset()

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    try:
        dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    except (FileNotFoundError, RuntimeError, ValueError):
        return _missing_dataset()

    if len(dataset) == 0 or len(dataset.classes) < 2:
        return _missing_dataset()

    class_names = list(dataset.classes)
    label_map = {str(index): name for index, name in enumerate(class_names)}
    val_size = max(1, int(round(len(dataset) * 0.2)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        return _missing_dataset()

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = _build_model(torch, models, len(class_names))
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        average_loss = running_loss / max(1, len(train_loader.dataset))
        print(f"epoch {epoch + 1}/{EPOCHS} loss={average_loss:.4f}")

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    accuracy, report, confusion = _metrics(y_true, y_pred, class_names)
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": "mobilenet_v3_small",
        "num_classes": len(class_names),
        "image_size": IMAGE_SIZE,
        "label_map": label_map,
    }
    torch.save(checkpoint, MODELS_DIR / "herb_classifier.pth")
    (MODELS_DIR / "herb_label_map.json").write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "accuracy": round(accuracy, 6),
        "per_class": report,
        "confusion_matrix": confusion,
        "classes": class_names,
    }
    (MODELS_DIR / "herb_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"accuracy: {accuracy:.4f}")
    print("per-class F1:")
    for name in class_names:
        print(f"  {name}: {report[name]['f1-score']:.4f}")
    print("confusion matrix:")
    for row in confusion:
        print(" ".join(str(value) for value in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
