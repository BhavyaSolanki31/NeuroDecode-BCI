
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================================================
# Configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 12

CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]

TRAIN_SUBJECTS = [f"A{i:02d}" for i in range(1, 8)]
VAL_SUBJECT = "A08"
TEST_SUBJECT = "A09"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# Data loading
# ==================================================

def load_subject(subject):
    path = DATA_DIR / f"{subject}_epochs.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed data: {path}"
        )

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

    if X.ndim != 3:
        raise ValueError(
            f"{subject}: expected trials x channels x time, "
            f"got {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{subject}: EEG data and labels do not match"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{subject}: EEG data contains NaN or infinity"
        )

    if not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError(
            f"{subject}: unexpected labels {np.unique(y)}"
        )

    return X, y


def load_subjects(subjects):
    all_X = []
    all_y = []

    for subject in subjects:
        X, y = load_subject(subject)
        all_X.append(X)
        all_y.append(y)

        print(
            f"Loaded {subject}: X={X.shape}, y={y.shape}"
        )

    return np.concatenate(all_X), np.concatenate(all_y)


# ==================================================
# EEGNet-style CNN
# Input: batch x 1 x channels x time
# ==================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4):
        super().__init__()

        filters = 8
        depth_multiplier = 2
        depth_filters = filters * depth_multiplier

        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                filters,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filters),
        )

        # Spatial filtering across EEG channels
        self.spatial = nn.Sequential(
            nn.Conv2d(
                filters,
                depth_filters,
                kernel_size=(n_channels, 1),
                groups=filters,
                bias=False,
            ),
            nn.BatchNorm2d(depth_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )

        # Separable temporal convolution
        self.separable = nn.Sequential(
            nn.Conv2d(
                depth_filters,
                depth_filters,
                kernel_size=(1, 16),
                padding="same",
                groups=depth_filters,
                bias=False,
            ),
            nn.Conv2d(
                depth_filters,
                16,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )

        # Determine classifier input size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, 1001)
            features = self._features(dummy)
            flattened_size = features.flatten(
                start_dim=1
            ).shape[1]

        self.classifier = nn.Linear(
            flattened_size, n_classes
        )

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# ==================================================
# Training and validation utilities
# ==================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)
        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return average_loss, accuracy


# ==================================================
# Main experiment
# ==================================================

def main():
    set_seed()

    print(f"\nDevice: {DEVICE}")
    print("Loading train, validation, and test participants...")

    X_train, y_train = load_subjects(TRAIN_SUBJECTS)
    X_val, y_val = load_subject(VAL_SUBJECT)
    X_test, y_test = load_subject(TEST_SUBJECT)

    # Convert volts to microvolts
    X_train = X_train * 1e6
    X_val = X_val * 1e6
    X_test = X_test * 1e6

    # --------------------------------------------------
    # Channel normalization using training data only
    # --------------------------------------------------

    channel_mean = X_train.mean(
        axis=(0, 2), keepdims=True
    )
    channel_std = X_train.std(
        axis=(0, 2), keepdims=True
    )

    channel_std = np.maximum(channel_std, 1e-6)

    # Apply training-derived statistics to every split
    X_train = (X_train - channel_mean) / channel_std
    X_val = (X_val - channel_mean) / channel_std
    X_test = (X_test - channel_mean) / channel_std

    # Expected input: trials x 1 x channels x time
    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]
    X_test = X_test[:, None, :, :]

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(X_train.copy()),
        torch.from_numpy(y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val.copy()),
        torch.from_numpy(y_val),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test.copy()),
        torch.from_numpy(y_test),
    )

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------
    # Model and optimizer
    # --------------------------------------------------

    model = EEGNet(
        n_channels=X_train.shape[2],
        n_classes=len(CLASS_NAMES),
    ).to(DEVICE)

    # Keep the original loss for a reproducible baseline.
    # All four training classes have equal counts.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining trials: {len(train_dataset)}")
    print(f"Validation trials: {len(val_dataset)}")
    print(f"Test trials: {len(test_dataset)}")
    print("\nStarting EEGNet training...\n")

    # --------------------------------------------------
    # Training loop with early stopping
    # --------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
        })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a model checkpoint."
        )

    # Restore the best validation checkpoint
    model.load_state_dict(best_state)

    # --------------------------------------------------
    # Save model and training history
    # --------------------------------------------------

    model_path = RESULTS_DIR / "eegnet_best.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_channels": X_train.shape[2],
            "n_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "channel_mean": channel_mean.tolist(),
            "channel_std": channel_std.tolist(),
        },
        model_path,
    )

    history_path = RESULTS_DIR / "eegnet_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # --------------------------------------------------
    # Learning curves
    # --------------------------------------------------

    epochs_run = [item["epoch"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs_run,
        [item["train_loss"] for item in history],
        label="Train",
    )
    axes[0].plot(
        epochs_run,
        [item["val_loss"] for item in history],
        label="Validation",
    )
    axes[0].set_title("EEGNet Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        epochs_run,
        [item["train_accuracy"] for item in history],
        label="Train",
    )
    axes[1].plot(
        epochs_run,
        [item["val_accuracy"] for item in history],
        label="Validation",
    )
    axes[1].set_title("EEGNet Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()

    curve_path = RESULTS_DIR / "eegnet_learning_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    # --------------------------------------------------
    # Final held-out test evaluation
    # --------------------------------------------------

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            logits = model(inputs.to(DEVICE))

            all_predictions.extend(
                logits.argmax(dim=1).cpu().numpy()
            )
            all_labels.extend(labels.numpy())

    test_accuracy = accuracy_score(
        all_labels, all_predictions
    )

    test_balanced_accuracy = balanced_accuracy_score(
        all_labels, all_predictions
    )

    test_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        f"Balanced accuracy: {test_balanced_accuracy:.4f}"
    )
    print(f"Macro F1: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print(
        "Confusion matrix "
        "(actual rows, predicted columns):"
    )
    print(matrix)

    # --------------------------------------------------
    # Save confusion matrix plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Actual class",
        title="EEGNet Confusion Matrix — A09",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    matrix_path = (
        RESULTS_DIR / "eegnet_confusion_matrix.png"
    )
    fig.savefig(matrix_path, dpi=150)
    plt.close(fig)

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {
        "model": (
            "EEGNet-style CNN with "
            "training-only channel normalization"
        ),
        "train_subjects": TRAIN_SUBJECTS,
        "validation_subject": VAL_SUBJECT,
        "test_subject": TEST_SUBJECT,
        "best_epoch": best_epoch,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(
            test_balanced_accuracy
        ),
        "test_macro_f1": float(test_macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = RESULTS_DIR / "eegnet_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  History: {history_path}")
    print(f"  Learning curves: {curve_path}")
    print(f"  Confusion matrix: {matrix_path}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
