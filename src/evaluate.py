from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
TRAIN_SUBJECTS = [f"A{i:02d}" for i in range(1, 8)]
TEST_SUBJECT = "A09"


def load_subject(subject):
    """Load preprocessed EEG epochs for one participant."""
    file_path = DATA_DIR / f"{subject}_epochs.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing preprocessed data: {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)

    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(f"Unexpected data dimensions for {subject}")

    if not np.array_equal(np.unique(y), np.arange(4)):
        raise ValueError(f"Unexpected class labels for {subject}: {np.unique(y)}")

    return X, y


def main():
    # Load training participants.
    training_data = [load_subject(s) for s in TRAIN_SUBJECTS]
    X_train = np.concatenate([item[0] for item in training_data], axis=0)
    y_train = np.concatenate([item[1] for item in training_data], axis=0)

    # Keep A09 completely separate from model fitting.
    X_test, y_test = load_subject(TEST_SUBJECT)

    model = Pipeline([
        (
            "csp",
            CSP(
                n_components=4,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            ),
        ),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    print("Training CSP + LDA...")
    model.fit(X_train, y_train)

    print(f"Evaluating on held-out participant {TEST_SUBJECT}...")
    predictions = model.predict(X_test)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "CSP + LDA",
        "training_subjects": TRAIN_SUBJECTS,
        "test_subject": TEST_SUBJECT,
        "test_trials": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = RESULTS_DIR / "csp_lda_evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    # Save a visual confusion matrix.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(values_format="d", xticks_rotation=35)
    plt.title(f"CSP + LDA — Held-out participant {TEST_SUBJECT}")
    plt.tight_layout()

    figure_path = RESULTS_DIR / "csp_lda_confusion_matrix.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nEvaluation results")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"Metrics saved: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Confusion matrix saved: {figure_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
