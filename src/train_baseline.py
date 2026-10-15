from pathlib import Path
import json

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]

# Keep entire participants separate to evaluate generalization to unseen people.
TRAIN_SUBJECTS = [f"A{i:02d}" for i in range(1, 8)]
VALIDATION_SUBJECT = "A08"
TEST_SUBJECT = "A09"


def load_subject(subject):
    file_path = DATA_DIR / f"{subject}_epochs.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float64)
        y = data["y"].astype(np.int64)

    if X.ndim != 3 or len(X) != len(y):
        raise ValueError(f"Unexpected data dimensions for {subject}")

    if not np.array_equal(np.unique(y), np.arange(4)):
        raise ValueError(f"Unexpected class labels for {subject}: {np.unique(y)}")

    return X, y


def main():
    train_data = [load_subject(subject) for subject in TRAIN_SUBJECTS]
    X_train = np.concatenate([item[0] for item in train_data], axis=0)
    y_train = np.concatenate([item[1] for item in train_data], axis=0)

    X_val, y_val = load_subject(VALIDATION_SUBJECT)
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

    print("Training CSP + LDA baseline...")
    print(f"Training participants: {', '.join(TRAIN_SUBJECTS)}")
    print(f"Training trials: {len(y_train)}")
    model.fit(X_train, y_train)

    print("\n--- Validation: A08 ---")
    val_predictions = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_val, val_predictions):.4f}"
    )
    print(f"Macro F1: {f1_score(y_val, val_predictions, average='macro'):.4f}")

    # A09 is reserved for final testing and is not used to fit the model.
    print("\n--- Held-out test participant: A09 ---")
    test_predictions = model.predict(X_test)

    metrics = {
        "model": "CSP + LDA",
        "training_subjects": TRAIN_SUBJECTS,
        "validation_subject": VALIDATION_SUBJECT,
        "test_subject": TEST_SUBJECT,
        "test_trials": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, test_predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, test_predictions, average="macro")
        ),
        "class_names": CLASS_NAMES,
        "classification_report": classification_report(
            y_test,
            test_predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=[0, 1, 2, 3]
        ).tolist(),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    output_file = RESULTS_DIR / "csp_lda_baseline.json"
    output_file.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"Metrics saved to: {output_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()