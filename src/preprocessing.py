
from pathlib import Path
import re

import mne
import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "BCICIV_2a_gdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Motor-imagery class labels used by the ML models
CLASS_CODES = {
    "769": 0,  # Left hand
    "770": 1,  # Right hand
    "771": 2,  # Feet
    "772": 3,  # Tongue
}

CLASS_NAMES = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

# Non-zero event IDs for MNE
MNE_EVENT_ID = {
    "769": 1,  # Left hand
    "770": 2,  # Right hand
    "771": 3,  # Feet
    "772": 4,  # Tongue
}


def preprocess_recording(path):
    """Filter a training recording and extract labelled EEG epochs."""

    raw = mne.io.read_raw_gdf(
        path,
        preload=True,
        verbose="ERROR",
    )

    try:
        # Correct EOG channel types
        eog_channels = [
            name
            for name in raw.ch_names
            if name.upper().startswith("EOG-")
        ]

        if len(eog_channels) != 3:
            raise ValueError(
                f"Expected 3 EOG channels, found {len(eog_channels)}"
            )

        raw.set_channel_types(
            {name: "eog" for name in eog_channels},
            verbose="ERROR",
        )

        # Keep only the 22 EEG channels
        raw.pick(picks="eeg")

        if len(raw.ch_names) != 22:
            raise ValueError(
                f"Expected 22 EEG channels, found {len(raw.ch_names)}"
            )

        # Save the actual selected EEG channel names
        eeg_channels = list(raw.ch_names)

        # Band-pass filter: 8–30 Hz
        raw.filter(
            l_freq=8.0,
            h_freq=30.0,
            verbose="ERROR",
        )

        # Extract motor-imagery events using non-zero MNE IDs
        events, _ = mne.events_from_annotations(
            raw,
            event_id=MNE_EVENT_ID,
            verbose="ERROR",
        )

        if len(events) == 0:
            raise ValueError("No motor-imagery events found.")

        # Extract 4-second epochs from cue onset
        epochs = mne.Epochs(
            raw,
            events,
            event_id=MNE_EVENT_ID,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            event_repeated="error",
            verbose="ERROR",
        )

        # EEG data: trials x channels x time samples
        X = epochs.get_data().astype(np.float32)

        # Convert MNE IDs 1–4 to ML labels 0–3
        y = (epochs.events[:, -1] - 1).astype(np.int64)

        if len(X) != len(y):
            raise ValueError("EEG trial and label counts do not match.")

        if len(y) == 0:
            raise ValueError("No valid epochs remain after preprocessing.")

        if not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError(f"Unexpected class labels: {np.unique(y)}")

        return X, y, eeg_channels

    finally:
        raw.close()


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    # T recordings contain the labelled training trials
    files = sorted(DATA_DIR.glob("*T.gdf"))

    if len(files) != 9:
        raise ValueError(
            f"Expected 9 training recordings, found {len(files)}"
        )

    for path in files:
        print(f"\nProcessing {path.name}...")

        X, y, eeg_channels = preprocess_recording(path)

        # Extract participant ID, e.g. A01 from A01T.gdf
        match = re.fullmatch(r"(A\d{2})T\.gdf", path.name)

        if match is None:
            raise ValueError(f"Unexpected filename: {path.name}")

        participant = match.group(1)
        output_path = OUTPUT_DIR / f"{participant}_epochs.npz"

        class_counts = np.bincount(y, minlength=4)

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            class_names=np.array(CLASS_NAMES),
            participant=participant,
            sampling_rate=np.array(250.0),
            eeg_channels=np.array(eeg_channels),
        )

        print(f"  Epoch shape: {X.shape}")
        print(f"  Trials per class: {class_counts}")
        print(f"  Saved: {output_path}")

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
