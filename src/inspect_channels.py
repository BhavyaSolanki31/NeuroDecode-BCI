
from pathlib import Path
from collections import Counter
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "BCICIV_2a_gdf"

files = sorted(DATA_DIR.glob("*T.gdf"))

if not files:
    raise FileNotFoundError(f"No training GDF files found in: {DATA_DIR}")

path = files[0]

print("=" * 60)
print("EEG CHANNEL INSPECTION")
print("=" * 60)
print(f"Recording: {path.name}")

raw = mne.io.read_raw_gdf(
    path,
    preload=False,
    verbose="ERROR"
)

try:
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Total channels before correction: {len(raw.ch_names)}")

    print("\nOriginal channel names and types:")
    for name, channel_type in zip(
        raw.ch_names, raw.get_channel_types()
    ):
        print(f"{name:<20} {channel_type}")

    eog_channels = [
        name for name in raw.ch_names
        if name.upper().startswith("EOG-")
    ]

    if len(eog_channels) != 3:
        raise ValueError(
            f"Expected 3 EOG channels, found {len(eog_channels)}: "
            f"{eog_channels}"
        )

    raw.set_channel_types(
        {name: "eog" for name in eog_channels},
        verbose="ERROR"
    )

    channel_types = raw.get_channel_types()
    type_counts = Counter(channel_types)

    eeg_channels = [
        name for name, kind in zip(raw.ch_names, channel_types)
        if kind == "eeg"
    ]

    print("\nCorrected channel type summary:")
    for kind, count in type_counts.items():
        print(f"{kind}: {count}")

    print(f"\nEEG channel count: {len(eeg_channels)}")
    print("EEG channels:")
    print(", ".join(eeg_channels))

    print(f"\nEOG channel count: {len(eog_channels)}")
    print("EOG channels:")
    print(", ".join(eog_channels))

    assert len(eeg_channels) == 22, (
        f"Expected 22 EEG channels, found {len(eeg_channels)}"
    )

    assert len(eog_channels) == 3, (
        f"Expected 3 EOG channels, found {len(eog_channels)}"
    )

    print("\nChannel verification passed.")

finally:
    raw.close()
