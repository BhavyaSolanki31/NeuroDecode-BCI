from pathlib import Path
from collections import Counter
import mne

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "BCICIV_2a_gdf"

CLASS_CODES = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

files = sorted(DATA_DIR.glob("*T.gdf"))

if not files:
    raise FileNotFoundError(f"No training GDF files found in: {DATA_DIR}")

print(f"Dataset folder: {DATA_DIR}")
print(f"Training files found: {len(files)}")

for path in files:
    print(f"\n{path.name}")

    raw = mne.io.read_raw_gdf(
        path,
        preload=False,
        verbose="ERROR"
    )

    try:
        counts = Counter(raw.annotations.description)

        for code, label in CLASS_CODES.items():
            print(f"  {label}: {counts.get(code, 0)}")
    finally:
        raw.close()

print("\nLabel verification completed")

