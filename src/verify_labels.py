
from pathlib import Path
from collections import Counter
import mne

DATA_DIR = Path("BCICIV_2a_gdf")

CLASS_CODES = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}

for path in sorted(DATA_DIR.glob("*T.gdf")):
    print(f"\n{path.name}")

    raw = mne.io.read_raw_gdf(
        path,
        preload=False,
        verbose="ERROR"
    )

    counts = Counter(raw.annotations.description)

    for code, label in CLASS_CODES.items():
        print(f"  {label}: {counts.get(code, 0)}")

    raw.close()

print("\nLabel verification completed.")
