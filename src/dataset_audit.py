
from pathlib import Path
from collections import Counter
import csv
import mne

DATA_DIR = Path("BCICIV_2a_gdf")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}"
    )

files = sorted(DATA_DIR.glob("*.gdf"))

if not files:
    raise FileNotFoundError("No GDF files found.")

summary = []

for path in files:
    print(f"\nAuditing {path.name}...")

    raw = mne.io.read_raw_gdf(
        path,
        preload=False,
        verbose="ERROR"
    )

    annotations = Counter(raw.annotations.description)

    row = {
        "file": path.name,
        "channels": len(raw.ch_names),
        "sampling_rate_hz": raw.info["sfreq"],
        "duration_seconds": round(raw.times[-1], 2),
        "annotation_counts": dict(annotations),
    }
    summary.append(row)

    print("Channels:", row["channels"])
    print("Sampling rate:", row["sampling_rate_hz"], "Hz")
    print("Duration:", row["duration_seconds"], "seconds")
    print("Annotation counts:", dict(annotations))

    raw.close()

report_path = OUTPUT_DIR / "dataset_audit.csv"

with report_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "channels",
            "sampling_rate_hz",
            "duration_seconds",
            "annotation_counts",
        ],
    )
    writer.writeheader()

    for row in summary:
        writer.writerow({
            **row,
            "annotation_counts": str(row["annotation_counts"]),
        })

print(f"\nAudited {len(summary)} GDF files.")
print(f"Report saved to: {report_path}")
