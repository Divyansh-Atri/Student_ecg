"""Build a segment labeling manifest from processed ECG outputs.

Scans Processed_ECG metadata and segment CSVs to produce ml/segment_manifest.csv
for manual quality and arrhythmia labels before model training.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

SEGMENT_RE = re.compile(r"segment_(\d+)\.csv$", re.IGNORECASE)


def build_manifest(processed_ecg_dir: Path, output_path: Path) -> pd.DataFrame:
    """Collect segment paths and metadata into a labeling manifest CSV.

    Args:
        processed_ecg_dir: Root folder (e.g. Processed_ECG) with group/recording trees.
        output_path: Destination CSV path for the manifest.

    Returns:
        DataFrame written to output_path with one row per segment file.
    """
    processed_ecg_dir = Path(processed_ecg_dir)
    output_path = Path(output_path)
    rows: list[dict] = []

    for meta_path in sorted(processed_ecg_dir.glob("**/clean/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        segments_dir = meta_path.parent / "segments"
        if not segments_dir.is_dir():
            continue

        group_name = str(meta.get("group_name", meta_path.parent.parent.parent.name))
        recording_ts = meta.get("recording_timestamp_ms", 0)
        quality = meta.get("quality", {}) if isinstance(meta.get("quality"), dict) else {}
        quality_label = str(quality.get("label", ""))
        quality_score = float(quality.get("score_0_100", float("nan")))
        hr = float(meta.get("rpeaks", {}).get("heart_rate_bpm", float("nan")))

        for seg_path in sorted(segments_dir.glob("segment_*.csv")):
            m = SEGMENT_RE.search(seg_path.name)
            seg_index = int(m.group(1)) - 1 if m else 0
            rows.append(
                {
                    "segment_path": str(seg_path.resolve()),
                    "group_name": group_name,
                    "recording_timestamp_ms": recording_ts,
                    "quality_label": quality_label,
                    "quality_score": quality_score,
                    "heart_rate_bpm": hr,
                    "segment_index": seg_index,
                    "manual_quality_label": "",
                    "arrhythmia_label": "",
                }
            )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """CLI entry: build ml/segment_manifest.csv from Processed_ECG."""
    root = Path(__file__).resolve().parents[1]
    processed = root / "Processed_ECG"
    out = root / "ml" / "segment_manifest.csv"
    df = build_manifest(processed, out)
    print(f"Found {len(df)} segments. Manifest saved to {out}")


if __name__ == "__main__":
    main()
