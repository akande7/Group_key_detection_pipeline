"""
Combine multiple independently-captured normal-traffic feature CSVs
(each produced by extract_features.py from a separate capture session)
into a single normal dataset for use as prepare_data.py's NORMAL_FILE.

Each input file is windowed independently (window_number starts at 0
in each), so window_number is re-numbered sequentially across the
combined dataset to remain unique. A source_session column is added
so the originating session can still be identified later if needed
(e.g. for methodology write-up or consistency checks) -- this column
is not intended to be fed into the model as a feature.
"""

from pathlib import Path

import pandas as pd

# List the per-session feature CSVs to combine, in any order.
INPUT_FILES = [
    Path("normal_features.csv"),
    Path("second_normal_features.csv"),
    Path("third_normal_features.csv"),
]

OUTPUT_FILE = Path("combined_normal_features.csv")


def combine_sessions(input_files: list[Path], output_file: Path) -> pd.DataFrame:
    frames = []

    for session_number, path in enumerate(input_files, start=1):
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(f"Session file contains no observations: {path}")

        dataframe["source_session"] = session_number
        frames.append(dataframe)

    combined = pd.concat(frames, ignore_index=True)

    # Re-number windows sequentially so they are unique across all sessions.
    combined["window_number"] = combined.index

    combined.to_csv(output_file, index=False)

    return combined


def main() -> None:
    combined = combine_sessions(INPUT_FILES, OUTPUT_FILE)

    print("\nSession combination complete")
    print("-" * 50)
    print(f"Sessions combined: {len(INPUT_FILES)}")
    for path in INPUT_FILES:
        print(f"  - {path}")
    print(f"Total windows: {len(combined)}")
    print(f"Windows per session:")
    print(combined["source_session"].value_counts().sort_index().to_string())
    print(
        "cross_layer_inconsistency_count total: "
        f"{combined['cross_layer_inconsistency_count'].sum()}"
    )
    print(f"Output CSV: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
