from __future__ import annotations

import argparse
from pathlib import Path

from trialmatch.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the required TrialGPT dataset split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for split JSONL files and the patient manifest.",
    )
    args = parser.parse_args()
    manifest = prepare_dataset(args.output_dir)
    print(f"Prepared dataset in {args.output_dir}")
    print(f"Patient counts: {manifest['patient_counts']}")
    print(f"Annotation counts: {manifest['annotation_counts']}")


if __name__ == "__main__":
    main()
