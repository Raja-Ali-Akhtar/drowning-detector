"""Verify dataset health before training.

Checks pose file integrity, shape consistency, class balance, annotation coverage,
and flags potential issues. Run this before every training session.

Usage:
    python drowning_detector/scripts/verify_dataset.py
    python drowning_detector/scripts/verify_dataset.py --strict
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT
from drowning_detector.core.constants import (
    JOINT_DIMS,
    LABEL_DROWNING,
    LABEL_NAMES,
    NUM_JOINTS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
)

EXPECTED_SHAPE = (SEQUENCE_LENGTH, NUM_JOINTS, JOINT_DIMS)


class DatasetReport:
    """Collects and reports dataset health checks."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def log(self, msg: str) -> None:
        self.info.append(msg)
        logger.info(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            "DATASET VERIFICATION REPORT",
            "=" * 60,
            f"  Errors:   {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Status:   {'PASS' if self.passed else 'FAIL'}",
            "=" * 60,
        ]
        if self.errors:
            lines.append("\nERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def check_pose_files(data_root: Path, report: DatasetReport) -> dict[str, int]:
    """Verify all .npy pose files have correct shape and valid values.

    Returns:
        Dict mapping class_folder → count of valid files.
    """
    poses_root = data_root / "poses"
    class_counts: dict[str, int] = {}

    if not poses_root.exists():
        report.error(f"Poses directory not found: {poses_root}")
        return class_counts

    corrupted = 0
    bad_shape = 0
    all_zero = 0
    high_zero_ratio = 0
    total = 0

    for class_dir in sorted(poses_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        npy_files = sorted(class_dir.glob("*.npy"))
        valid = 0

        for npy_path in npy_files:
            total += 1
            try:
                data = np.load(npy_path)
            except Exception:
                report.error(f"Corrupted file: {npy_path}")
                corrupted += 1
                continue

            if data.shape != EXPECTED_SHAPE:
                report.error(
                    f"Bad shape {data.shape} (expected {EXPECTED_SHAPE}): {npy_path.name}"
                )
                bad_shape += 1
                continue

            # Check for all-zero sequences (no pose detected at all)
            if np.all(data == 0):
                report.warn(f"All-zero pose sequence: {npy_path.name}")
                all_zero += 1
                continue

            # Check ratio of zero frames (frames where pose was not detected)
            zero_frames = np.sum(np.all(data == 0, axis=(1, 2)))
            zero_ratio = zero_frames / SEQUENCE_LENGTH
            if zero_ratio > 0.3:
                report.warn(
                    f"High zero-frame ratio ({zero_ratio:.0%}): {npy_path.name}"
                )
                high_zero_ratio += 1

            # Check for NaN/Inf
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                report.error(f"NaN/Inf values in: {npy_path.name}")
                continue

            # Check coordinate range (should be 0-1 for x,y)
            xy_vals = data[:, :, :2]
            if np.any(xy_vals < -0.5) or np.any(xy_vals > 1.5):
                report.warn(f"Coordinates outside expected range in: {npy_path.name}")

            valid += 1

        class_counts[class_dir.name] = valid
        report.log(f"Class '{class_dir.name}': {valid}/{len(npy_files)} valid pose files")

    report.log(f"Total pose files scanned: {total}")
    if corrupted:
        report.error(f"{corrupted} corrupted files found")
    if bad_shape:
        report.error(f"{bad_shape} files with incorrect shape")
    if all_zero:
        report.warn(f"{all_zero} all-zero sequences (no pose detected)")
    if high_zero_ratio:
        report.warn(f"{high_zero_ratio} files with >30% zero frames")

    return class_counts


def check_class_balance(
    class_counts: dict[str, int],
    report: DatasetReport,
    min_drowning: int = 400,
) -> None:
    """Check class distribution and minimum sample counts."""
    if not class_counts:
        report.error("No valid pose files found")
        return

    report.log("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        report.log(f"  {cls}: {count} samples")

    # Check minimum drowning samples
    drowning_count = class_counts.get("drowning", 0)
    if drowning_count < min_drowning:
        report.error(
            f"Insufficient drowning samples: {drowning_count} (need >= {min_drowning})"
        )
    else:
        report.log(f"Drowning samples: {drowning_count} (meets minimum of {min_drowning})")

    # Check class imbalance ratio
    total = sum(class_counts.values())
    if total > 0 and drowning_count > 0:
        ratio = (total - drowning_count) / drowning_count
        if ratio > 10:
            report.warn(
                f"Severe class imbalance: negative/positive ratio is {ratio:.1f}:1 "
                "(consider oversampling or WeightedRandomSampler)"
            )
        elif ratio > 5:
            report.warn(f"Class imbalance: negative/positive ratio is {ratio:.1f}:1")


def check_annotations(
    data_root: Path,
    class_counts: dict[str, int],
    report: DatasetReport,
) -> None:
    """Verify annotations.csv exists and is consistent with pose files."""
    annotations_path = data_root / "annotations.csv"

    if not annotations_path.exists():
        report.warn("annotations.csv not found — run build_annotations.py first")
        return

    try:
        df = pd.read_csv(annotations_path)
    except Exception as e:
        report.error(f"Failed to read annotations.csv: {e}")
        return

    required_cols = {"pose_path", "label", "split", "confidence"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        report.error(f"Missing columns in annotations.csv: {missing_cols}")
        return

    report.log(f"Annotations: {len(df)} entries")

    # Check splits exist
    splits = df["split"].unique()
    for required_split in ["train", "val", "test"]:
        if required_split not in splits:
            report.error(f"Missing '{required_split}' split in annotations")

    # Log split distribution
    split_counts = df["split"].value_counts().to_dict()
    report.log(f"Split distribution: {split_counts}")

    # Check label distribution per split
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            continue
        label_dist = split_df["label"].value_counts().to_dict()
        report.log(f"  {split_name}: {label_dist}")

        # Check that drowning class exists in each split
        if LABEL_DROWNING not in label_dist:
            report.error(f"No drowning samples in '{split_name}' split")

    # Cross-check with pose files
    total_poses = sum(class_counts.values())
    if abs(len(df) - total_poses) > 0:
        report.warn(
            f"Annotation count ({len(df)}) differs from pose file count ({total_poses}). "
            "Re-run build_annotations.py to sync."
        )

    # Check for missing pose files referenced in annotations
    missing_files = 0
    for pose_path in df["pose_path"]:
        full_path = data_root / pose_path
        if not full_path.exists():
            missing_files += 1
    if missing_files:
        report.error(f"{missing_files} pose files referenced in annotations but not found on disk")


def verify_dataset(
    data_root: Path,
    min_drowning: int = 400,
) -> DatasetReport:
    """Run all dataset verification checks.

    Args:
        data_root: Root data directory.
        min_drowning: Minimum number of drowning samples required.

    Returns:
        DatasetReport with all findings.
    """
    report = DatasetReport()

    report.log(f"Verifying dataset at: {data_root}")
    report.log(f"Expected pose shape: {EXPECTED_SHAPE}")

    # Check pose files
    class_counts = check_pose_files(data_root, report)

    # Check class balance
    check_class_balance(class_counts, report, min_drowning)

    # Check annotations
    check_annotations(data_root, class_counts, report)

    return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify dataset health before training.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "drowning_detector" / "data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--min-drowning",
        type=int,
        default=400,
        help="Minimum drowning samples required (default: 400).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 on any warning).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    report = verify_dataset(args.data_root, args.min_drowning)
    print(report.summary())

    if not report.passed:
        sys.exit(1)
    if args.strict and report.warnings:
        logger.error("Strict mode: treating {} warnings as errors", len(report.warnings))
        sys.exit(1)

    sys.exit(0)
