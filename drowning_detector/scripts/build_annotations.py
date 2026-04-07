"""Build and update annotations.csv from the pose dataset directory structure.

Scans data/poses/{class}/ directories, assigns labels based on folder name,
performs train/val/test splits, and writes annotations.csv.

Usage:
    python drowning_detector/scripts/build_annotations.py
    python drowning_detector/scripts/build_annotations.py --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT
from drowning_detector.core.constants import (
    JOINT_DIMS,
    LABEL_DROWNING,
    LABEL_NAMES,
    LABEL_NORMAL,
    LABEL_TREADING,
    NUM_JOINTS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
)

# Class folder → label mapping
CLASS_LABEL_MAP: dict[str, int] = {
    "drowning": LABEL_DROWNING,
    "swimming": LABEL_NORMAL,
    "splashing": LABEL_NORMAL,
    "treading": LABEL_TREADING,
}


def compute_pose_confidence(npy_path: Path) -> float:
    """Compute average keypoint visibility as a confidence score.

    Args:
        npy_path: Path to a .npy pose file of shape (T, 14, 3).

    Returns:
        Mean visibility score [0.0, 1.0].
    """
    try:
        data = np.load(npy_path)
        if data.shape != (SEQUENCE_LENGTH, NUM_JOINTS, JOINT_DIMS):
            logger.warning("Unexpected shape {} in {}", data.shape, npy_path.name)
            return 0.0
        # Visibility is the 3rd channel
        visibility = data[:, :, 2]
        return float(np.mean(visibility))
    except Exception as e:
        logger.error("Failed to read {}: {}", npy_path, e)
        return 0.0


def scan_poses_directory(data_root: Path) -> pd.DataFrame:
    """Scan all pose directories and build a records list.

    Args:
        data_root: Root data directory containing poses/ subdirectory.

    Returns:
        DataFrame with columns: clip_path, pose_path, label, label_name, class_folder, confidence.
    """
    poses_root = data_root / "poses"
    clips_root = data_root / "clips"

    records: list[dict] = []

    for class_folder, label in CLASS_LABEL_MAP.items():
        pose_dir = poses_root / class_folder
        clip_dir = clips_root / class_folder

        if not pose_dir.exists():
            logger.warning("Pose directory not found: {}", pose_dir)
            continue

        npy_files = sorted(pose_dir.glob("*.npy"))
        logger.info("Found {} pose files in {}", len(npy_files), pose_dir)

        for npy_path in npy_files:
            # Try to find matching clip
            clip_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                candidate = clip_dir / f"{npy_path.stem}{ext}"
                if candidate.exists():
                    clip_path = str(candidate.relative_to(data_root))
                    break

            confidence = compute_pose_confidence(npy_path)
            pose_rel_path = str(npy_path.relative_to(data_root))

            records.append({
                "pose_path": pose_rel_path,
                "clip_path": clip_path or "",
                "label": label,
                "label_name": LABEL_NAMES[label],
                "class_folder": class_folder,
                "confidence": round(confidence, 4),
            })

    df = pd.DataFrame(records)
    logger.info("Total samples scanned: {}", len(df))
    return df


def assign_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign train/val/test splits with stratification by label.

    Args:
        df: DataFrame with at least a 'label' column.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added 'split' column.
    """
    rng = np.random.RandomState(seed)

    splits = pd.Series("train", index=df.index)

    for label in df["label"].unique():
        label_mask = df["label"] == label
        label_indices = df[label_mask].index.to_numpy()
        rng.shuffle(label_indices)

        n = len(label_indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_indices = label_indices[:n_test]
        val_indices = label_indices[n_test:n_test + n_val]

        splits.iloc[test_indices] = "test"
        splits.iloc[val_indices] = "val"

    df = df.copy()
    df["split"] = splits
    return df


def build_annotations(
    data_root: Path,
    output_path: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Build the full annotations CSV.

    Args:
        data_root: Root data directory.
        output_path: Path to write annotations.csv.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        min_confidence: Minimum confidence to include sample.

    Returns:
        Final annotations DataFrame.
    """
    df = scan_poses_directory(data_root)

    if df.empty:
        logger.error("No pose files found. Run extract_poses.py first.")
        return df

    # Filter low-confidence poses
    before_count = len(df)
    df = df[df["confidence"] >= min_confidence].reset_index(drop=True)
    if before_count > len(df):
        logger.info(
            "Filtered {} low-confidence samples (threshold: {})",
            before_count - len(df),
            min_confidence,
        )

    # Assign splits
    df = assign_splits(df, val_ratio, test_ratio)

    # Log class distribution
    logger.info("Class distribution:")
    for label_name, group in df.groupby("label_name"):
        split_counts = group["split"].value_counts().to_dict()
        logger.info("  {}: {} total | {}", label_name, len(group), split_counts)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Annotations saved to {} ({} samples)", output_path, len(df))

    return df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build annotations.csv from pose dataset directory structure.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "drowning_detector" / "data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for annotations.csv (default: data_root/annotations.csv).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum pose confidence to include (default: 0.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or args.data_root / "annotations.csv"

    build_annotations(
        data_root=args.data_root,
        output_path=output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_confidence=args.min_confidence,
    )
