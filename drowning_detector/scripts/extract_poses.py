"""Extract 14-joint pose sequences from video clips using MediaPipe.

Processes all clips in data/clips/{class}/ and saves pose arrays as .npy files
in data/poses/{class}/. Each output has shape (T, 14, 3) where 3 = (x, y, visibility).

Usage:
    python drowning_detector/scripts/extract_poses.py
    python drowning_detector/scripts/extract_poses.py --classes drowning treading
    python drowning_detector/scripts/extract_poses.py --input data/clips/drowning/ --output data/poses/drowning/
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT
from drowning_detector.core.constants import JOINT_DIMS, NUM_JOINTS, POSE_FPS, SEQUENCE_LENGTH

# ── MediaPipe landmark indices → our 14-joint skeleton ──────────────────
# MediaPipe Pose has 33 landmarks. We map to our 14-joint subset.
MP_TO_SKELETON: dict[int, int] = {
    0: 0,    # nose → nose
    11: 1,   # left shoulder → left shoulder
    12: 2,   # right shoulder → right shoulder
    13: 3,   # left elbow → left elbow
    14: 4,   # right elbow → right elbow
    15: 5,   # left wrist → left wrist
    16: 6,   # right wrist → right wrist
    23: 7,   # left hip → left hip
    24: 8,   # right hip → right hip
    25: 9,   # left knee → left knee
    26: 10,  # right knee → right knee
    27: 11,  # left ankle → left ankle
    28: 12,  # right ankle → right ankle
}

# Head centre (joint 13) is computed as midpoint of left ear (7) and right ear (8)
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_CLASSES = ["drowning", "treading", "swimming", "splashing"]


def extract_pose_from_clip(
    clip_path: Path,
    target_fps: int = POSE_FPS,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Extract a 14-joint pose sequence from a single video clip.

    Args:
        clip_path: Path to the video clip.
        target_fps: Target frame rate for sampling.
        sequence_length: Target number of frames (pad/truncate to this).

    Returns:
        np.ndarray of shape (sequence_length, 14, 3).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: {}", clip_path)
        pose.close()
        return np.zeros((sequence_length, NUM_JOINTS, JOINT_DIMS), dtype=np.float32)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0  # fallback

    # Frame sampling interval to achieve target_fps
    sample_interval = max(1, round(source_fps / target_fps))

    frames_data: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at target fps
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = np.zeros((NUM_JOINTS, JOINT_DIMS), dtype=np.float32)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Map MediaPipe landmarks to our 14-joint skeleton
            for mp_idx, our_idx in MP_TO_SKELETON.items():
                lm = landmarks[mp_idx]
                keypoints[our_idx] = [lm.x, lm.y, lm.visibility]

            # Compute head centre as midpoint of ears
            left_ear = landmarks[MP_LEFT_EAR]
            right_ear = landmarks[MP_RIGHT_EAR]
            keypoints[13] = [
                (left_ear.x + right_ear.x) / 2,
                (left_ear.y + right_ear.y) / 2,
                (left_ear.visibility + right_ear.visibility) / 2,
            ]

        frames_data.append(keypoints)
        frame_idx += 1

    cap.release()
    pose.close()

    if not frames_data:
        logger.warning("No frames extracted from: {}", clip_path)
        return np.zeros((sequence_length, NUM_JOINTS, JOINT_DIMS), dtype=np.float32)

    sequence = np.array(frames_data, dtype=np.float32)  # (T_actual, 14, 3)

    # Pad or truncate to standard sequence length
    sequence = _normalize_sequence_length(sequence, sequence_length)

    return sequence


def _normalize_sequence_length(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Pad short sequences with zeros at end, truncate long ones from start.

    Args:
        sequence: Raw pose sequence of shape (T, 14, 3).
        target_length: Desired number of frames.

    Returns:
        np.ndarray of shape (target_length, 14, 3).
    """
    current_length = len(sequence)

    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        # Truncate from the start (keep most recent frames)
        return sequence[-target_length:]
    else:
        # Pad with zeros at the end
        pad_length = target_length - current_length
        padding = np.zeros((pad_length, NUM_JOINTS, JOINT_DIMS), dtype=np.float32)
        return np.concatenate([sequence, padding], axis=0)


def process_class_directory(
    input_dir: Path,
    output_dir: Path,
    target_fps: int = POSE_FPS,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[int, int]:
    """Process all clips in a class directory and save pose .npy files.

    Args:
        input_dir: Directory containing video clips for one class.
        output_dir: Directory to save .npy pose files.
        target_fps: Target frame rate.
        sequence_length: Target sequence length.

    Returns:
        Tuple of (successful, failed) counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file()
    ])

    if not clips:
        logger.warning("No clips found in {}", input_dir)
        return 0, 0

    logger.info("Processing {} clips from {}", len(clips), input_dir)

    successful = 0
    failed = 0

    for i, clip_path in enumerate(clips):
        try:
            pose_sequence = extract_pose_from_clip(clip_path, target_fps, sequence_length)

            output_path = output_dir / f"{clip_path.stem}.npy"
            np.save(output_path, pose_sequence)

            # Validate shape
            assert pose_sequence.shape == (sequence_length, NUM_JOINTS, JOINT_DIMS), (
                f"Unexpected shape: {pose_sequence.shape}"
            )

            successful += 1

            if (i + 1) % 50 == 0:
                logger.info("Progress: {}/{} clips processed", i + 1, len(clips))

        except Exception as e:
            logger.error("Failed to process {}: {}", clip_path.name, e)
            failed += 1

    logger.info(
        "Class '{}': {} successful, {} failed out of {} total",
        input_dir.name,
        successful,
        failed,
        len(clips),
    )
    return successful, failed


def extract_all_poses(
    data_root: Path,
    classes: list[str] | None = None,
    target_fps: int = POSE_FPS,
    sequence_length: int = SEQUENCE_LENGTH,
) -> None:
    """Extract poses for all classes.

    Args:
        data_root: Root data directory containing clips/ and poses/ subdirs.
        classes: List of class names to process. None = all classes.
        target_fps: Target frame rate.
        sequence_length: Target sequence length.
    """
    classes = classes or ALL_CLASSES
    clips_root = data_root / "clips"
    poses_root = data_root / "poses"

    total_success = 0
    total_failed = 0

    for class_name in classes:
        input_dir = clips_root / class_name
        output_dir = poses_root / class_name

        if not input_dir.exists():
            logger.warning("Clips directory not found: {}", input_dir)
            continue

        success, failed = process_class_directory(
            input_dir, output_dir, target_fps, sequence_length
        )
        total_success += success
        total_failed += failed

    logger.info(
        "Extraction complete: {} successful, {} failed across {} classes",
        total_success,
        total_failed,
        len(classes),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract 14-joint pose sequences from video clips using MediaPipe.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input directory of clips. Overrides --classes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for .npy files. Required if --input is set.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=f"Class names to process (default: all). Options: {ALL_CLASSES}",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=POSE_FPS,
        help=f"Target frame rate (default: {POSE_FPS}).",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"Target sequence length in frames (default: {SEQUENCE_LENGTH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input:
        # Single directory mode
        if not args.output:
            logger.error("--output is required when using --input")
            sys.exit(1)
        if not args.input.exists():
            logger.error("Input directory does not exist: {}", args.input)
            sys.exit(1)

        process_class_directory(args.input, args.output, args.fps, args.seq_length)
    else:
        # Batch mode: process all classes
        data_root = PROJECT_ROOT / "drowning_detector" / "data"
        extract_all_poses(data_root, args.classes, args.fps, args.seq_length)
