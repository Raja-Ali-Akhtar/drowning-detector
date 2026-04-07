"""Batch clip raw videos into fixed-length segments using ffmpeg.

Splits raw video files into 5-second clips at 10 FPS for pose extraction.
Supports recursive directory scanning and parallel processing.

Usage:
    python drowning_detector/scripts/clip_videos.py \
        --input data/raw_video/positive/ \
        --output data/clips/drowning/ \
        --duration 5 \
        --fps 10
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from drowning_detector.core.constants import POSE_FPS

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails to read the file.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to get duration for {video_path}: {e}")


def clip_single_video(
    video_path: Path,
    output_dir: Path,
    duration: float = 5.0,
    fps: int = POSE_FPS,
    overlap: float = 0.0,
) -> list[Path]:
    """Split a single video into fixed-length clips.

    Args:
        video_path: Path to the source video.
        output_dir: Directory to write clips to.
        duration: Clip length in seconds.
        fps: Output frame rate.
        overlap: Overlap between consecutive clips in seconds.

    Returns:
        List of paths to generated clip files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    try:
        total_duration = get_video_duration(video_path)
    except RuntimeError:
        logger.warning("Skipping unreadable video: {}", video_path)
        return []

    step = duration - overlap
    clips: list[Path] = []
    clip_idx = 0
    start = 0.0

    while start + duration <= total_duration + 0.1:  # small tolerance for last clip
        output_path = output_dir / f"{stem}_clip{clip_idx:04d}.mp4"

        cmd = [
            "ffmpeg",
            "-y",                          # overwrite
            "-ss", f"{start:.2f}",         # seek to start
            "-i", str(video_path),         # input
            "-t", f"{duration:.2f}",       # clip duration
            "-r", str(fps),                # output fps
            "-c:v", "libx264",            # re-encode for consistent format
            "-preset", "fast",
            "-crf", "23",
            "-an",                         # strip audio
            "-loglevel", "error",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            clips.append(output_path)
            logger.debug("Created clip: {}", output_path.name)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed for {} at {:.1f}s: {}", video_path.name, start, e.stderr)

        clip_idx += 1
        start += step

    return clips


def find_videos(input_dir: Path) -> list[Path]:
    """Recursively find all video files in a directory.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of video file paths.
    """
    videos = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file()
    ]
    return sorted(videos)


def clip_videos_batch(
    input_dir: Path,
    output_dir: Path,
    duration: float = 5.0,
    fps: int = POSE_FPS,
    overlap: float = 0.0,
    max_workers: int = 4,
) -> int:
    """Batch clip all videos in a directory.

    Args:
        input_dir: Directory containing source videos.
        output_dir: Directory to write clips to.
        duration: Clip length in seconds.
        fps: Output frame rate.
        overlap: Overlap between clips in seconds.
        max_workers: Number of parallel workers.

    Returns:
        Total number of clips created.
    """
    videos = find_videos(input_dir)
    if not videos:
        logger.warning("No video files found in {}", input_dir)
        return 0

    logger.info("Found {} videos in {}", len(videos), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(clip_single_video, v, output_dir, duration, fps, overlap): v
            for v in videos
        }

        for future in as_completed(futures):
            video = futures[future]
            try:
                clips = future.result()
                total_clips += len(clips)
                logger.info(
                    "Clipped {}: {} clips generated",
                    video.name,
                    len(clips),
                )
            except Exception as e:
                logger.error("Failed to process {}: {}", video.name, e)

    logger.info(
        "Batch complete: {} total clips from {} videos",
        total_clips,
        len(videos),
    )
    return total_clips


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch clip raw videos into fixed-length segments for pose extraction.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing raw video files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for generated clips.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Clip duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=POSE_FPS,
        help=f"Output frame rate (default: {POSE_FPS}).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between consecutive clips in seconds (default: 0.0).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        logger.error("Input directory does not exist: {}", args.input)
        sys.exit(1)

    total = clip_videos_batch(
        input_dir=args.input,
        output_dir=args.output,
        duration=args.duration,
        fps=args.fps,
        overlap=args.overlap,
        max_workers=args.workers,
    )

    logger.info("Done. {} clips written to {}", total, args.output)
