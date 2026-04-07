"""Download swimming-related clips from the Kinetics-700 dataset.

Kinetics-700 provides 10-second YouTube clips with activity labels. We download
clips from relevant classes: swimming, treading water, diving, water polo, etc.

The Kinetics dataset is distributed as CSV files with YouTube IDs + timestamps.
This script downloads the CSVs, filters for relevant classes, and downloads clips.

Usage:
    # Download Kinetics swimming/water activity clips
    python drowning_detector/scripts/download_kinetics.py \
        --output data/raw_video/swimming/ \
        --split train

    # List available water-related classes
    python drowning_detector/scripts/download_kinetics.py --list-classes
"""

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT

# Kinetics-700 CSV URLs (hosted by DeepMind)
KINETICS_CSV_URLS = {
    "train": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/train.csv",
    "val": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/validate.csv",
    "test": "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020/test.csv",
}

# Kinetics-700 classes relevant to our project
RELEVANT_CLASSES = {
    # Direct swimming/water activities
    "swimming backstroke",
    "swimming breast stroke",
    "swimming butterfly stroke",
    "front crawl",
    "treading water",
    "diving cliff",
    "springboard diving",
    "water sliding",
    "water polo",
    "synchronized swimming",
    # Pool/water adjacent
    "canoe slalom",
    "surfing water",
    "kayaking",
    "snorkeling",
    "scuba diving",
}

# Map Kinetics classes to our output folders
CLASS_OUTPUT_MAP: dict[str, str] = {
    "swimming backstroke": "swimming",
    "swimming breast stroke": "swimming",
    "swimming butterfly stroke": "swimming",
    "front crawl": "swimming",
    "treading water": "treading",
    "diving cliff": "splashing",
    "springboard diving": "splashing",
    "water sliding": "splashing",
    "water polo": "treading",
    "synchronized swimming": "swimming",
    "canoe slalom": "swimming",
    "surfing water": "swimming",
    "kayaking": "swimming",
    "snorkeling": "swimming",
    "scuba diving": "swimming",
}


def download_kinetics_csv(split: str, cache_dir: Path) -> Path:
    """Download and cache the Kinetics CSV annotation file.

    Args:
        split: Dataset split ("train", "val", or "test").
        cache_dir: Directory to cache downloaded CSV files.

    Returns:
        Path to the downloaded CSV file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"kinetics700_{split}.csv"

    if csv_path.exists():
        logger.info("Using cached CSV: {}", csv_path)
        return csv_path

    url = KINETICS_CSV_URLS[split]
    logger.info("Downloading Kinetics-700 {} CSV...", split)

    try:
        import httpx

        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
        logger.info("Saved CSV to {}", csv_path)
    except Exception as e:
        logger.error("Failed to download CSV: {}", e)
        raise

    return csv_path


def parse_kinetics_csv(csv_path: Path, classes: set[str]) -> list[dict]:
    """Parse Kinetics CSV and filter for relevant classes.

    Args:
        csv_path: Path to Kinetics CSV file.
        classes: Set of class names to filter for.

    Returns:
        List of dicts with keys: label, youtube_id, time_start, time_end, split.
    """
    records = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            if label.lower() in {c.lower() for c in classes}:
                records.append({
                    "label": label,
                    "youtube_id": row.get("youtube_id", "").strip(),
                    "time_start": int(row.get("time_start", 0)),
                    "time_end": int(row.get("time_end", 0)),
                })

    logger.info("Found {} clips for {} relevant classes", len(records), len(classes))

    # Log per-class counts
    class_counts: dict[str, int] = {}
    for r in records:
        class_counts[r["label"]] = class_counts.get(r["label"], 0) + 1
    for cls, count in sorted(class_counts.items()):
        logger.info("  {}: {} clips", cls, count)

    return records


def download_kinetics_clip(
    record: dict,
    output_dir: Path,
) -> bool:
    """Download a single Kinetics clip using yt-dlp.

    Args:
        record: Dict with youtube_id, time_start, time_end, label.
        output_dir: Directory to save the clip.

    Returns:
        True if download succeeded.
    """
    youtube_id = record["youtube_id"]
    start = record["time_start"]
    end = record["time_end"]
    label = record["label"].replace(" ", "_")

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_path = output_dir / f"kinetics_{label}_{youtube_id}_{start}-{end}.mp4"

    if output_path.exists():
        return True  # already downloaded

    # Use yt-dlp with ffmpeg section download for efficiency
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "--output", str(output_path),
        "--retries", "3",
        "--no-overwrites",
        url,
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def download_kinetics_batch(
    records: list[dict],
    base_output_dir: Path,
    max_workers: int = 4,
    max_per_class: Optional[int] = None,
) -> dict[str, tuple[int, int]]:
    """Download Kinetics clips in parallel, organized by our class folders.

    Args:
        records: List of clip records from parse_kinetics_csv.
        base_output_dir: Base output directory (e.g. data/raw_video/).
        max_workers: Number of parallel download threads.
        max_per_class: Maximum clips to download per class (None = all).

    Returns:
        Dict mapping class_name → (successful, failed) counts.
    """
    # Group records by our output class
    grouped: dict[str, list[dict]] = {}
    for record in records:
        our_class = CLASS_OUTPUT_MAP.get(record["label"].lower(), "swimming")
        grouped.setdefault(our_class, []).append(record)

    results: dict[str, tuple[int, int]] = {}

    for our_class, class_records in grouped.items():
        if max_per_class:
            class_records = class_records[:max_per_class]

        output_dir = base_output_dir / our_class
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading {} clips for class '{}' to {}",
            len(class_records),
            our_class,
            output_dir,
        )

        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_kinetics_clip, r, output_dir): r
                for r in class_records
            }
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1

                total = successful + failed
                if total % 25 == 0:
                    logger.info(
                        "  Progress: {}/{} ({} ok, {} failed)",
                        total,
                        len(class_records),
                        successful,
                        failed,
                    )

        results[our_class] = (successful, failed)
        logger.info(
            "Class '{}': {} downloaded, {} failed",
            our_class,
            successful,
            failed,
        )

    return results


def list_relevant_classes() -> None:
    """Print all relevant Kinetics classes and their mappings."""
    logger.info("Relevant Kinetics-700 classes and our label mapping:")
    for kinetics_class in sorted(RELEVANT_CLASSES):
        our_class = CLASS_OUTPUT_MAP.get(kinetics_class, "unknown")
        logger.info("  {} → {}", kinetics_class, our_class)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download swimming-related clips from Kinetics-700.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "drowning_detector" / "data" / "raw_video",
        help="Base output directory for raw videos.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Kinetics split to download (default: train).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max clips per class (default: all).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Parallel download threads (default: 4).",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List relevant Kinetics classes and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_classes:
        list_relevant_classes()
        sys.exit(0)

    cache_dir = PROJECT_ROOT / "drowning_detector" / "data" / ".cache"

    csv_path = download_kinetics_csv(args.split, cache_dir)
    records = parse_kinetics_csv(csv_path, RELEVANT_CLASSES)

    if not records:
        logger.error("No relevant clips found in {} split", args.split)
        sys.exit(1)

    results = download_kinetics_batch(
        records,
        args.output,
        max_workers=args.workers,
        max_per_class=args.max_per_class,
    )

    logger.info("Download summary:")
    for cls, (ok, fail) in results.items():
        logger.info("  {}: {} downloaded, {} failed", cls, ok, fail)
