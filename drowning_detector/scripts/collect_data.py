"""Master data collection orchestrator.

Coordinates downloading from all sources and runs the full pipeline:
1. Download from YouTube (search + URL lists)
2. Download from Kinetics-700
3. Download from HMDB51
4. Clip all raw videos
5. Extract poses
6. Build annotations
7. Verify dataset

Usage:
    # Run everything
    python drowning_detector/scripts/collect_data.py --all

    # Download only (skip pipeline)
    python drowning_detector/scripts/collect_data.py --download-only

    # Pipeline only (already have raw videos)
    python drowning_detector/scripts/collect_data.py --pipeline-only

    # YouTube mining only
    python drowning_detector/scripts/collect_data.py --youtube
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "drowning_detector" / "data"
RAW_VIDEO_DIR = DATA_ROOT / "raw_video"
CLIPS_DIR = DATA_ROOT / "clips"

# YouTube search queries for each class
YOUTUBE_SEARCH_QUERIES: dict[str, list[str]] = {
    "drowning": [
        "drowning pool CCTV footage",
        "drowning rescue pool camera",
        "real drowning caught on camera pool",
        "lifeguard saves drowning swimmer",
        "pool drowning detection camera",
    ],
    "treading": [
        "treading water pool overhead",
        "water polo practice camera",
        "synchronized swimming practice overhead",
        "treading water lesson pool",
    ],
    "swimming": [
        "swimming pool overhead camera CCTV",
        "swimming lanes overhead view",
        "swimming training pool camera",
        "public pool CCTV footage",
        "lap swimming overhead camera",
    ],
    "splashing": [
        "kids playing pool camera",
        "pool party overhead footage",
        "water park fun pool camera",
        "diving board pool camera",
    ],
}


def run_youtube_downloads(
    max_results_per_query: int = 20,
    max_workers: int = 4,
) -> None:
    """Run YouTube search downloads for all classes."""
    from drowning_detector.scripts.download_youtube import download_from_search

    logger.info("=" * 60)
    logger.info("STAGE: YouTube Mining")
    logger.info("=" * 60)

    for class_name, queries in YOUTUBE_SEARCH_QUERIES.items():
        output_dir = RAW_VIDEO_DIR / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for query in queries:
            logger.info("Searching: '{}' → {}/", query, class_name)
            try:
                download_from_search(
                    query=query,
                    output_dir=output_dir,
                    max_results=max_results_per_query,
                )
            except Exception as e:
                logger.error("Search failed for '{}': {}", query, e)

    # Also check for URL list files
    for class_name in YOUTUBE_SEARCH_QUERIES:
        urls_file = DATA_ROOT / f"youtube_urls_{class_name}.txt"
        if urls_file.exists():
            from drowning_detector.scripts.download_youtube import (
                download_from_urls,
                load_urls_from_file,
            )

            urls = load_urls_from_file(urls_file)
            if urls:
                logger.info("Downloading {} URLs from {}", len(urls), urls_file)
                download_from_urls(urls, RAW_VIDEO_DIR / class_name)


def run_kinetics_download(
    max_per_class: int | None = None,
    workers: int = 4,
) -> None:
    """Download Kinetics-700 swimming clips."""
    from drowning_detector.scripts.download_kinetics import (
        RELEVANT_CLASSES,
        download_kinetics_batch,
        download_kinetics_csv,
        parse_kinetics_csv,
    )

    logger.info("=" * 60)
    logger.info("STAGE: Kinetics-700 Download")
    logger.info("=" * 60)

    cache_dir = DATA_ROOT / ".cache"

    for split in ["train", "val"]:
        try:
            csv_path = download_kinetics_csv(split, cache_dir)
            records = parse_kinetics_csv(csv_path, RELEVANT_CLASSES)

            if records:
                download_kinetics_batch(
                    records,
                    RAW_VIDEO_DIR,
                    max_workers=workers,
                    max_per_class=max_per_class,
                )
        except Exception as e:
            logger.error("Kinetics {} download failed: {}", split, e)


def run_hmdb51_download() -> None:
    """Download HMDB51 swim clips."""
    from drowning_detector.scripts.download_hmdb51 import (
        RELEVANT_CLASSES,
        copy_to_output,
        download_hmdb51_archive,
        extract_hmdb51,
    )

    logger.info("=" * 60)
    logger.info("STAGE: HMDB51 Download")
    logger.info("=" * 60)

    import shutil
    import tempfile

    cache_dir = DATA_ROOT / ".cache"

    try:
        archive_path = download_hmdb51_archive(cache_dir)
        temp_dir = Path(tempfile.mkdtemp(prefix="hmdb51_"))
        extracted = extract_hmdb51(archive_path, temp_dir, set(RELEVANT_CLASSES.keys()))
        if extracted:
            copy_to_output(extracted, RAW_VIDEO_DIR)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logger.error("HMDB51 download failed: {}", e)


def run_pipeline() -> None:
    """Run the full data processing pipeline: clip → pose → annotate → verify."""
    from drowning_detector.scripts.build_annotations import build_annotations
    from drowning_detector.scripts.clip_videos import clip_videos_batch
    from drowning_detector.scripts.extract_poses import extract_all_poses
    from drowning_detector.scripts.verify_dataset import verify_dataset

    logger.info("=" * 60)
    logger.info("STAGE: Data Processing Pipeline")
    logger.info("=" * 60)

    # Step 1: Clip raw videos
    logger.info("--- Step 1/4: Clipping raw videos ---")
    for class_name in ["drowning", "treading", "swimming", "splashing"]:
        input_dir = RAW_VIDEO_DIR / class_name
        output_dir = CLIPS_DIR / class_name

        if not input_dir.exists():
            logger.warning("No raw videos for class '{}', skipping", class_name)
            continue

        clip_videos_batch(input_dir, output_dir, duration=5.0, fps=10)

    # Step 2: Extract poses
    logger.info("--- Step 2/4: Extracting poses ---")
    extract_all_poses(DATA_ROOT)

    # Step 3: Build annotations
    logger.info("--- Step 3/4: Building annotations ---")
    build_annotations(
        data_root=DATA_ROOT,
        output_path=DATA_ROOT / "annotations.csv",
    )

    # Step 4: Verify dataset
    logger.info("--- Step 4/4: Verifying dataset ---")
    report = verify_dataset(DATA_ROOT)
    print(report.summary())

    if report.passed:
        logger.info("Dataset is ready for training!")
    else:
        logger.warning("Dataset has issues — review the report above")


def print_status() -> None:
    """Print current data collection status."""
    logger.info("=" * 60)
    logger.info("DATA COLLECTION STATUS")
    logger.info("=" * 60)

    for stage_name, stage_dir in [("Raw Video", RAW_VIDEO_DIR), ("Clips", CLIPS_DIR)]:
        logger.info("\n{}:", stage_name)
        if not stage_dir.exists():
            logger.info("  (directory not found)")
            continue

        for class_name in ["drowning", "treading", "swimming", "splashing"]:
            class_dir = stage_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.*")))
                logger.info("  {}: {} files", class_name, count)
            else:
                logger.info("  {}: 0 files", class_name)

    poses_dir = DATA_ROOT / "poses"
    logger.info("\nPoses:")
    if poses_dir.exists():
        for class_name in ["drowning", "treading", "swimming", "splashing"]:
            class_dir = poses_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.npy")))
                logger.info("  {}: {} files", class_name, count)
            else:
                logger.info("  {}: 0 files", class_name)

    annotations = DATA_ROOT / "annotations.csv"
    if annotations.exists():
        import pandas as pd

        df = pd.read_csv(annotations)
        logger.info("\nAnnotations: {} total samples", len(df))
        logger.info(df["label_name"].value_counts().to_string())
    else:
        logger.info("\nAnnotations: not built yet")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Master data collection orchestrator.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run everything: download all sources + full pipeline.",
    )
    mode_group.add_argument(
        "--download-only",
        action="store_true",
        help="Download from all sources, skip processing pipeline.",
    )
    mode_group.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run processing pipeline only (clip → pose → annotate → verify).",
    )
    mode_group.add_argument(
        "--youtube",
        action="store_true",
        help="YouTube mining only.",
    )
    mode_group.add_argument(
        "--kinetics",
        action="store_true",
        help="Kinetics-700 download only.",
    )
    mode_group.add_argument(
        "--hmdb51",
        action="store_true",
        help="HMDB51 download only.",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Print current data collection status.",
    )

    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=20,
        help="Max YouTube results per search query (default: 20).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max Kinetics clips per class (default: all).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Parallel workers (default: 4).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.all or args.download_only:
        run_youtube_downloads(args.max_results_per_query, args.workers)
        run_kinetics_download(args.max_per_class, args.workers)
        run_hmdb51_download()

    if args.youtube:
        run_youtube_downloads(args.max_results_per_query, args.workers)

    if args.kinetics:
        run_kinetics_download(args.max_per_class, args.workers)

    if args.hmdb51:
        run_hmdb51_download()

    if args.all or args.pipeline_only:
        run_pipeline()

    # Print final status
    print_status()
