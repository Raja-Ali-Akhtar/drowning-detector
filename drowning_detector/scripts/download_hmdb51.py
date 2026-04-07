"""Download swimming-related clips from the HMDB51 dataset.

HMDB51 is a human motion recognition dataset with 51 action classes.
The dataset is distributed as a single .rar archive from the HMDB website.
We extract only the "swim" class clips.

Usage:
    python drowning_detector/scripts/download_hmdb51.py \
        --output data/raw_video/swimming/

    # Only extract specific classes
    python drowning_detector/scripts/download_hmdb51.py \
        --output data/raw_video/ \
        --classes swim dive
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT

# HMDB51 download URL (official mirror)
HMDB51_URL = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"

# HMDB51 classes relevant to our project
RELEVANT_CLASSES = {
    "swim": "swimming",
    "dive": "splashing",
}


def download_hmdb51_archive(cache_dir: Path) -> Path:
    """Download the HMDB51 dataset archive.

    Args:
        cache_dir: Directory to cache the downloaded archive.

    Returns:
        Path to the downloaded .rar file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "hmdb51_org.rar"

    if archive_path.exists():
        logger.info("Using cached archive: {}", archive_path)
        return archive_path

    logger.info("Downloading HMDB51 dataset (~2GB)...")
    logger.info("This may take a while depending on your connection speed.")

    try:
        import httpx

        with httpx.stream("GET", HMDB51_URL, follow_redirects=True, timeout=600) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(archive_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total and downloaded % (50 * 1024 * 1024) < 8192:  # Log every ~50MB
                        pct = (downloaded / total) * 100
                        logger.info("  Download progress: {:.0f}%", pct)

        logger.info("Download complete: {}", archive_path)
    except Exception as e:
        # Clean up partial download
        if archive_path.exists():
            archive_path.unlink()
        logger.error("Failed to download HMDB51: {}", e)
        raise

    return archive_path


def extract_hmdb51(
    archive_path: Path,
    extract_dir: Path,
    classes: set[str],
) -> dict[str, list[Path]]:
    """Extract relevant classes from the HMDB51 archive.

    HMDB51 structure: outer .rar → per-class .rar files → .avi clips.

    Args:
        archive_path: Path to the HMDB51 .rar archive.
        extract_dir: Temporary directory for extraction.
        classes: Set of HMDB51 class names to extract.

    Returns:
        Dict mapping class_name → list of extracted video paths.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Check for unrar or 7z
    extractor = _find_archive_tool()
    if not extractor:
        logger.error(
            "No archive extraction tool found. Install one of:\n"
            "  - unrar: sudo apt install unrar\n"
            "  - 7z: sudo apt install p7zip-full\n"
            "  - On Windows: install 7-Zip or WinRAR"
        )
        sys.exit(1)

    # Step 1: Extract outer archive to get per-class .rar files
    logger.info("Extracting outer archive...")
    _extract_archive(extractor, archive_path, extract_dir)

    # Step 2: Find and extract relevant class archives
    results: dict[str, list[Path]] = {}

    for hmdb_class in classes:
        class_archive = extract_dir / f"{hmdb_class}.rar"
        if not class_archive.exists():
            logger.warning("Class archive not found: {}", class_archive)
            continue

        class_dir = extract_dir / hmdb_class
        class_dir.mkdir(exist_ok=True)

        logger.info("Extracting class '{}' ...", hmdb_class)
        _extract_archive(extractor, class_archive, class_dir)

        # Collect video files
        videos = sorted(class_dir.glob("*.avi"))
        results[hmdb_class] = videos
        logger.info("  Extracted {} clips for '{}'", len(videos), hmdb_class)

    return results


def _find_archive_tool() -> str | None:
    """Find an available archive extraction tool."""
    for tool in ["unrar", "7z", "7za"]:
        if shutil.which(tool):
            return tool
    return None


def _extract_archive(tool: str, archive_path: Path, output_dir: Path) -> None:
    """Extract an archive using the available tool."""
    if tool == "unrar":
        cmd = ["unrar", "x", "-o+", str(archive_path), str(output_dir) + "/"]
    elif tool in ("7z", "7za"):
        cmd = [tool, "x", f"-o{output_dir}", "-y", str(archive_path)]
    else:
        raise RuntimeError(f"Unsupported archive tool: {tool}")

    subprocess.run(cmd, capture_output=True, text=True, check=True)


def copy_to_output(
    extracted: dict[str, list[Path]],
    base_output_dir: Path,
) -> int:
    """Copy extracted clips to the project data directory.

    Args:
        extracted: Dict mapping hmdb_class → list of video paths.
        base_output_dir: Base output directory (e.g. data/raw_video/).

    Returns:
        Total number of clips copied.
    """
    total = 0

    for hmdb_class, videos in extracted.items():
        our_class = RELEVANT_CLASSES.get(hmdb_class, "swimming")
        output_dir = base_output_dir / our_class
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos:
            dest = output_dir / f"hmdb51_{video_path.name}"
            if not dest.exists():
                shutil.copy2(video_path, dest)
                total += 1

        logger.info("Copied {} clips for '{}' → {}", len(videos), hmdb_class, output_dir)

    return total


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract swimming clips from HMDB51.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "drowning_detector" / "data" / "raw_video",
        help="Base output directory for raw videos.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=f"HMDB51 classes to extract (default: {list(RELEVANT_CLASSES.keys())}).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary extraction directory (useful for debugging).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classes = set(args.classes) if args.classes else set(RELEVANT_CLASSES.keys())

    cache_dir = PROJECT_ROOT / "drowning_detector" / "data" / ".cache"

    # Download
    archive_path = download_hmdb51_archive(cache_dir)

    # Extract to temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix="hmdb51_"))
    try:
        extracted = extract_hmdb51(archive_path, temp_dir, classes)

        if not extracted:
            logger.error("No clips extracted")
            sys.exit(1)

        # Copy to output
        total = copy_to_output(extracted, args.output)
        logger.info("Total clips copied: {}", total)

    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp directory")
        else:
            logger.info("Temp directory kept at: {}", temp_dir)
