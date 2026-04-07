"""Download videos from YouTube using yt-dlp for dataset building.

Supports downloading from:
- Direct URLs or URL lists
- Search queries (e.g. "drowning pool CCTV")
- Playlist URLs

Downloads are saved to data/raw_video/{class}/ directories.

Usage:
    # Download from a list of URLs
    python drowning_detector/scripts/download_youtube.py \
        --urls-file data/youtube_urls_drowning.txt \
        --output data/raw_video/drowning/

    # Download from search query
    python drowning_detector/scripts/download_youtube.py \
        --search "drowning pool CCTV footage" \
        --max-results 50 \
        --output data/raw_video/drowning/

    # Download from playlist
    python drowning_detector/scripts/download_youtube.py \
        --urls "https://www.youtube.com/playlist?list=PLxxxxx" \
        --output data/raw_video/swimming/
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from drowning_detector.core.config import PROJECT_ROOT

# Maximum video duration in seconds (skip very long videos)
MAX_DURATION = 600  # 10 minutes
# Preferred resolution
PREFERRED_HEIGHT = 720


def build_yt_dlp_opts(
    output_dir: Path,
    max_duration: int = MAX_DURATION,
    preferred_height: int = PREFERRED_HEIGHT,
) -> dict:
    """Build yt-dlp options dictionary.

    Args:
        output_dir: Directory to save downloaded videos.
        max_duration: Skip videos longer than this (seconds).
        preferred_height: Preferred video height in pixels.

    Returns:
        yt-dlp options dict.
    """
    return {
        "format": f"bestvideo[height<={preferred_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={preferred_height}][ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s_%(title).50s.%(ext)s"),
        "merge_output_format": "mp4",
        "match_filter": f"duration <= {max_duration}",
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": True,
        "no_color": True,
        "retries": 3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        "writeinfojson": True,  # save metadata alongside video
    }


def download_from_urls(
    urls: list[str],
    output_dir: Path,
    max_duration: int = MAX_DURATION,
) -> tuple[int, int]:
    """Download videos from a list of URLs.

    Args:
        urls: List of YouTube URLs.
        output_dir: Directory to save downloads.
        max_duration: Skip videos longer than this.

    Returns:
        Tuple of (successful, failed) download counts.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    opts = build_yt_dlp_opts(output_dir, max_duration)

    successful = 0
    failed = 0

    for i, url in enumerate(urls):
        url = url.strip()
        if not url or url.startswith("#"):
            continue

        logger.info("[{}/{}] Downloading: {}", i + 1, len(urls), url)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            successful += 1
        except Exception as e:
            logger.error("Failed to download {}: {}", url, e)
            failed += 1

    logger.info("Downloads complete: {} successful, {} failed", successful, failed)
    return successful, failed


def download_from_search(
    query: str,
    output_dir: Path,
    max_results: int = 50,
    max_duration: int = MAX_DURATION,
) -> tuple[int, int]:
    """Download videos from a YouTube search query.

    Args:
        query: Search query string.
        output_dir: Directory to save downloads.
        max_results: Maximum number of search results to download.
        max_duration: Skip videos longer than this.

    Returns:
        Tuple of (successful, failed) download counts.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    opts = build_yt_dlp_opts(output_dir, max_duration)

    search_url = f"ytsearch{max_results}:{query}"
    logger.info("Searching YouTube: '{}' (max {} results)", query, max_results)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([search_url])
    except Exception as e:
        logger.error("Search download failed: {}", e)
        return 0, 1

    # Count downloaded files
    downloaded = list(output_dir.glob("*.mp4"))
    logger.info("Downloaded {} videos for query: '{}'", len(downloaded), query)
    return len(downloaded), 0


def load_urls_from_file(file_path: Path) -> list[str]:
    """Load URLs from a text file (one URL per line).

    Args:
        file_path: Path to the URL list file.

    Returns:
        List of URL strings.
    """
    if not file_path.exists():
        logger.error("URL file not found: {}", file_path)
        return []

    urls = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    logger.info("Loaded {} URLs from {}", len(urls), file_path)
    return urls


def create_url_templates(data_root: Path) -> None:
    """Create template URL list files for each class.

    Args:
        data_root: Root data directory.
    """
    templates = {
        "youtube_urls_drowning.txt": [
            "# Drowning Detection — YouTube URLs",
            "# Add one URL per line. Lines starting with # are ignored.",
            "#",
            "# Search suggestions:",
            "#   - drowning pool CCTV",
            "#   - drowning rescue pool camera",
            "#   - drowning detection footage",
            "#   - pool emergency rescue",
            "#   - lifeguard drowning save",
            "#",
        ],
        "youtube_urls_treading.txt": [
            "# Treading Water (Hard Negative) — YouTube URLs",
            "#",
            "# Search suggestions:",
            "#   - treading water pool",
            "#   - water polo treading",
            "#   - swimming lesson treading water",
            "#   - synchronized swimming practice",
            "#",
        ],
        "youtube_urls_swimming.txt": [
            "# Swimming (Easy Negative) — YouTube URLs",
            "#",
            "# Search suggestions:",
            "#   - swimming pool overhead camera",
            "#   - swimming training CCTV",
            "#   - public pool swimming footage",
            "#   - swimming lanes overhead view",
            "#   - lap swimming pool camera",
            "#",
        ],
        "youtube_urls_splashing.txt": [
            "# Splashing / Playing (Easy Negative) — YouTube URLs",
            "#",
            "# Search suggestions:",
            "#   - kids playing pool",
            "#   - pool party footage",
            "#   - water park fun",
            "#   - pool splashing diving",
            "#",
        ],
    }

    for filename, lines in templates.items():
        filepath = data_root / filename
        if not filepath.exists():
            filepath.write_text("\n".join(lines) + "\n")
            logger.info("Created template: {}", filepath)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download YouTube videos for dataset building.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--urls",
        nargs="+",
        help="One or more YouTube URLs to download.",
    )
    source_group.add_argument(
        "--urls-file",
        type=Path,
        help="Path to a text file with one URL per line.",
    )
    source_group.add_argument(
        "--search",
        type=str,
        help="YouTube search query.",
    )
    source_group.add_argument(
        "--create-templates",
        action="store_true",
        help="Create template URL list files in data/ directory.",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for downloaded videos.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Max results for search mode (default: 50).",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=MAX_DURATION,
        help=f"Skip videos longer than this in seconds (default: {MAX_DURATION}).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = PROJECT_ROOT / "drowning_detector" / "data"

    if args.create_templates:
        create_url_templates(data_root)
        sys.exit(0)

    if not args.output:
        logger.error("--output is required for download mode")
        sys.exit(1)

    if args.urls:
        download_from_urls(args.urls, args.output, args.max_duration)
    elif args.urls_file:
        urls = load_urls_from_file(args.urls_file)
        if urls:
            download_from_urls(urls, args.output, args.max_duration)
    elif args.search:
        download_from_search(args.search, args.output, args.max_results, args.max_duration)
