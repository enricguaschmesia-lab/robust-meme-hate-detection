"""Download a Hateful Memes mirror from Hugging Face Hub.

Recommended source:
    cs5242-hateful-memes/hateful-memes-data

The user-suggested `emily49/hateful-memes` repository is kept as an option, but
it only exposes JSONL metadata via the Hub API and does not include image files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCES = {
    "cs5242": {
        "repo_id": "cs5242-hateful-memes/hateful-memes-data",
        "patterns": ["*.jsonl", "img/*.png", "LICENSE.txt", "README.md"],
        "jsonl_files": ["train.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl", "test_seen.jsonl", "test_unseen.jsonl"],
    },
    "neuralcatcher": {
        "repo_id": "neuralcatcher/hateful_memes",
        "patterns": ["*.jsonl", "img/*.png", "LICENSE.txt", "README.md"],
        "jsonl_files": ["train.jsonl", "dev_seen.jsonl", "dev_unseen.jsonl", "test_seen.jsonl", "test_unseen.jsonl"],
    },
    "emily49_metadata": {
        "repo_id": "emily49/hateful-memes",
        "patterns": ["*.jsonl", "LICENSE.txt", "README.md"],
        "jsonl_files": ["train.jsonl", "dev.jsonl", "test.jsonl"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="cs5242",
        help="Hugging Face mirror to download. Default is the complete cs5242 mirror.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/hateful_memes"),
        help="Local directory where files will be downloaded.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Not required for the recommended public mirrors.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "snapshot", "direct"],
        default="auto",
        help=(
            "Download method. `snapshot` uses huggingface_hub; `direct` streams raw resolve URLs. "
            "`auto` tries snapshot first, then direct."
        ),
    )
    parser.add_argument("--revision", default="main", help="Hub revision to download from.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file for direct downloads.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected source and destination without downloading.",
    )
    return parser.parse_args()


def _resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def _download_url(url: str, destination: Path, retries: int = 3, ignore_errors: bool = False) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return True

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    headers = {"User-Agent": "robust-meme-hate-detection/0.1"}
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(destination)
            return True
        except HTTPError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if ignore_errors and exc.code == 404:
                return False
            if attempt == retries:
                if ignore_errors:
                    return False
                raise RuntimeError(f"Failed to download {url}") from exc
            time.sleep(2 * attempt)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                if ignore_errors:
                    return False
                raise RuntimeError(f"Failed to download {url}") from exc
            time.sleep(2 * attempt)
    return False


def _collect_image_refs(output_dir: Path, jsonl_files: list[str]) -> list[str]:
    refs: set[str] = set()
    for jsonl_file in jsonl_files:
        path = output_dir / jsonl_file
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                image_ref = item.get("img") or item.get("image") or item.get("image_path")
                if image_ref:
                    refs.add(str(image_ref).replace("\\", "/"))
    return sorted(refs)


def _direct_download(
    repo_id: str,
    revision: str,
    output_dir: Path,
    jsonl_files: list[str],
    max_images: int | None,
    retries: int,
) -> None:
    static_files = ["README.md", "LICENSE.txt", *jsonl_files]
    for filename in static_files:
        url = _resolve_url(repo_id, revision, filename)
        print(f"Downloading {filename}")
        _download_url(url, output_dir / filename, retries=retries)

    image_refs = _collect_image_refs(output_dir, jsonl_files)
    if max_images is not None:
        image_refs = image_refs[:max_images]

    print(f"Downloading {len(image_refs)} image files")
    for index, image_ref in enumerate(image_refs, start=1):
        if index == 1 or index % 100 == 0 or index == len(image_refs):
            print(f"  [{index}/{len(image_refs)}] {image_ref}")
        url = _resolve_url(repo_id, revision, image_ref)
        ok = _download_url(url, output_dir / image_ref, retries=retries, ignore_errors=True)
        if not ok:
            print(f"  failed or missing after retries, skipped for this pass: {image_ref}", file=sys.stderr)


def _snapshot_download(repo_id: str, output_dir: Path, patterns: list[str], token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: huggingface_hub. Install it with `pip install -r requirements.txt`.") from exc

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=patterns,
        token=token,
    )


def main() -> int:
    args = parse_args()
    source = SOURCES[args.source]
    repo_id = source["repo_id"]
    patterns = source["patterns"]
    jsonl_files = source["jsonl_files"]

    print(f"Selected source: {args.source} ({repo_id})")
    print(f"Output directory: {args.output_dir}")
    print("Allow patterns:")
    for pattern in patterns:
        print(f"  - {pattern}")

    if args.source == "emily49_metadata":
        print(
            "Warning: emily49/hateful-memes appears to contain JSONL metadata only. "
            "Use cs5242 or neuralcatcher for image files.",
            file=sys.stderr,
        )

    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.method in {"auto", "snapshot"}:
        try:
            _snapshot_download(repo_id, args.output_dir, patterns, args.token)
        except Exception as exc:
            if args.method == "snapshot":
                raise
            print(f"Snapshot download failed; falling back to direct file downloads. Reason: {exc}", file=sys.stderr)
        else:
            print("Download complete.")
            return 0

    _direct_download(
        repo_id=repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        jsonl_files=jsonl_files,
        max_images=args.max_images,
        retries=args.retries,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
