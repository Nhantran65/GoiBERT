"""Build the Stage P general-plus-domain corpus for continued pretraining."""

from __future__ import annotations

import argparse
import codecs
import json
import lzma
import random
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterable


CC100_VI_URL = "https://data.statmt.org/cc-100/vi.txt.xz"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
MIB = 1024 * 1024
MIN_LINE_CHARS = 20


def vietnameseish(text: str) -> bool:
    """Use a cheap stdlib-only Latin/Vietnamese text heuristic."""
    if len(text) < MIN_LINE_CHARS:
        return False
    allowed = 0
    for character in text:
        category = unicodedata.category(character)
        name = unicodedata.name(character, "")
        if (
            character.isspace()
            or character.isdigit()
            or category.startswith(("P", "S"))
            or (category.startswith("L") and "LATIN" in name)
        ):
            allowed += 1
    return allowed / len(text) >= 0.60


def _filtered_lines(decoded: str, carry: str) -> tuple[list[str], str]:
    parts = (carry + decoded).split("\n")
    complete, carry = parts[:-1], parts[-1]
    return [line.strip() for line in complete if vietnameseish(line.strip())], carry


def download_cc100_prefix(range_bytes: int, output_bytes: int) -> list[str]:
    """Download and decode only a bounded compressed prefix of Vietnamese CC100."""
    request = urllib.request.Request(
        CC100_VI_URL,
        headers={
            "Range": f"bytes=0-{range_bytes - 1}",
            "User-Agent": "TwoHearts-GoiBERT-Stage-P/1.0",
        },
    )
    decompressor = lzma.LZMADecompressor()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    lines: list[str] = []
    emitted_bytes = 0
    downloaded_bytes = 0
    next_progress_bytes = 64 * MIB
    carry = ""
    remaining = range_bytes
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            while remaining > 0 and emitted_bytes < output_bytes:
                chunk = response.read(min(MIB, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                downloaded_bytes += len(chunk)
                decoded = decoder.decode(decompressor.decompress(chunk))
                candidates, carry = _filtered_lines(decoded, carry)
                for line in candidates:
                    line_bytes = len(line.encode("utf-8")) + 1
                    if emitted_bytes + line_bytes > output_bytes:
                        return lines
                    lines.append(line)
                    emitted_bytes += line_bytes
                if downloaded_bytes >= next_progress_bytes:
                    print(
                        f"CC100 compressed={downloaded_bytes / MIB:.0f} MiB "
                        f"filtered={emitted_bytes / MIB:.0f} MiB lines={len(lines):,}",
                        flush=True,
                    )
                    next_progress_bytes += 64 * MIB
    except (EOFError, lzma.LZMAError):
        # A byte-range prefix is intentionally a truncated XZ stream. Complete decoded
        # lines before the truncation remain valid corpus input.
        pass
    return lines


def _clean_domain_lines(lines: Iterable[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def load_domain_sources(repo_root: Path) -> dict[str, list[str]]:
    """Load train and validation text plus PhoATIS utterances from local files."""
    from data import load_dataset

    sources: dict[str, list[str]] = {}
    for dataset in ("vsfc", "visfd", "vlsp_restaurant"):
        splits = load_dataset(dataset, repo_root)
        sources[dataset] = _clean_domain_lines(
            [*splits.train_texts, *splits.valid_texts]
        )
    phoatis_paths = sorted((repo_root / "data" / "raw" / "phoatis").glob("*/seq.in"))
    if not phoatis_paths:
        raise FileNotFoundError("No PhoATIS */seq.in files were found")
    phoatis: list[str] = []
    for path in phoatis_paths:
        phoatis.extend(path.read_text(encoding="utf-8").splitlines())
    sources["phoatis"] = _clean_domain_lines(phoatis)
    return sources


def deduplicate_domain_sources(
    sources: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Deduplicate exact domain lines globally while retaining source attribution."""
    seen: set[str] = set()
    unique_sources: dict[str, list[str]] = {}
    block: list[str] = []
    for source, lines in sources.items():
        unique: list[str] = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            unique.append(line)
            block.append(line)
        unique_sources[source] = unique
    return unique_sources, block


def write_corpus(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output_dir = repo_root / "data" / "corpus_v1"
    corpus_path = output_dir / "mlm_mix.txt"
    stats_path = output_dir / "stats.json"
    general_mb = 1 if args.smoke else args.general_mb
    range_mb = 2 if args.smoke else args.range_mb
    repeat = min(args.domain_repeat, 2) if args.smoke else args.domain_repeat
    general_lines = download_cc100_prefix(range_mb * MIB, general_mb * MIB)
    if not general_lines:
        raise RuntimeError("The CC100 prefix produced no complete filtered lines")
    raw_sources = load_domain_sources(repo_root)
    domain_sources, domain_block = deduplicate_domain_sources(raw_sources)
    mixed_lines = [*general_lines, *(domain_block * repeat)]
    random.Random(args.seed).shuffle(mixed_lines)

    # Intent: produce one deterministic domain-weighted corpus from bounded public data.
    # Status: exact domain duplicates are removed before block repetition and seeded shuffle.
    # Next: run mlm_pretrain.py against the emitted corpus on LUMI.
    # Blocker: none after the local datasets and CC100 endpoint are reachable.
    # Confidence: high; stats record both unique and emitted source counts.
    output_dir.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, line in enumerate(mixed_lines, start=1):
            handle.write(line)
            handle.write("\n")
            if index % 1_000_000 == 0:
                print(
                    f"Corpus write progress: {index:,}/{len(mixed_lines):,} lines",
                    flush=True,
                )
    stats = {
        "source_line_counts": {
            "cc100_vi": len(general_lines),
            **{name: len(lines) for name, lines in domain_sources.items()},
        },
        "emitted_line_counts": {
            "cc100_vi": len(general_lines),
            **{name: len(lines) * repeat for name, lines in domain_sources.items()},
        },
        "total_lines": len(mixed_lines),
        "total_bytes": corpus_path.stat().st_size,
        "seed": args.seed,
        "domain_repeat": repeat,
        "general_mb": general_mb,
        "compressed_range_mb": range_mb,
        "cc100_url": CC100_VI_URL,
        "minimum_line_characters": MIN_LINE_CHARS,
        "allowed_character_ratio": 0.60,
        "domain_splits": "train+valid",
        "smoke": args.smoke,
    }
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote {corpus_path} lines={len(mixed_lines):,} bytes={stats['total_bytes']:,}",
        flush=True,
    )
    print(f"Wrote {stats_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-mb", type=int, default=1500)
    parser.add_argument(
        "--range-mb",
        type=int,
        default=2048,
        help="Maximum compressed CC100 prefix to request; never downloads the full object.",
    )
    parser.add_argument("--domain-repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.general_mb < 1 or args.range_mb < 1 or args.domain_repeat < 1:
        parser.error("--general-mb, --range-mb, and --domain-repeat must be positive")
    return args


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    general_mb = 1 if args.smoke else args.general_mb
    range_mb = 2 if args.smoke else args.range_mb
    repeat = min(args.domain_repeat, 2) if args.smoke else args.domain_repeat
    if args.dry_run:
        print(
            f"Dry run: url={CC100_VI_URL} compressed_range_mb={range_mb} "
            f"general_mb={general_mb} domain_repeat={repeat} seed={args.seed}"
        )
        print(f"Output corpus: {repo_root / 'data' / 'corpus_v1' / 'mlm_mix.txt'}")
        print(f"Output stats: {repo_root / 'data' / 'corpus_v1' / 'stats.json'}")
        print("Dry run complete: no network request or heavy import was performed.")
        return
    write_corpus(args)


if __name__ == "__main__":
    main()
