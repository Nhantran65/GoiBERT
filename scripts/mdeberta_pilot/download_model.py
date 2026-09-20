"""Download only the pinned original mDeBERTa model/tokenizer files."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mdeberta_pilot import common  # noqa: E402


def main() -> int:
    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, str] = {}
    for filename in common.DOWNLOAD_FILES:
        path = hf_hub_download(
            repo_id=common.MODEL_REPO,
            filename=filename,
            revision=common.MODEL_COMMIT,
            local_dir=common.MODEL_DIR,
        )
        downloaded[filename] = str(path)
    allowed_metadata = {"README.md"}
    forbidden = sorted(
        path.name
        for path in common.MODEL_DIR.iterdir()
        if path.is_file() and path.name not in set(common.DOWNLOAD_FILES) | allowed_metadata
    )
    if forbidden:
        raise RuntimeError(f"Unexpected files in mDeBERTa directory: {forbidden}")
    tokenizer, tokenizer_manifest = common.load_tokenizer()
    del tokenizer
    common.write_json(
        common.ARTIFACT_ROOT / "model_download.json",
        {
            "status": "downloaded",
            "model": {
                "repo": common.MODEL_REPO,
                "commit": common.MODEL_COMMIT,
                "path": str(common.MODEL_DIR.relative_to(common.ROOT)),
                "files": downloaded,
                "sha256": {name: common.sha256_file(common.MODEL_DIR / name) for name in common.DOWNLOAD_FILES},
            },
            "tokenizer": tokenizer_manifest,
            "excluded": ["generator_config.json", "pytorch_model.generator.bin", "tf_model.h5"],
        },
    )
    print(f"downloaded {common.MODEL_REPO}@{common.MODEL_COMMIT} to {common.MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
