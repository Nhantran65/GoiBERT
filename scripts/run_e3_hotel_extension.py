"""Run the single approved E3 Hotel screening extension.

The original E3 source and orchestrator are intentionally left frozen.  This
small launcher reuses their trainer and loss, temporarily adds the audited
Hotel task contract, and writes the result below ``artifacts/e3/hotel_extension``.
It is deliberately separate from ``execute_e3_validation.py`` so the original
two-trial plan cannot be broadened accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import goibert_e3.train as e3_train  # noqa: E402
from goibert.train import _json_default, sha256_file  # noqa: E402


TASK_ID = "absa_hotel"
SEED = 42
EPOCHS = 30
DEVICE = "cuda:2"
EXTENSION_PROTOCOL = "E3-conditional-polarity-hotel-extension-v1"
EXTENSION_REVISION = "e3-hotel-20260909-r1"
EXPECTED_E3_SOURCE_HASH = "a17ec51f8386162e34ed35a7d4695feb06bb813d21c8c812b221b1084b1b2777"
HOTEL_COUNTS = (4,) * 34
HOTEL_POLARITY_SLOTS = tuple(range(34))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _extension_source_hash() -> str:
    return sha256_file(Path(__file__).resolve())


def _hotel_polarity_slots(task_id: str, counts: Sequence[int]) -> tuple[int, ...]:
    if task_id == TASK_ID:
        resolved = tuple(int(value) for value in counts)
        if resolved != HOTEL_COUNTS:
            raise ValueError(
                "Hotel slot schema drifted: expected 34 four-class slots, "
                f"found {resolved!r}"
            )
        return HOTEL_POLARITY_SLOTS
    return _ORIGINAL_POLARITY_SLOTS(task_id, counts)


def _annotate_artifact(artifact_dir: Path, extension_hash: str, source_hashes: dict[str, str]) -> None:
    annotation = {
        "extension_protocol": EXTENSION_PROTOCOL,
        "extension_revision": EXTENSION_REVISION,
        "extension_source_hash": extension_hash,
        "original_e3_source_hash": source_hashes["e3_source"],
        "e1_dependency_source_hash": source_hashes["e1_dependency"],
        "task_id": TASK_ID,
        "seed": SEED,
        "slot_class_counts": list(HOTEL_COUNTS),
        "polarity_slots": list(HOTEL_POLARITY_SLOTS),
        "mapping_contract": "UIT-ABSA Hotel: 34 aspects; every slot has absent + 3 polarity classes",
        "include_test": False,
        "test_access": "not_loaded",
    }
    _write_json(artifact_dir / "hotel_extension_provenance.json", annotation)
    for name in ("metadata.json", "resolved_config.json"):
        path = artifact_dir / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(annotation)
            _write_json(path, payload)


def run(output_dir: Path, device: str = DEVICE) -> dict[str, Any]:
    source_hashes = e3_train.source_hashes(REPO_ROOT)
    if source_hashes["e3_source"] != EXPECTED_E3_SOURCE_HASH:
        raise RuntimeError(
            "Frozen E3 dependency changed: "
            f"expected {EXPECTED_E3_SOURCE_HASH}, found {source_hashes['e3_source']}"
        )
    extension_hash = _extension_source_hash()

    global _ORIGINAL_TASK_IDS, _ORIGINAL_POLARITY_SLOTS
    _ORIGINAL_TASK_IDS = e3_train.TASK_IDS
    _ORIGINAL_POLARITY_SLOTS = e3_train._polarity_slots
    if TASK_ID in _ORIGINAL_TASK_IDS:
        raise RuntimeError("Hotel is unexpectedly present in the frozen E3 task list")

    # The monkeypatch is process-local.  It lets the frozen trainer reuse its
    # exact optimizer, loader, scheduler, checkpoint, and validation path.
    e3_train.TASK_IDS = _ORIGINAL_TASK_IDS + (TASK_ID,)
    e3_train._polarity_slots = _hotel_polarity_slots
    try:
        result = e3_train.train_trial(
            task_id=TASK_ID,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            seed=SEED,
            device_name=device,
            precision="bf16",
            attention_implementation="sdpa",
            epochs=EPOCHS,
            expected_source_hash=EXPECTED_E3_SOURCE_HASH,
        )
    finally:
        e3_train.TASK_IDS = _ORIGINAL_TASK_IDS
        e3_train._polarity_slots = _ORIGINAL_POLARITY_SLOTS

    artifact_dir = Path(result["artifact_dir"])
    _annotate_artifact(artifact_dir, extension_hash, source_hashes)
    result["extension_protocol"] = EXTENSION_PROTOCOL
    result["extension_source_hash"] = extension_hash
    result["original_e3_source_hash"] = source_hashes["e3_source"]
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the single E3 Hotel validation extension")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "e3" / "hotel_extension" / "seed_42",
    )
    parser.add_argument("--device", default=DEVICE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args.output_dir.resolve(), args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
