#!/usr/bin/env python3
"""Export the exact 0.6B Vietnamese MNTP training stream to HF Parquet.

The source token matrix remains unchanged. Training used the first
``steps * global_batch`` shuffled rows, not every row in the originally
reserved 250M-token range.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "bidirlm_vi_mntp"
SOURCE_MANIFEST = ROOT / "artifacts" / "bidirlm_vi_mntp" / "data_manifest.json"
TRAIN_REPORT = ROOT / "artifacts" / "bidirlm_vi_mntp" / "mntp" / "mntp_report.json"
DEST = ROOT / "releases" / "bidirlm_vi_mntp_250m_dataset"
BATCH_ROWS = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_split(path: Path, token_matrix: np.ndarray, indices: np.ndarray) -> dict:
    schema = pa.schema(
        [
            ("source_sequence_index", pa.int64()),
            ("input_ids", pa.list_(pa.uint32(), 512)),
        ]
    )
    temp = path.with_name(path.name + ".tmp")
    with pq.ParquetWriter(temp, schema=schema, compression="zstd", compression_level=3) as writer:
        for start in range(0, len(indices), BATCH_ROWS):
            selected = np.ascontiguousarray(token_matrix[indices[start : start + BATCH_ROWS]])
            ids = pa.FixedSizeListArray.from_arrays(pa.array(selected.ravel(), type=pa.uint32()), 512)
            table = pa.Table.from_arrays([pa.array(indices[start : start + len(selected)], type=pa.int64()), ids], schema=schema)
            writer.write_table(table, row_group_size=BATCH_ROWS)
    temp.replace(path)

    reader = pq.ParquetFile(path)
    if reader.metadata.num_rows != len(indices):
        raise AssertionError((reader.metadata.num_rows, len(indices)))
    row = 0
    for group in range(reader.num_row_groups):
        chunk = reader.read_row_group(group)
        got_indices = chunk.column("source_sequence_index").to_numpy()
        got_tokens = chunk.column("input_ids").combine_chunks().values.to_numpy().reshape(-1, 512)
        n = len(got_indices)
        if not np.array_equal(got_indices, indices[row : row + n]):
            raise AssertionError(f"source indices differ at row {row}")
        if not np.array_equal(got_tokens, token_matrix[indices[row : row + n]]):
            raise AssertionError(f"token IDs differ at row {row}")
        row += n
    if row != len(indices):
        raise AssertionError((row, len(indices)))
    return {"rows": int(row), "tokens": int(row * 512), "bytes": path.stat().st_size, "sha256": sha256(path), "full_row_parity": True}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    source_meta = json.loads(SOURCE_MANIFEST.read_text())
    training = json.loads(TRAIN_REPORT.read_text())
    tokens_path = SOURCE / "tokens.npy"
    order_path = SOURCE / "order.npy"
    heldout_path = SOURCE / "heldout_indices.npy"
    for name, path in (("tokens.npy", tokens_path), ("order.npy", order_path), ("heldout_indices.npy", heldout_path)):
        actual = sha256(path)
        expected = source_meta["files"][name]["sha256"]
        if actual != expected:
            raise AssertionError(f"source hash mismatch: {name}: {actual} != {expected}")
    matrix = np.load(tokens_path, mmap_mode="r")
    order = np.load(order_path, mmap_mode="r")
    heldout = np.load(heldout_path)
    report_data = training["data"]
    n_train = int(report_data["train_sequences"])
    if n_train != int(training["hyperparameters"]["steps"] * training["hyperparameters"]["global_batch"]):
        raise AssertionError("actual training step count is inconsistent")
    train_indices = np.asarray(order[:n_train])
    if matrix.shape != (source_meta["dataset"]["sequences"], 512) or train_indices.max() >= matrix.shape[0]:
        raise AssertionError("source matrix or permutation mismatch")
    if len(set(train_indices.tolist())) != n_train or len(set(heldout.tolist())) != len(heldout):
        raise AssertionError("duplicate sequence indices")
    if set(train_indices.tolist()) & set(heldout.tolist()):
        raise AssertionError("train/heldout overlap")
    if len(heldout) != int(report_data["heldout_sequences"]):
        raise AssertionError("heldout count mismatch")
    if not np.array_equal(heldout, order[source_meta["dataset"]["train_sequences"] : source_meta["dataset"]["train_sequences"] + len(heldout)]):
        raise AssertionError("heldout source order mismatch")
    for start in range(0, len(train_indices), BATCH_ROWS):
        if int(matrix[train_indices[start : start + BATCH_ROWS]].max()) >= int(training["model"]["config"]["vocab_size"]):
            raise AssertionError(f"out-of-vocabulary token IDs near row {start}")

    print(f"Exporting {n_train} train sequences and {len(heldout)} heldout sequences", flush=True)
    files = {
        "train.parquet": export_split(DEST / "train.parquet", matrix, train_indices),
        "validation.parquet": export_split(DEST / "validation.parquet", matrix, heldout),
    }
    metadata = {
        "description": "Exact token sequences consumed by BidirLM-0.6B Vietnamese MNTP stage 1",
        "source_dataset": "HPLT/HPLT3.0 vie_Latn; project general pool v3, bins 7-9",
        "source_dataset_revision": "3394d6ba8dae4da834e3b11771daf95028a960b1",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "training_report_sha256": sha256(TRAIN_REPORT),
        "source_token_matrix_sha256": source_meta["files"]["tokens.npy"]["sha256"],
        "source_order_sha256": source_meta["files"]["order.npy"]["sha256"],
        "tokenizer": source_meta["tokenizer"],
        "train_steps": training["hyperparameters"]["steps"],
        "global_batch": training["hyperparameters"]["global_batch"],
        "sequence_length": 512,
        "train_selection": "order[:7629*64] from full tokenized pool",
        "heldout_selection": "order[488281:490281] from full tokenized pool; no train overlap",
        "source_sequences_reserved_for_train": source_meta["dataset"]["train_sequences"],
        "source_train_sequences_not_consumed": int(source_meta["dataset"]["train_sequences"] - n_train),
        "files": files,
    }
    (DEST / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(files, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
