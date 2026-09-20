"""Versioned runtime repair and recovery entrypoint for E2.

The E2 recipe and the original scheduler are frozen.  This adapter fixes the
runtime contract mismatch that stopped the first queue: ``source_hashes``
contains the E1 dependency hash, while the frozen ``metadata_matches``
callable accepts the three scheduler hashes only.  The adapter verifies the
E1 hash separately, then calls the frozen validator with its actual API.

It also reconciles dead children before handing control back to the frozen
queue.  Completed artifacts are adopted exactly; incomplete dead attempts are
archived with file hashes and marked as interrupted.  A fresh rerun of those
same planned identities requires the explicit ``--fresh-rerun-interrupted``
flag, so a controller crash can never silently select a rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


PROTOCOL = "E2-runtime-recovery-v2"
REVISION = "e2-20260908-r1"


class RecoveryBlocked(RuntimeError):
    """The repair adapter stopped without starting a duplicate child."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def metadata_matches_compat(
    frozen_scheduler: Any,
    metadata: Mapping[str, Any],
    artifact_dir: Path,
    spec: Any,
    *,
    e1_dependency_hash: str | None = None,
    e2_code_hash: str,
    e2_config_hash: str,
    orchestrator_hash: str,
) -> bool:
    """Adapt the four-key state mapping to the frozen validator's API.

    ``e1_dependency_hash`` is checked before the call.  The actual frozen
    callable receives only its three declared keyword arguments.  A missing
    E1 keyword is allowed for the frozen scheduler's other call sites because
    the validator independently checks the artifact's E1 source hash.
    """

    if e1_dependency_hash is not None and e1_dependency_hash != frozen_scheduler.E1_DEPENDENCY_HASH:
        return False
    return frozen_scheduler._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES(
        metadata,
        artifact_dir,
        spec,
        e2_code_hash=e2_code_hash,
        e2_config_hash=e2_config_hash,
        orchestrator_hash=orchestrator_hash,
    )


def install_metadata_compat(frozen_scheduler: Any) -> None:
    """Install the audited adapter in the frozen module's global namespace."""

    if not hasattr(frozen_scheduler, "_E2_RECOVERY_V2_FROZEN_METADATA_MATCHES"):
        frozen_scheduler._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES = frozen_scheduler.metadata_matches

    def adapted(
        metadata: Mapping[str, Any],
        artifact_dir: Path,
        spec: Any,
        *,
        e1_dependency_hash: str | None = None,
        e2_code_hash: str,
        e2_config_hash: str,
        orchestrator_hash: str,
    ) -> bool:
        return metadata_matches_compat(
            frozen_scheduler,
            metadata,
            artifact_dir,
            spec,
            e1_dependency_hash=e1_dependency_hash,
            e2_code_hash=e2_code_hash,
            e2_config_hash=e2_config_hash,
            orchestrator_hash=orchestrator_hash,
        )

    frozen_scheduler.metadata_matches = adapted


@contextlib.contextmanager
def durable_children(frozen_scheduler: Any) -> Iterator[None]:
    """Make runner children independent of an accidental scheduler exit.

    ``start_new_session`` gives every training/cache child its own session and
    process group.  The scheduler still records the real PID and waits for
    completion normally; this only prevents a controller shell/session failure
    from taking all children down with it.
    """

    original = frozen_scheduler.subprocess.Popen

    def durable_popen(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("start_new_session", True)
        return original(*args, **kwargs)

    frozen_scheduler.subprocess.Popen = durable_popen
    try:
        yield
    finally:
        frozen_scheduler.subprocess.Popen = original


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _artifact_matches(
    frozen_scheduler: Any,
    artifact: Path,
    spec: Any,
    hashes: Mapping[str, str],
) -> bool:
    """Use the frozen contract through the versioned adapter."""

    if spec.stage == "stage0_cache":
        summary = frozen_scheduler._load_cache_summary(artifact)
        return summary is not None and metadata_matches_compat(
            frozen_scheduler,
            summary,
            artifact,
            spec,
            **dict(hashes),
        )
    metadata = frozen_scheduler._read_metadata(artifact / "metadata.json")
    return metadata is not None and metadata_matches_compat(
        frozen_scheduler,
        metadata,
        artifact,
        spec,
        **dict(hashes),
    )


def _archive_interrupted(
    output_root: Path,
    archive_root: Path,
    job: Mapping[str, Any],
    spec: Any,
) -> dict[str, Any]:
    """Move a dead incomplete attempt into an immutable recovery archive."""

    source = Path(str(job.get("output_dir", spec.expected_dir)))
    target = archive_root / "interrupted" / spec.slug() / "artifact"
    target.parent.mkdir(parents=True, exist_ok=True)
    source_exists = source.exists() or source.is_symlink()
    if source_exists:
        shutil.move(str(source), str(target))
    log_path = Path(str(job.get("log_path", ""))) if job.get("log_path") else None
    archived_log: str | None = None
    log_hash: str | None = None
    if log_path is not None and log_path.is_file():
        log_target = archive_root / "interrupted" / spec.slug() / "runner.log"
        shutil.copy2(log_path, log_target)
        archived_log = str(log_target)
        log_hash = sha256_file(log_target)
    return {
        "spec": frozen_asdict(spec),
        "original_output_dir": str(source),
        "archive_dir": str(target.parent),
        "artifact_archived": source_exists,
        "artifact_files": _file_manifest(target) if source_exists else [],
        "original_log_path": str(log_path) if log_path else None,
        "archived_log_path": archived_log,
        "archived_log_sha256": log_hash,
        "status": "interrupted",
        "restart_policy": "fresh_rerun_only; no exact resume claim",
    }


def frozen_asdict(spec: Any) -> dict[str, Any]:
    # Avoid importing dataclasses at module import time in the runner path.
    from dataclasses import asdict

    return asdict(spec)


def _new_archive_root(output_root: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = output_root / "recovery" / f"runtime_v2_{stamp}_{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=False)
    return base


def reconcile_stale_jobs_v2(
    frozen_scheduler: Any,
    *,
    repo_root: Path,
    output_root: Path,
    hashes: Mapping[str, str],
    memory_guards: Mapping[str, float],
    allow_fresh_rerun: bool,
) -> dict[str, Any]:
    """Adopt valid dead jobs and archive incomplete dead jobs.

    The caller must hold the orchestration lock.  A live PID always blocks;
    there is no PID-based duplicate launch fallback.
    """

    orchestrator_root = output_root / ".orchestrator"
    store = frozen_scheduler.StateStore(orchestrator_root, hashes, memory_guards=memory_guards)
    jobs = [job for job in store.state.get("jobs", []) if job.get("status") == "running"]
    live = [job for job in jobs if frozen_scheduler._pid_alive(int(job.get("pid", -1)))]
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": now(),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "frozen_scheduler_hash": hashes.get("orchestrator_hash"),
        "frozen_recipe_hash": hashes.get("orchestrator_hash"),
        "e1_dependency_hash": hashes.get("e1_dependency_hash"),
        "runtime_repair_hash": sha256_file(Path(__file__).resolve()),
        "host_pid_checks": [
            {
                "pid": int(job.get("pid", -1)),
                "alive": job in live,
                "spec": job.get("spec"),
            }
            for job in jobs
        ],
        "jobs_seen": len(jobs),
        "completed_adopted": [],
        "interrupted_archived": [],
        "fresh_rerun_authorized": bool(allow_fresh_rerun),
    }
    report_path = orchestrator_root / "recovery_v2_last_preflight.json"
    if live:
        report["status"] = "blocked_live_children"
        report["blocked_live_jobs"] = [job.get("spec") for job in live]
        write_json(report_path, report)
        raise RecoveryBlocked(
            "Live E2 child detected; refusing duplicate launch: "
            + ", ".join(f"pid={job.get('pid')} spec={job.get('spec')}" for job in live)
        )

    archive_root: Path | None = None
    stale_partial = False
    for job in jobs:
        spec = frozen_scheduler.TrialSpec(**(job.get("spec") or {}))
        artifact = Path(str(job.get("output_dir", spec.expected_dir)))
        if artifact.exists() and _artifact_matches(frozen_scheduler, artifact, spec, hashes):
            record = frozen_scheduler._validate_completed_artifact(artifact, spec, hashes=hashes)
            frozen_scheduler._append_approved(store, record)
            frozen_scheduler._mark_job(
                store,
                job,
                "completed",
                artifact_dir=record["artifact_dir"],
                recovered=True,
                recovery_runtime="v2",
            )
            report["completed_adopted"].append(
                {"spec": frozen_asdict(spec), "artifact_dir": record["artifact_dir"]}
            )
            continue

        stale_partial = True
        if archive_root is None:
            archive_root = _new_archive_root(output_root)
        archived = _archive_interrupted(output_root, archive_root, job, spec)
        store.state.setdefault("interrupted_attempts", []).append(archived)
        frozen_scheduler._mark_job(
            store,
            job,
            "interrupted",
            recovery_runtime="v2",
            archive_dir=archived["archive_dir"],
            restart_policy=archived["restart_policy"],
        )
        report["interrupted_archived"].append(archived)

    if stale_partial and not allow_fresh_rerun:
        report["status"] = "blocked_restart_requires_explicit_flag"
        report["archive_root"] = str(archive_root) if archive_root else None
        write_json(report_path, report)
        raise RecoveryBlocked(
            "Dead incomplete E2 attempts were archived; rerun requires "
            "--fresh-rerun-interrupted after reviewing recovery_v2_last_preflight.json"
        )

    report["status"] = "ready_for_fresh_queue"
    report["archive_root"] = str(archive_root) if archive_root else None
    report["fresh_rerun_count"] = len(report["interrupted_archived"])
    store.state["runtime_repair"] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "runtime_repair_hash": report["runtime_repair_hash"],
        "frozen_scheduler_hash": report["frozen_scheduler_hash"],
        "recovered_at": report["created_at"],
        "fresh_rerun_authorized": bool(allow_fresh_rerun),
        "interrupted_count": len(report["interrupted_archived"]),
    }
    store.approved_manifest()
    store.save()
    write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "stage0_teacher", "stage0_cache", "stagea_kd", "stagea_nc", "stagea_ce_replay", "stageb"),
        default="all",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-smoke-gate", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--student-memory-guard-gib", type=float, default=5.0)
    parser.add_argument("--teacher-memory-guard-gib", type=float, default=5.0)
    parser.add_argument("--nc-memory-guard-gib", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--fresh-rerun-interrupted",
        action="store_true",
        help="Explicitly authorize fresh full trials for archived dead partial attempts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    if not parsed.execute:
        print("BLOCKED: recovery v2 requires --execute", file=sys.stderr)
        return 2
    repo_root = parsed.repo_root.resolve()
    output_root = parsed.output_root.resolve()
    from scripts import execute_e2_validation as frozen_scheduler

    # Compute and preserve the frozen recipe hash before installing adapters.
    hashes = frozen_scheduler.source_hashes(repo_root)
    install_metadata_compat(frozen_scheduler)
    memory_guards = {
        "student": float(parsed.student_memory_guard_gib),
        "teacher": float(parsed.teacher_memory_guard_gib),
        "nc": float(parsed.nc_memory_guard_gib),
    }
    if min(memory_guards.values()) <= 0:
        print("BLOCKED: memory guards must be positive", file=sys.stderr)
        return 2

    lock = frozen_scheduler._lock_orchestration(output_root / ".orchestrator")
    try:
        report = reconcile_stale_jobs_v2(
            frozen_scheduler,
            repo_root=repo_root,
            output_root=output_root,
            hashes=hashes,
            memory_guards=memory_guards,
            allow_fresh_rerun=parsed.fresh_rerun_interrupted,
        )
    except (frozen_scheduler.OrchestrationBlocked, RecoveryBlocked) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        lock.close()
        return 2
    finally:
        # The frozen scheduler takes the same lock for its own durable run.
        try:
            lock.close()
        except Exception:
            pass

    namespace = argparse.Namespace(
        stage=parsed.stage,
        repo_root=repo_root,
        output_root=output_root,
        devices=list(parsed.devices),
        poll_seconds=float(parsed.poll_seconds),
        student_memory_guard_gib=memory_guards["student"],
        teacher_memory_guard_gib=memory_guards["teacher"],
        nc_memory_guard_gib=memory_guards["nc"],
        root_smoke_gate=parsed.root_smoke_gate.resolve(),
        execute=True,
    )
    print(
        json.dumps(
            {
                "recovery_report": str(output_root / ".orchestrator" / "recovery_v2_last_preflight.json"),
                "frozen_scheduler_hash": hashes["orchestrator_hash"],
                "runtime_repair_hash": sha256_file(Path(__file__).resolve()),
                "interrupted_archived": len(report["interrupted_archived"]),
                "completed_adopted": len(report["completed_adopted"]),
            },
            sort_keys=True,
        )
    )
    try:
        with durable_children(frozen_scheduler):
            return int(frozen_scheduler.run_orchestration(namespace))
    except frozen_scheduler.OrchestrationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
