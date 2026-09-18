#!/usr/bin/env python3
"""Export selected TNG50-1 parent halos from the IllustrisTNG Lab filesystem."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import h5py
    import numpy as np
except ImportError:  # Allow --list-targets to work outside the TNG Lab.
    h5py = None
    np = None


DEFAULT_SIMULATION = "TNG50-1"
DEFAULT_OUTPUT_DIR = Path("missing_halo_cutouts")
TARGETS_FILE = Path(__file__).with_name("targets.csv")

PARTICLE_FIELDS = {
    0: (
        "Coordinates",
        "Masses",
        "GFM_Metallicity",
        "Density",
        "ElectronAbundance",
        "InternalEnergy",
        "StarFormationRate",
    ),
    4: (
        "Coordinates",
        "GFM_StellarFormationTime",
        "GFM_InitialMass",
        "GFM_Metallicity",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_targets(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        targets = [(int(row["snapshot"]), int(row["halo"])) for row in rows]
    if len(targets) != len(set(targets)):
        raise ValueError(f"Duplicate snapshot/halo entries in {path}")
    return targets


def locate_base_path(simulation: str, explicit: str | None) -> Path:
    candidates: Iterable[Path]
    if explicit:
        candidates = (Path(explicit).expanduser(),)
    else:
        candidates = (
            Path("sims.TNG") / simulation / "output",
            Path.home() / "sims.TNG" / simulation / "output",
            Path("/home/jovyan/sims.TNG") / simulation / "output",
        )

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the mounted simulation output. Checked:\n  "
        + "\n  ".join(checked)
        + "\nPass it explicitly with --base-path."
    )


def snapshot_piece(base_path: Path, snapshot: int) -> Path:
    path = base_path / f"snapdir_{snapshot:03d}" / f"snap_{snapshot:03d}.0.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot header file not found: {path}")
    return path


def copy_snapshot_metadata(source_path: Path, destination: h5py.File) -> None:
    with h5py.File(source_path, "r") as source:
        for group_name in ("Header", "Parameters", "Configuration"):
            if group_name not in source:
                continue
            target_group = destination.require_group(group_name)
            for key, value in source[group_name].attrs.items():
                target_group.attrs[key] = value


def write_dataset_safely(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write through a temporary dataset so an interrupted write is resumable."""
    temporary_name = f"__incomplete__{name}"
    if name in group:
        return
    if temporary_name in group:
        del group[temporary_name]

    array = np.asarray(data)
    options = {}
    if array.size > 0:
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(temporary_name, data=array, **options)
    group.file.flush()
    group.move(temporary_name, name)
    group.file.flush()


def set_particle_counts(destination: h5py.File, counts: np.ndarray) -> None:
    header = destination.require_group("Header")
    totals = counts.astype(np.uint64)
    header.attrs["NumPart_ThisFile"] = (totals & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    header.attrs["NumPart_Total"] = (totals & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    header.attrs["NumPart_Total_HighWord"] = (totals >> np.uint64(32)).astype(np.uint32)
    header.attrs["NumFilesPerSnapshot"] = np.int32(1)


def export_halo(base_path: Path, output_dir: Path, simulation: str, snapshot: int, halo: int) -> Path:
    if h5py is None or np is None:
        raise RuntimeError(
            "h5py and numpy are required for export. They are preinstalled in "
            "the IllustrisTNG Lab; elsewhere, run: pip install -r requirements.txt"
        )
    try:
        import illustris_python as il
    except ImportError as exc:
        raise RuntimeError(
            "illustris_python is unavailable. Run this in the IllustrisTNG Lab, "
            "or install the official illustris_python package."
        ) from exc

    final_path = output_dir / f"cutout_{simulation}_{snapshot}_halo_{halo}.hdf5"
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        print(f"SKIP existing {final_path}", flush=True)
        return final_path

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(6, dtype=np.uint64)

    with h5py.File(partial_path, "a") as destination:
        copy_snapshot_metadata(snapshot_piece(base_path, snapshot), destination)

        for particle_type, fields in PARTICLE_FIELDS.items():
            print(
                f"  loading snapshot={snapshot} halo={halo} PartType{particle_type}",
                flush=True,
            )
            payload = il.snapshot.loadHalo(
                str(base_path), snapshot, halo, particle_type, fields=list(fields)
            )
            count = int(payload.get("count", 0))
            counts[particle_type] = count
            if count == 0:
                continue

            group = destination.require_group(f"PartType{particle_type}")
            for field in fields:
                if field not in payload:
                    raise KeyError(
                        f"illustris_python did not return {field} for snapshot "
                        f"{snapshot}, halo {halo}, PartType{particle_type}"
                    )
                print(f"    writing {field}: {np.asarray(payload[field]).shape}", flush=True)
                write_dataset_safely(group, field, payload[field])

        set_particle_counts(destination, counts)
        destination.attrs["Exporter"] = "tng50-halo-cutout-exporter"
        destination.attrs["SnapshotNumber"] = snapshot
        destination.attrs["ParentHaloID"] = halo
        destination.flush()

    os.replace(partial_path, final_path)
    print(f"DONE {final_path}", flush=True)
    return final_path


def load_status(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument("--base-path", help="Path ending in TNG50-1/output")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", type=Path, default=TARGETS_FILE)
    parser.add_argument(
        "--only",
        nargs=2,
        type=int,
        metavar=("SNAPSHOT", "HALO"),
        action="append",
        help="Export only this pair; may be repeated",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N selected targets")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = load_targets(args.targets)

    if args.only:
        requested = [tuple(pair) for pair in args.only]
        unknown = set(requested) - set(targets)
        if unknown:
            raise ValueError(f"Requested pairs are not in {args.targets}: {sorted(unknown)}")
        targets = requested
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        targets = targets[: args.limit]

    if args.list_targets or args.dry_run:
        for snapshot, halo in targets:
            print(f"{snapshot},{halo}")
        print(f"{len(targets)} unique parent halos", file=sys.stderr)
        if args.list_targets or args.dry_run:
            return 0

    base_path = locate_base_path(args.simulation, args.base_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "generation_status.json"
    status = load_status(status_path)

    print(f"Simulation data: {base_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Targets: {len(targets)}", flush=True)

    failures = 0
    for index, (snapshot, halo) in enumerate(targets, start=1):
        key = f"{snapshot}:{halo}"
        print(f"[{index}/{len(targets)}] snapshot={snapshot} halo={halo}", flush=True)
        try:
            result = export_halo(
                base_path, output_dir, args.simulation, snapshot, halo
            )
            status["completed"][key] = {"path": str(result), "updated_at": utc_now()}
            status["failed"].pop(key, None)
        except Exception as exc:
            failures += 1
            status["failed"][key] = {
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            }
            print(f"FAILED {key}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            if args.fail_fast:
                save_status(status_path, status)
                raise
        finally:
            save_status(status_path, status)

    print(
        f"Finished: {len(targets) - failures} succeeded/skipped, {failures} failed. "
        f"Status: {status_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
