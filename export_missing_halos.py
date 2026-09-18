#!/usr/bin/env python3
"""Export galaxy-centered regions from TNG50-1 parent halos in the TNG Lab."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass
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
DEFAULT_OUTPUT_DIR = Path("missing_galaxy_cutouts_300kpc")
DEFAULT_SIZE_KPC = 300.0
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


@dataclass(frozen=True, order=True)
class Target:
    snapshot: int
    subhalo: int
    parent_halo: int

    @property
    def key(self) -> str:
        return f"{self.snapshot}:{self.subhalo}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_targets(path: Path) -> list[Target]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        targets = [
            Target(
                snapshot=int(row["snapshot"]),
                subhalo=int(row["subhalo"]),
                parent_halo=int(row["parent_halo"]),
            )
            for row in rows
        ]
    keys = [(target.snapshot, target.subhalo) for target in targets]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate snapshot/subhalo entries in {path}")
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


def snapshot_geometry(source_path: Path) -> tuple[float, float, float]:
    with h5py.File(source_path, "r") as source:
        header = source["Header"].attrs
        return float(header["Time"]), float(header["HubbleParam"]), float(header["BoxSize"])


def load_halo_field(il, base_path: Path, target: Target, particle_type: int, field: str):
    payload = il.snapshot.loadHalo(
        str(base_path),
        target.snapshot,
        target.parent_halo,
        particle_type,
        fields=[field],
    )
    if isinstance(payload, dict):
        count = int(payload.get("count", 0))
        if count == 0:
            if field == "Coordinates":
                return np.empty((0, 3), dtype=np.float32)
            return np.empty(0, dtype=np.float32)
        if field not in payload:
            raise KeyError(
                f"illustris_python did not return {field} for snapshot "
                f"{target.snapshot}, halo {target.parent_halo}, PartType{particle_type}"
            )
        return np.asarray(payload[field])
    return np.asarray(payload)


def periodic_cube_mask(
    coordinates: np.ndarray,
    center: np.ndarray,
    box_size: float,
    half_width_comoving_kpc_over_h: float,
) -> np.ndarray:
    delta = coordinates - center
    delta -= box_size * np.rint(delta / box_size)
    return np.all(np.abs(delta) <= half_width_comoving_kpc_over_h, axis=1)


def write_dataset_safely(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write through a temporary dataset so an interrupted write is resumable."""
    temporary_name = f"__incomplete__{name}"
    array = np.asarray(data)
    if name in group:
        if group[name].shape != array.shape:
            raise RuntimeError(
                f"Existing partial dataset {group[name].name} has shape "
                f"{group[name].shape}, expected {array.shape}"
            )
        return
    if temporary_name in group:
        del group[temporary_name]

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


def validate_existing_cutout(
    path: Path,
    target: Target,
    size_kpc: float,
    *,
    allow_uninitialized: bool = False,
) -> None:
    with h5py.File(path, "r") as source:
        actual_snapshot = int(source.attrs.get("SnapshotNumber", -1))
        actual_subhalo = int(source.attrs.get("SubhaloID", -1))
        actual_parent = int(source.attrs.get("ParentHaloID", -1))
        actual_size = float(source.attrs.get("CutoutSizePhysicalKpc", -1.0))
        present = set(source.attrs)
    if allow_uninitialized:
        partial_mismatch = (
            ("SnapshotNumber" in present and actual_snapshot != target.snapshot)
            or ("SubhaloID" in present and actual_subhalo != target.subhalo)
            or ("ParentHaloID" in present and actual_parent != target.parent_halo)
            or (
                "CutoutSizePhysicalKpc" in present
                and not np.isclose(actual_size, size_kpc)
            )
        )
        if not partial_mismatch:
            return
    if (
        actual_snapshot != target.snapshot
        or actual_subhalo != target.subhalo
        or actual_parent != target.parent_halo
        or not np.isclose(actual_size, size_kpc)
    ):
        raise RuntimeError(
            f"{path} exists but is not the requested {size_kpc:g} kpc cutout "
            f"for subhalo {target.subhalo}. Move it aside or choose another --output-dir."
        )


def export_target(
    base_path: Path,
    output_dir: Path,
    simulation: str,
    target: Target,
    size_kpc: float,
) -> Path:
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

    final_path = output_dir / (
        f"cutout_{simulation}_{target.snapshot}_{target.subhalo}_"
        f"{target.parent_halo}.hdf5"
    )
    partial_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists():
        validate_existing_cutout(final_path, target, size_kpc)
        print(f"SKIP existing {final_path}", flush=True)
        return final_path

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = snapshot_piece(base_path, target.snapshot)
    scale_factor, hubble_parameter, box_size = snapshot_geometry(source_path)
    half_width = 0.5 * size_kpc * hubble_parameter / scale_factor

    subhalo_record = il.groupcat.loadSingle(
        str(base_path), target.snapshot, subhaloID=target.subhalo
    )
    center = np.asarray(subhalo_record["SubhaloPos"], dtype=np.float64)
    actual_parent = int(subhalo_record["SubhaloGrNr"])
    if actual_parent != target.parent_halo:
        raise ValueError(
            f"Target manifest says parent halo {target.parent_halo}, but the group "
            f"catalog reports {actual_parent} for snapshot {target.snapshot}, "
            f"subhalo {target.subhalo}"
        )

    if partial_path.exists():
        validate_existing_cutout(
            partial_path, target, size_kpc, allow_uninitialized=True
        )

    counts = np.zeros(6, dtype=np.uint64)
    with h5py.File(partial_path, "a") as destination:
        destination.attrs["Exporter"] = "tng50-halo-cutout-exporter"
        destination.attrs["SnapshotNumber"] = target.snapshot
        destination.attrs["SubhaloID"] = target.subhalo
        destination.attrs["ParentHaloID"] = target.parent_halo
        destination.attrs["CutoutCenterComovingKpcOverH"] = center
        destination.attrs["CutoutSizePhysicalKpc"] = size_kpc
        destination.attrs["CutoutGeometry"] = "axis-aligned cube"
        destination.attrs["CoordinateUnits"] = "comoving kpc/h"
        destination.flush()
        copy_snapshot_metadata(source_path, destination)

        for particle_type, fields in PARTICLE_FIELDS.items():
            label = f"PartType{particle_type}"
            print(f"  loading {label} coordinates from parent halo", flush=True)
            coordinates = load_halo_field(
                il, base_path, target, particle_type, "Coordinates"
            )
            mask = periodic_cube_mask(coordinates, center, box_size, half_width)
            selected_count = int(np.count_nonzero(mask))
            counts[particle_type] = selected_count
            print(
                f"    selected {selected_count:,}/{len(coordinates):,} particles "
                f"inside {size_kpc:g} physical kpc cube",
                flush=True,
            )
            if selected_count == 0:
                continue

            group = destination.require_group(label)
            write_dataset_safely(group, "Coordinates", coordinates[mask])
            for field in fields:
                if field == "Coordinates":
                    continue
                print(f"    loading and writing {field}", flush=True)
                values = load_halo_field(il, base_path, target, particle_type, field)
                if len(values) != len(coordinates):
                    raise RuntimeError(
                        f"{label}/{field} has {len(values)} rows, but Coordinates "
                        f"has {len(coordinates)}"
                    )
                write_dataset_safely(group, field, values[mask])

        set_particle_counts(destination, counts)
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
        "--size-kpc",
        type=float,
        default=DEFAULT_SIZE_KPC,
        help="Physical side length of the galaxy-centered cube (default: 300)",
    )
    parser.add_argument(
        "--only",
        nargs=2,
        type=int,
        metavar=("SNAPSHOT", "SUBHALO"),
        action="append",
        help="Export only this galaxy; may be repeated",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N selected targets")
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependencies_available = np is not None and h5py is not None
    if args.size_kpc <= 0:
        raise ValueError("--size-kpc must be positive")

    targets = load_targets(args.targets)
    if args.only:
        by_key = {(target.snapshot, target.subhalo): target for target in targets}
        requested_keys = [tuple(pair) for pair in args.only]
        unknown = set(requested_keys) - set(by_key)
        if unknown:
            raise ValueError(
                f"Requested snapshot/subhalo pairs are not in {args.targets}: "
                f"{sorted(unknown)}"
            )
        targets = [by_key[key] for key in requested_keys]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        targets = targets[: args.limit]

    if args.list_targets or args.dry_run:
        for target in targets:
            print(f"{target.snapshot},{target.subhalo},{target.parent_halo}")
        parent_count = len({(target.snapshot, target.parent_halo) for target in targets})
        print(
            f"{len(targets)} galaxies in {parent_count} unique parent halos",
            file=sys.stderr,
        )
        return 0

    if not dependencies_available:
        raise RuntimeError(
            "h5py and numpy are required for export. They are preinstalled in "
            "the IllustrisTNG Lab; elsewhere, run: pip install -r requirements.txt"
        )

    base_path = locate_base_path(args.simulation, args.base_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "generation_status.json"
    status = load_status(status_path)

    print(f"Simulation data: {base_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Targets: {len(targets)}", flush=True)
    print(f"Cutout: {args.size_kpc:g} physical kpc cube", flush=True)

    failures = 0
    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] snapshot={target.snapshot} "
            f"subhalo={target.subhalo} parent_halo={target.parent_halo}",
            flush=True,
        )
        try:
            result = export_target(
                base_path, output_dir, args.simulation, target, args.size_kpc
            )
            status["completed"][target.key] = {
                "path": str(result),
                "size_kpc": args.size_kpc,
                "updated_at": utc_now(),
            }
            status["failed"].pop(target.key, None)
        except Exception as exc:
            failures += 1
            status["failed"][target.key] = {
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": utc_now(),
            }
            print(f"FAILED {target.key}: {exc}", file=sys.stderr, flush=True)
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
